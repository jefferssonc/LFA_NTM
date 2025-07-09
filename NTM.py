import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import re
from tqdm.auto import tqdm
import warnings
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

TOKENS = ['(', ')', '¬', '∧', '∨', '→', '↔', 'A','B','C','D','E', ',', '=', 'True','False']
tok2idx = {t:i+1 for i,t in enumerate(TOKENS)}
tok2idx['<PAD>'] = 0

def tokenize(expr, vals):
    s = expr + ',' + vals
    pattern = r'¬|∧|∨|→|↔|\w+|[(),=]'
    toks = re.findall(pattern, s)
    unknown_tokens = [t for t in toks if t not in tok2idx]
    if unknown_tokens:
        raise ValueError(f"Tokens desconhecidos encontrados: {set(unknown_tokens)}")
    return [tok2idx[t] for t in toks]

class LogicDataset(Dataset):
    def __init__(self, path, max_len=64):
        df = pd.read_csv(path)
        required_columns = ['expressao', 'valores', 'resultado']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Coluna obrigatória '{col}' ausente no CSV.")
        self.X, self.Y, self.ops = [], [], []

        for idx, r in df.iterrows():
            expr = r['expressao']
            vals = r['valores']
            result = str(r['resultado']).strip().lower()

            if result not in ('true', 'false'):
                warnings.warn(f"Linha {idx}: resultado inválido '{result}'. Pulando.")
                continue

            try:
                seq = tokenize(expr, vals)
            except Exception as e:
                warnings.warn(f"Linha {idx}: erro na tokenização: {e}. Pulando.")
                continue

            if len(seq) > max_len:
                seq = seq[:max_len]
            else:
                seq += [0] * (max_len - len(seq))
            self.X.append(seq)
            self.Y.append(1.0 if result == 'true' else 0.0)
            op_count = len(re.findall(r'[¬∧∨→↔]', expr))
            self.ops.append(op_count)

        self.X = torch.tensor(self.X, dtype=torch.long)
        self.Y = torch.tensor(self.Y, dtype=torch.float)
        self.ops = torch.tensor(self.ops, dtype=torch.long)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.ops[i]

def make_curriculum_splits(ds):
    idx_easy = (ds.ops <= 2).nonzero().flatten().tolist()
    idx_med = ((ds.ops >= 3) & (ds.ops <= 4)).nonzero().flatten().tolist()
    idx_hard = (ds.ops >= 5).nonzero().flatten().tolist()
    return idx_easy, idx_med, idx_hard

class NTMCell(nn.Module):
    def __init__(self, inp_size, ctrl_size, N, D, heads, dropout=0.1):
        super().__init__()
        self.ctrl_size, self.N, self.D, self.heads = ctrl_size, N, D, heads
        self.lstm1 = nn.LSTMCell(inp_size + heads*D, ctrl_size)
        self.lstm2 = nn.LSTMCell(ctrl_size, ctrl_size)
        self.dropout = nn.Dropout(dropout)
        self.read_k = weight_norm(nn.Linear(ctrl_size, heads*D))
        self.write_k = weight_norm(nn.Linear(ctrl_size, heads*D))
        self.read_b = weight_norm(nn.Linear(ctrl_size, heads))
        self.write_b = weight_norm(nn.Linear(ctrl_size, heads))
        self.erase = weight_norm(nn.Linear(ctrl_size, heads*D))
        self.add = weight_norm(nn.Linear(ctrl_size, heads*D))
        self.shift = weight_norm(nn.Linear(ctrl_size, heads*3))
        self.gamma = weight_norm(nn.Linear(ctrl_size, heads))

    def content_weights(self, M, K, beta):
        Mn = F.normalize(M, dim=-1)
        Kn = F.normalize(K, dim=-1)
        cos = torch.einsum('bnd,bhd->bhn', Mn, Kn)
        return F.softmax(beta.unsqueeze(-1) * cos, dim=-1)

    def shift_and_sharpen(self, w, shifts, gamma):
        rolls = torch.stack([
            torch.roll(w, -1, -1),
            w,
            torch.roll(w, +1, -1)
        ], dim=-1)
        w_s = torch.einsum('bhnk,bhk->bhn', rolls, F.softmax(shifts, -1))
        g = 1 + F.softplus(gamma).unsqueeze(-1)
        w_g = w_s ** g
        return w_g / (w_g.sum(-1, keepdim=True) + 1e-12)

    def write_memory(self, M, w, erase, add):
        e = erase.view(-1,self.heads,1,self.D)
        a = add.view(-1,self.heads,1,self.D)
        w_ = w.view(-1,self.heads,self.N,1)
        M1 = M * (1 - torch.sum(w_*e, dim=1))
        M2 = torch.sum(w_*a, dim=1)
        return M1 + M2

    def forward(self, x, prev):
        (h1,c1,h2,c2), (w_prev, r_prev, M_prev) = prev
        B = x.size(0)
        inp = torch.cat([x, r_prev.view(B,-1)], dim=-1)
        h1, c1 = self.lstm1(inp, (h1, c1))
        h1 = self.dropout(h1)
        h2, c2 = self.lstm2(h1, (h2, c2))
        h2 = self.dropout(h2)

        RK = self.read_k(h2).view(B,self.heads,self.D)
        WK = self.write_k(h2).view(B,self.heads,self.D)
        Br = F.softplus(self.read_b(h2))
        Bw = F.softplus(self.write_b(h2))
        erase = torch.sigmoid(self.erase(h2)).view(B,self.heads,self.D)
        addv = torch.tanh(self.add(h2)).view(B,self.heads,self.D)
        shifts = self.shift(h2).view(B,self.heads,3)
        gamma = self.gamma(h2)

        w_w = self.content_weights(M_prev, WK, Bw)
        M = self.write_memory(M_prev, w_w, erase, addv)
        w_r = self.content_weights(M, RK, Br)
        w_r = self.shift_and_sharpen(w_r, shifts, gamma)
        r = torch.einsum('bhn,bnd->bhd', w_r, M)
        new_state = ((h1,c1,h2,c2), (w_r, r, M))
        return new_state, r

class NTM(nn.Module):
    def __init__(self, vocab_sz, emb_sz, ctrl_sz, N, D, heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, emb_sz, padding_idx=0)
        self.cell = NTMCell(emb_sz, ctrl_sz, N, D, heads)
        self.out = nn.Linear(ctrl_sz + heads*D, 1)

    def forward(self, x):
        B, T = x.size()
        emb = self.embed(x)
        h1 = torch.zeros(B, self.cell.ctrl_size, device=device)
        c1 = torch.zeros_like(h1)
        h2 = torch.zeros(B, self.cell.ctrl_size, device=device)
        c2 = torch.zeros_like(h2)
        w0 = torch.zeros(B, self.cell.heads, self.cell.N, device=device)
        r0 = torch.zeros(B, self.cell.heads, self.cell.D, device=device)
        M0 = torch.zeros(B, self.cell.N, self.cell.D, device=device)
        state = ((h1,c1,h2,c2), (w0,r0,M0))

        for t in range(T):
            state, r = self.cell(emb[:,t,:], state)
        (_,_,h2,_), (_,rT,_) = state
        feat = torch.cat([h2, rT.view(B,-1)], dim=-1)
        logits = self.out(feat)
        return logits.squeeze(-1)

stage_metrics = {}  # Global dictionary to store metrics

def evaluate_metrics(model, loader, stage_name=''):
    model.eval()
    all_preds, all_probs, all_targets = [], [], []

    with torch.no_grad():
        for X, Y, _ in loader:
            X, Y = X.to(device), Y.to(device)
            logits = model(X)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(Y.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = float('nan')
    cm = confusion_matrix(all_targets, all_preds)

    # Store metrics
    stage_metrics[stage_name] = {
        'Accuracy': acc,
        'Recall': recall,
        'F1-Score': f1,
        'AUC': auc
    }

    # Plot confusion matrix with blue colors
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')  # Use blue color map
    plt.title(f'Confusion Matrix: {stage_name}')
    plt.savefig(f'confusion_matrix_{stage_name.replace(" ", "_")}.png', bbox_inches='tight')
    plt.close()

    return acc


def plot_stage_metrics(stage_metrics):
    stages = list(stage_metrics.keys())
    metrics = list(next(iter(stage_metrics.values())).keys())  # ['Accuracy', 'Recall', ...]

    for metric in metrics:
        values = [stage_metrics[stage][metric] for stage in stages]
        plt.figure(figsize=(6,4))
        plt.plot(stages, values, marker='o', linestyle='-', color='blue')
        plt.title(f'{metric} por Estágio')
        plt.xlabel('Estágio')
        plt.ylabel(metric)
        plt.ylim(0, 1.05)
        plt.grid(True)
        plt.savefig(f'metric_{metric.replace(" ", "_")}.png', bbox_inches='tight')
        plt.close()


# Hiperparâmetros
EMB, CTRL = 128, 256
N, D, H = 64, 32, 4
BS = 128
E1, E2, E3 = 15, 8, 8
LR = 5e-4

model = NTM(len(tok2idx), EMB, CTRL, N, D, H).to(device)
crit = nn.BCEWithLogitsLoss()
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=E1+E2+E3, eta_min=1e-6)

full_ds = LogicDataset('dataset_50000_balanceado.csv')
idx_easy, idx_med, idx_hard = make_curriculum_splits(full_ds)
loader_easy = DataLoader(Subset(full_ds, idx_easy), batch_size=BS, shuffle=True, drop_last=True)
loader_med = DataLoader(Subset(full_ds, idx_med), batch_size=BS, shuffle=True, drop_last=True)
loader_full = DataLoader(full_ds, batch_size=BS, shuffle=True, drop_last=True)
loader_eval = DataLoader(full_ds, batch_size=BS, shuffle=False)

print('\n=== STAGE 1: easy ===')
for ep in range(1, E1+1):
    model.train()
    loop = tqdm(loader_easy, desc=f'S1 Ep{ep}')
    for X,Y,_ in loop:
        X,Y = X.to(device), Y.to(device)
        logits = model(X)
        loss = crit(logits, Y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        loop.set_postfix(loss=loss.item())

# Avaliação ao final do estágio 1
evaluate_metrics(model, loader_easy, 'Stage 1 - Easy')




print('\n=== STAGE 2: medium ===')
for p in model.cell.parameters():
    p.requires_grad = False
for p in model.out.parameters():
    p.requires_grad = True

for ep in range(1, E2+1):
    model.train()
    loop = tqdm(loader_med, desc=f'S2 Ep{ep}')
    for X,Y,_ in loop:
        X,Y = X.to(device), Y.to(device)
        logits = model(X)
        loss = crit(logits, Y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.out.parameters(), 1.0)
        opt.step()
        sched.step()
        loop.set_postfix(loss=loss.item())

# Avaliação ao final do estágio 2
evaluate_metrics(model, loader_med, 'Stage 2 - Medium')




print('\n=== STAGE 3: full ===')
for p in model.parameters():
    p.requires_grad = True

for ep in range(1, E3+1):
    model.train()
    loop = tqdm(loader_full, desc=f'S3 Ep{ep}')
    for X,Y,_ in loop:
        X,Y = X.to(device), Y.to(device)
        logits = model(X)
        loss = crit(logits, Y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        loop.set_postfix(loss=loss.item())

# Avaliação ao final do estágio 3
evaluate_metrics(model, loader_full, 'Stage 3 - Full')




print('\n=== Final Evaluation ===')
evaluate_metrics(model, loader_eval, 'Final Evaluation')

plot_stage_metrics(stage_metrics)


