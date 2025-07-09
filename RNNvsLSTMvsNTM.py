import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import pandas as pd
import re
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import seaborn as sns


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Tokenizer
TOKENS = ['(', ')', '¬', '∧', '∨', '→', '↔',
          'A','B','C','D','E', ',', '=', 'True','False']
tok2idx = {t:i+1 for i,t in enumerate(TOKENS)}
tok2idx['<PAD>'] = 0

def tokenize(expr, vals):
    s = expr + ',' + vals
    pattern = r'¬|∧|∨|→|↔|\w+|[(),=]'
    toks = re.findall(pattern, s)
    return [tok2idx[t] for t in toks if t in tok2idx]

class LogicDataset(Dataset):
    def __init__(self, path, max_len=64):
        df = pd.read_csv(path)
        self.X, self.Y, self.ops = [], [], []
        for _, r in df.iterrows():
            seq = tokenize(r['expressao'], r['valores'])
            seq = seq[:max_len] + [0]*(max_len - len(seq))
            self.X.append(seq)
            self.Y.append(int(r['resultado']))
            self.ops.append(len(re.findall(r'[¬∧∨→↔]', r['expressao'])))
        self.X = torch.tensor(self.X, dtype=torch.long)
        self.Y = torch.tensor(self.Y, dtype=torch.float)
        self.ops = torch.tensor(self.ops, dtype=torch.long)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.ops[i]

def evaluate(model, loader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for X, Y, _ in loader:
            X, Y = X.to(device), Y.to(device)
            logits = model(X)
            pred = (logits > 0).float()
            correct += (pred == Y).sum().item()
            total += Y.size(0)
    return correct / total

def plot_metrics(metrics, stage):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(metrics['loss'], label='Loss')
    plt.xlabel('Época'); plt.ylabel('Loss'); plt.title(f'Loss - {stage}')
    plt.grid(); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(metrics['acc'], label='Acc')
    plt.xlabel('Época'); plt.ylabel('Acurácia'); plt.title(f'Acurácia - {stage}')
    plt.grid(); plt.legend()
    plt.tight_layout()
    plt.show()

def train_stage(model, loader, optimizer, scheduler, criterion, epochs, stage_name, early_stop_patience=3):
    best_acc = 0
    patience = 0
    metrics = {'loss': [], 'acc': []}
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0
        loop = tqdm(loader, desc=f'{stage_name} Ep{ep}')
        for X, Y, _ in loop:
            X, Y = X.to(device), Y.to(device)
            logits = model(X)
            loss = criterion(logits, Y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        scheduler.step()
        avg_loss = total_loss / len(loader)
        acc = evaluate(model, loader)
        metrics['loss'].append(avg_loss)
        metrics['acc'].append(acc)
        print(f' → {stage_name} Ep{ep} acc={acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f' → Early stopping at epoch {ep} (best acc={best_acc:.4f})')
                break
    plot_metrics(metrics, stage_name)



class SimpleRNN(nn.Module):
    def __init__(self, vocab_sz, emb_sz, hidden_sz):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, emb_sz, padding_idx=0)
        self.rnn   = nn.RNN(emb_sz, hidden_sz, batch_first=True)
        self.out   = nn.Linear(hidden_sz, 1)
    def forward(self, x):
        emb = self.embed(x)
        _, h_n = self.rnn(emb)
        logits = self.out(h_n.squeeze(0))
        return logits.squeeze(-1)

class LSTMModel(nn.Module):
    def __init__(self, vocab_sz, emb_sz, hidden_sz):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, emb_sz, padding_idx=0)
        self.lstm  = nn.LSTM(emb_sz, hidden_sz, batch_first=True)
        self.out   = nn.Linear(hidden_sz, 1)
    def forward(self, x):
        emb = self.embed(x)
        _, (h_n, _) = self.lstm(emb)
        logits = self.out(h_n.squeeze(0))
        return logits.squeeze(-1)

class NTMCell(nn.Module):
    def __init__(self, inp_size, ctrl_size, N, D, heads, dropout=0.1):
        super().__init__()
        self.ctrl_size, self.N, self.D, self.heads = ctrl_size, N, D, heads
        self.lstm1   = nn.LSTMCell(inp_size + heads*D, ctrl_size)
        self.lstm2   = nn.LSTMCell(ctrl_size, ctrl_size)
        self.dropout = nn.Dropout(dropout)
        self.read_k  = weight_norm(nn.Linear(ctrl_size, heads*D))
        self.write_k = weight_norm(nn.Linear(ctrl_size, heads*D))
        self.read_b  = weight_norm(nn.Linear(ctrl_size, heads))
        self.write_b = weight_norm(nn.Linear(ctrl_size, heads))
        self.erase   = weight_norm(nn.Linear(ctrl_size, heads*D))
        self.add     = weight_norm(nn.Linear(ctrl_size, heads*D))
        self.shift   = weight_norm(nn.Linear(ctrl_size, heads*3))
        self.gamma   = weight_norm(nn.Linear(ctrl_size, heads))

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
        return ((h1,c1,h2,c2), (w_r, r, M)), r

class NTM(nn.Module):
    def __init__(self, vocab_sz, emb_sz, ctrl_sz, N, D, heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_sz, emb_sz, padding_idx=0)
        self.cell  = NTMCell(emb_sz, ctrl_sz, N, D, heads)
        self.out   = nn.Linear(ctrl_sz + heads*D, 1)
    def forward(self, x):
        B,T = x.size()
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



EMB, CTRL = 32, 64
N, D = 32, 16
BS = 128
EPOCHS = 20
LR = 5e-4

full_ds = LogicDataset('dataset_50000_balanceado.csv')
ops_np = full_ds.ops.numpy()
counts = Counter(ops_np)
valid_idx = [i for i, op in enumerate(ops_np) if counts[op] > 1]

train_idx, test_idx = train_test_split(
    np.array(valid_idx),
    stratify=ops_np[valid_idx],
    test_size=0.2,
    random_state=42
)

train_ds = Subset(full_ds, train_idx)
test_ds  = Subset(full_ds, test_idx)

loader_train = DataLoader(train_ds, batch_size=BS, shuffle=True, drop_last=True)
loader_test  = DataLoader(test_ds,  batch_size=BS, shuffle=False)

def train_and_evaluate(model, name):
    print(f'\n=== Treinando {name} ===')
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    crit = nn.BCEWithLogitsLoss()
    train_stage(model, loader_train, opt, sched, crit, EPOCHS, stage_name=name)
    acc = evaluate(model, loader_test)
    print(f'→ Acurácia {name}: {acc:.4f}')
    return acc

# Treinar os modelos para comparação final
acc_rnn = train_and_evaluate(SimpleRNN(len(tok2idx), EMB, CTRL), 'RNN')
acc_lstm = train_and_evaluate(LSTMModel(len(tok2idx), EMB, CTRL), 'LSTM')
acc_ntm_1head = train_and_evaluate(NTM(len(tok2idx), EMB, CTRL, N, D, 1), 'NTM_1head')
acc_ntm_2heads = train_and_evaluate(NTM(len(tok2idx), EMB, CTRL, N, D, 2), 'NTM_2heads')

def plot_final_comparison(acc_dict):
    names = list(acc_dict.keys())
    scores = list(acc_dict.values())
    best_idx = np.argmax(scores)

    plt.figure(figsize=(8,5))
    bars = sns.barplot(x=names, y=scores, palette="mako")
    plt.ylim(0, 1.0)
    plt.ylabel("Acurácia")
    plt.title("Comparação Final: RNN vs LSTM vs NTM (1 e 2 Cabeças)")
    for i, v in enumerate(scores):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
        if i == best_idx:
            bars.patches[i].set_edgecolor('red')
            bars.patches[i].set_linewidth(3)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Plotar comparação final
acc_dict = {
    'RNN': acc_rnn,
    'LSTM': acc_lstm,
    'NTM-1 Head': acc_ntm_1head,
    'NTM-2 Heads': acc_ntm_2heads
}
plot_final_comparison(acc_dict)
