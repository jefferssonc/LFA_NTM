import random
import pandas as pd

# Operadores lógicos
ops = {
    '¬': lambda x: not x,
    '∧': lambda x, y: x and y,
    '∨': lambda x, y: x or y,
    '→': lambda x, y: (not x) or y,
    '↔': lambda x, y: x == y
}

variables = ['A', 'B', 'C', 'D', 'E']

def random_var():
    return random.choice(variables)

def random_expr(depth=0, max_depth=4):
    if depth >= max_depth or (depth > 0 and random.random() < 0.2):
        return random_var()
    op = random.choice(['¬', '∧', '∨', '→', '↔'])
    if op == '¬':
        return f"¬({random_expr(depth + 1, max_depth)})"
    else:
        left = random_expr(depth + 1, max_depth)
        right = random_expr(depth + 1, max_depth)
        return f"({left} {op} {right})"

def eval_expr(expr, val_dict):
    def eval_recursive(tokens):
        token = tokens.pop(0)
        if token == '¬':
            tokens.pop(0)  # remove '('
            val = eval_recursive(tokens)
            tokens.pop(0)  # remove ')'
            return ops['¬'](val)
        elif token == '(':
            left = eval_recursive(tokens)
            op = tokens.pop(0)
            right = eval_recursive(tokens)
            tokens.pop(0)  # remove ')'
            return ops[op](left, right)
        elif token in val_dict:
            return val_dict[token]
        else:
            raise ValueError(f"Unknown token: {token}")

    expr = expr.replace('(', ' ( ').replace(')', ' ) ')
    tokens = expr.split()
    return eval_recursive(tokens)

def main():
    total = 50000
    target_true = total // 2
    target_false = total // 2

    count_true = 0
    count_false = 0

    data = []

    depths = [2, 3, 4, 5]  # escolhidas para diversidade e complexidade moderada

    print("Gerando expressões balanceadas...")

    while count_true < target_true or count_false < target_false:
        depth = random.choice(depths)
        expr = random_expr(depth=0, max_depth=depth)
        vars_in_expr = sorted(set(filter(lambda c: c in variables, expr)))
        val_dict = {var: random.choice([True, False]) for var in vars_in_expr}
        try:
            result = eval_expr(expr, val_dict)
            if result and count_true < target_true:
                valores = ', '.join([f"{k}={v}" for k, v in val_dict.items()])
                data.append([expr, valores, result])
                count_true += 1
            elif not result and count_false < target_false:
                valores = ', '.join([f"{k}={v}" for k, v in val_dict.items()])
                data.append([expr, valores, result])
                count_false += 1
        except Exception:
            continue  # Ignora expressões inválidas

    df = pd.DataFrame(data, columns=["expressao", "valores", "resultado"])
    df.to_csv("dataset_50000_balanceado.csv", index=False)
    print("Arquivo gerado: dataset_50000_balanceado.csv")
    print("Total True:", count_true)
    print("Total False:", count_false)
    print("Profundidade usada: 2 a 5 (aleatória por expressão)")
if __name__ == "__main__":
    main()

