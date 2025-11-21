import torch
import torch.nn as nn
import torch.nn.functional as F

epochs = 5000
n_layers = 6
context_size = 256
batch_size = 64
learning_rate = 3e-4
num_heads = 6
n_emb = 384
head_size = n_emb // num_heads
dropout = .2
device = 'mps'

with open('data/more.txt', 'r') as f:
    data = f.read()

vocab = sorted(set(data))
vocab_size = len(vocab)

stoi = {s:i for i, s in enumerate(vocab)}
itos = {i:s for s, i in stoi.items()}

encode = lambda x: [stoi[let] for let in x]
decode = lambda x: ''.join([itos[num] for num in x])

x = torch.tensor(encode(data))
n = int(len(x) * .9)

train_x = x[:n]
test_x = x[n:]


def get_batch(split, batch_size):
    data = train_x if split == 'train' else test_x
    ix = torch.randint(len(data) - context_size, (batch_size,))

    x = torch.stack([data[i:i+context_size] for i in ix])
    y = torch.stack([data[i+1:i+context_size+1] for i in ix])

    return x, y


class Head(nn.Module):
    def __init__(self, n_emb, head_size):
        super().__init__()
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((context_size, context_size))))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, T, C = x.shape
        q = self.query(x) # B, T, 16
        k = self.key(x) # B, T, 16
        v = self.value(x) # B, T, 16

        wei = q @ k.transpose(-2, -1) * C**-0.5 # B, T, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v # B, T, hs
        return out 


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_emb, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads*head_size, n_emb)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)        
        out = self.proj(out)
        out = self.dropout(out)
        
        return out

class FeedForward(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4*n_emb),
            nn.ReLU(),
            nn.Linear(4*n_emb, n_emb),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_emb, n_head):
        super().__init__()
        head_size = n_emb // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_emb)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x

class BigramLanguageModel(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(context_size, embed_dim)
        self.blocks = nn.Sequential(*[Block(n_emb, num_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size)

    def forward(self, x, target=None):
        # x is (B, T)
        _, T = x.shape
        tok_embs = self.tok_emb(x) # (B, T) --> (B, T, C)
        pos_embs = self.pos_emb(torch.arange(T)) # (T, C)
        x = tok_embs + pos_embs
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)

        if target is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # Want to preserve channel dimension
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)

        else:
            loss = None

        return logits, loss

    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            logits, _ = self(idx[:, -context_size:])
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            new_idx = torch.multinomial(probs, num_samples=1, replacement=True)
            idx = torch.cat((idx, new_idx), 1)
        return idx


model = BigramLanguageModel(n_emb)

idx = torch.zeros((1, 1), dtype=torch.long)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    xb, yb = get_batch('train', batch_size)
    logits, loss = model(xb, yb)
    
    if epoch % 100 == 0 or epoch+1 == epochs:
        print(f"Epoch {epoch+1} Loss: {loss.item()}")
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


idx = torch.zeros((1, 1), dtype=torch.long)

print("After training: ")
print(decode(model.generate(idx, 1000)[0].tolist()))




