import torch
import torch.nn as nn
import torch.nn.functional as F

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

context_size = 8
batch_size = 64


def get_batch(split, batch_size):
    data = train_x if split == 'train' else test_x
    ix = torch.randint(len(data) - context_size, (batch_size,))

    x = torch.stack([data[i:i+context_size] for i in ix])
    y = torch.stack([data[i+1:i+context_size+1] for i in ix])

    return x, y


class BigramLanguageModel(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_table = nn.Embedding(embed_dim, embed_dim)

    def forward(self, x, target=None):
        # x is (B, T)
        logits = self.embed_table(x) # (B, T) --> (B, T, C)

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
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            new_idx = torch.multinomial(probs, num_samples=1, replacement=True)
            idx = torch.cat((idx, new_idx), 1)
        return idx


model = BigramLanguageModel(vocab_size)

idx = torch.zeros((1, 1), dtype=torch.long)

print("Before training: ")
print(decode(model.generate(idx, 200)[0].tolist()))
print("-----------------------------------------------------")

epochs = 1000
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(epochs):
    xb, yb = get_batch('train', batch_size)
    logits, loss = model(xb, yb)
    
    if epoch % 100 == 0 or epoch+1 == epochs:
        print(f"Epoch {epoch} Loss: {loss.item()}")
    
    model.zero_grad()
    loss.backward()
    optimizer.step()


idx = torch.zeros((1, 1), dtype=torch.long)

print("After training: ")
print(decode(model.generate(idx, 200)[0].tolist()))




