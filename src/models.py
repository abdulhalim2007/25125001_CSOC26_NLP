# %%
import torch
import torch.nn as nn
import math
import pandas as pd
import numpy as np
# %%
with open('../data/input.txt', 'r')as f:
    text = f.read()
# %%
text[:500]
# %%
len(set(text))
# %%
stoi={}
itos={}
# %%
i=0
# %%
for char in text:
    if char not in stoi:
        stoi[char]=i
        itos[i]=char
        i+=1
# %%
len(stoi)
# %%
stoi['n']
# %%
itos[9]
# %%
stoi['\n']
# %%
itos[11]
# %%
encoded = torch.tensor([stoi[char] for char in text])
# %%
decoded = ''.join(itos[i.item()] for i in encoded[:100])
# %%
decoded
# %%
text[0:100]
# %%
encoded[:100]
# %%
train = encoded[:int((0.9)*len(text))]
test =  encoded[int((0.9)*len(text)):]
# %%
len(train)
# %%
len(test)
# %%
batchsize=4
blocksize=8
torch.manual_seed(1337)
# %%
def make_batch():
    indexx = torch.randint(len(train) - blocksize,(batchsize,))
    x = torch.stack([train[i:i+blocksize] for i in indexx])
    y = torch.stack([train[i+1:i+blocksize+1] for i in indexx])
    return x,y
# %%
xb,yb = make_batch()
# %%
xb.shape
# %%
yb.shape
# %%
class casualMHAttention(nn.Module):
    def __init__(self, d_model, num_heads, blocksize):
        super(casualMHAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.blocksize = blocksize
        self.query = nn.Linear(d_model,d_model)
        self.key = nn.Linear(d_model,d_model)
        self.value = nn.Linear(d_model,d_model)
        self.output = nn.Linear(d_model,d_model)
        mask = torch.tril(torch.ones(blocksize,blocksize))
        self.register_buffer('mask', mask.view((1,1,blocksize,blocksize)))
        self.headsize = d_model//num_heads

    def split_heads(self, x):
        batchsize,seq_len,d_model = x.size()
        return x.view(batchsize,seq_len,self.num_heads,self.headsize).transpose(1,2)

    def forward(self,x):
        batchsize,seq_len,d_model = x.size()
        q= self.split_heads(self.query(x))
        k= self.split_heads(self.key(x))
        v= self.split_heads(self.value(x))
        att = torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.headsize)
        att = att.masked_fill(self.mask[:,:,:seq_len,:seq_len]==0, float('-inf'))
        att = torch.softmax(att,dim=-1)@v
        att = att.transpose(1,2).contiguous().view(batchsize,seq_len,d_model)
        output = self.output(att)
        return output
# %%
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model):
        super(FeedForwardNetwork, self).__init__()
        self.dff = 4*d_model
        self.d_model = d_model
        self.layer1 = nn.Linear(self.d_model,self.dff)
        self.layer2 = nn.Linear(self.dff,self.d_model)
    def forward(self,x):
        x = nn.functional.gelu(self.layer1(x))
        x = self.layer2(x)
        return x
# %%
class PreLNResidualBlock(nn.Module):
    def __init__(self, d_model, sub_layer):
        super(PreLNResidualBlock, self).__init__()
        self.sub_layer = sub_layer
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
    def forward(self,x):
        normalized_x =self.norm(x)
        sub_layer_out = self.sub_layer(normalized_x)
        return x +sub_layer_out
# %%
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, blocksize):
        super(PositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(blocksize, d_model)
    def forward(self,x):
        seqlen = x.size(1)
        pos = torch.arange(0,seqlen,dtype=torch.long, device=x.device)
        return x + self.emb(pos)
# %%
class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, blocksize):
        super(Transformer, self).__init__()
        self.ff = PreLNResidualBlock(d_model, FeedForwardNetwork(d_model))
        self.att = PreLNResidualBlock(d_model, casualMHAttention(d_model, num_heads, blocksize))

    def forward(self, x):
        x = self.att(x)
        x = self.ff(x)
        return x
# %%
class LanguageModel(nn.Module):
    def __init__(self, d_model, num_heads, blocksize, vocab_size, num_layers):
        super(LanguageModel, self).__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEmbedding(d_model, blocksize)
        self.transformers = nn.Sequential(*[Transformer(d_model, num_heads, blocksize) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

    def forward(self, x):
        x = self.token_emb(x)
        x=self.pos_emb(x)

        x=self.transformers(x)
        x = self.ln(x)
        logits = self.head(x)
        return logits
# %%
# for unit testing

blocksize=8
batchsize=4
d_model = 64
num_heads= 8
# %%
x = torch.randn(batchsize,blocksize,d_model)
# %%
att = casualMHAttention(d_model, num_heads, blocksize)
out = att(x)
assert out.shape == (batchsize,blocksize, d_model)
print('okay')
# %%
ffn = FeedForwardNetwork(d_model)
out = ffn(x)
assert out.shape==x.shape
print('okay')
# %%
transformer_block = Transformer(d_model, num_heads, blocksize)
out = transformer_block(x)
assert out.shape==x.shape
print('okay')
# %%
model = LanguageModel(d_model, num_heads, blocksize,len(set(text)),num_layers=6)
x,y=make_batch()
logits = model(x)
assert logits.shape==(batchsize,blocksize, len(set(text)))
print('okay')
# %%
# to verify causal compliance
xtest = torch.randn(batchsize,blocksize,d_model,requires_grad=True)
attlayer = casualMHAttention(d_model, num_heads, blocksize)
out = attlayer(xtest)
# %%
t = 3
loss_at_t = out[0,t,:].sum()
# %%
loss_at_t.backward()
# %%
future_grads = xtest.grad[0,t+1:,:].abs().sum().item()
# %%
assert future_grads==0
print('okay')
# %%
# overfitting a single batch
model = LanguageModel(d_model, num_heads, blocksize,len(set(text)),num_layers=6)
x,y=make_batch()

optim = torch.optim.Adam(model.parameters(), lr=0.01)
epochs=250
for i in range(epochs):
    logits = model(x)
    b,t,c = logits.shape
    logitsflat = logits.view(b*t,c)
    tarflat = y.view(b*t)
    loss = torch.nn.functional.cross_entropy(logitsflat, tarflat)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()
    if i%10==0:
        print(f'epoch: {i} loss: {loss.item():.10f}')


# %%
