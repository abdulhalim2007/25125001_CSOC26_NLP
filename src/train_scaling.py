# %%
import torch
import torch.nn as nn
import math
import pandas as pd
import numpy as np
from models import LanguageModel
# %%
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device
# %%
with open('../data/input.txt', 'r') as f:
    text = f.read()
# %%
text[:500]
# %%
len(set(text))
# %%
stoi = {}
itos = {}
# %%
i = 0
for char in text:
    if char not in stoi:
        stoi[char] = i
        itos[i] = char
        i += 1
# %%
vocab_size=len(stoi)
vocab_size
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
decoded = ''.join(itos[i.item()] for i in encoded[:100])
# %%
decoded
# %%
text[0:100]
# %%
encoded[:100]
# %%
train = encoded[:int((0.9) * len(text))]
test = encoded[int((0.9) * len(text)):]
# %%
len(train)
# %%
len(test)
# %%
batchsize = 4
blocksize = 8
torch.manual_seed(1337)


def make_batch(blocksize=blocksize,batchsize=batchsize,_train=True):
    if _train:
        data=train
    else:
        data=test
    indexx = torch.randint(len(data) - blocksize, (batchsize,))
    x = torch.stack([data[i:i + blocksize] for i in indexx])
    y = torch.stack([data[i + 1:i + blocksize + 1] for i in indexx])
    return x, y

# %%
xb, yb = make_batch()
xb.shape
yb.shape
# %%
def count_non_embedding_params(model):
    return sum(p.numel() for name, p in model.named_parameters()
               if 'embed' not in name and 'lm_head' not in name)
# %%
#Parameter Scaling Sweep
# %%
model1 = LanguageModel(d_model=64,num_layers=2,num_heads=2,blocksize=128,vocab_size=vocab_size)
model2 = LanguageModel(d_model=128,num_layers=4,num_heads=4,blocksize=256,vocab_size=vocab_size)
model3 = LanguageModel(d_model=256,num_layers=6,num_heads=8,blocksize=256,vocab_size=vocab_size)
# %%
count_non_embedding_params(model1)
# %%
count_non_embedding_params(model2)
# %%
count_non_embedding_params(model3)
# %%
@torch.no_grad()
def get_val_los(model,blocksize,batchsize, iter):
    model.eval()
    losses=[]
    for i in range(iter):
        x,y = make_batch(blocksize=blocksize,batchsize=batchsize,_train=False)
        x=x.to(device)
        y=y.to(device)
        logits = model(x)
        B, T, C = logits.shape
        loss = torch.nn.functional.cross_entropy(logits.view(B*T,C), y.view(B*T))
        losses.append(loss.item())
    return sum(losses)/len(losses)

# %%
def train_sweep(model, blocksize, batchsize=32, steps=3000):
    model=model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    best_los_val = float('inf')

    for step in range(steps):
        x,y = make_batch(blocksize=blocksize,batchsize=batchsize)
        x=x.to(device)
        y=y.to(device)
        logits = model(x)
        B, T, C = logits.shape
        loss = torch.nn.functional.cross_entropy(logits.view(B*T,C), y.view(B*T))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if step%200==0 or step==steps-1:
            val_loss=get_val_los(model,blocksize,batchsize,50)
            model.train()
            best_los_val = min(best_los_val, val_loss)
            print(f'Step : {step+1}, Train loss: {loss.item():.4f}, best val loss: {best_los_val:.4f}')
    return best_los_val
# %%
best_val_loss_1 = train_sweep(model1,blocksize=128)
# %%
best_val_loss_2 = train_sweep(model2,blocksize=256)
# %%
best_val_loss_3 = train_sweep(model3,blocksize=256)
# %%
#Data Scaling Sweep
_model2 = LanguageModel(d_model=128,num_layers=4,num_heads=4,blocksize=256,vocab_size=vocab_size)
# %%
def make_batch_for_scaling(blocksize,batchsize,split=1,is_train=True):
    if is_train:
        data=train[:int(split*len(train))]
    else:
        data=test
    indexx = torch.randint(len(data) - blocksize, (batchsize,))
    x = torch.stack([data[i:i + blocksize] for i in indexx])
    y = torch.stack([data[i + 1:i + blocksize + 1] for i in indexx])
    return x, y
# %%
def train_data_scaling_sweep(model, blocksize, batchsize=32, steps=3000, split=1):
    model=model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    best_los_val = float('inf')

    for step in range(steps):
        x,y = make_batch_for_scaling(blocksize=blocksize,batchsize=batchsize, split=split)
        x=x.to(device)
        y=y.to(device)
        logits = model(x)
        B, T, C = logits.shape
        loss = torch.nn.functional.cross_entropy(logits.view(B*T,C), y.view(B*T))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if step%200==0 or step==steps-1:
            val_loss=get_val_los(model,blocksize,batchsize,50)
            model.train()
            best_los_val = min(best_los_val, val_loss)
            print(f'Step : {step+1}, Train loss: {loss.item():.4f}, best val loss: {best_los_val:.4f}')
    return best_los_val
# %%
best_val_loss_scalling_10 = train_data_scaling_sweep(_model2,256, 32,split=0.1)
# %%
_model2 = LanguageModel(d_model=128,num_layers=4,num_heads=4,blocksize=256,vocab_size=vocab_size)
best_val_loss_scalling_25 = train_data_scaling_sweep(_model2,256, 32,split=0.25)
# %%
_model2 = LanguageModel(d_model=128,num_layers=4,num_heads=4,blocksize=256,vocab_size=vocab_size)
best_val_loss_scalling_50 = train_data_scaling_sweep(_model2,256, 32,split=0.5)
# %%
_model2 = LanguageModel(d_model=128,num_layers=4,num_heads=4,blocksize=256,vocab_size=vocab_size)
best_val_loss_scalling_100 = train_data_scaling_sweep(_model2,256, 32,split=1)
# %%
best_val_loss_1
# %%
best_val_loss_2
# %%
best_val_loss_3
# %%
print(best_val_loss_scalling_10)
print(best_val_loss_scalling_25)
print(best_val_loss_scalling_50)
print(best_val_loss_scalling_100)
# %%
for name, _ in model1.named_parameters():
    print(name)
# %%
print(best_val_loss_1)
print(best_val_loss_2)
print(best_val_loss_3)

print(best_val_loss_scalling_10)
print(best_val_loss_scalling_25)
print(best_val_loss_scalling_50)
print(best_val_loss_scalling_100)

print(count_non_embedding_params(model1))
print(count_non_embedding_params(model2))
print(count_non_embedding_params(model3))
# %%
