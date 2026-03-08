import pickle 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import os
with open("../data/cmn.txt", 'r', encoding = 'utf-8') as f:
    English = [i.strip().split('\t')[0] for i in f]
with open("../data/cmn.txt", 'r', encoding = 'utf-8') as f:
    Chinese = [i.strip().split('\t')[1] for i in f]
for i in range(len(Chinese)):
    Chinese[i] = "$" + Chinese[i] + "&"
for i in range(len(English)):
    English[i] = "$ " + English[i] + " &"

with open("../data/w.pkl", 'rb') as f:
    w = pickle.load(f)
w.append('$')
w.append('&')
Word = len(w)
device = torch.device('cuda')
# 超参数
src_vocab_size = Word  # 源词汇表大小
tgt_vocab_size = Word  # 目标词汇表大小
d_model = 512  # 模型维度
num_heads = 8  # 注意力头数量
num_layers = 6  # 编码器和解码器层数
d_ff = 2048  # 前馈网络内层维度
max_seq_length = 100  # 最大序列长度
dropout = 0.1  # Dropout 概率
batch = 150

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model    
        self.num_heads = num_heads
        self.d_k = d_model // num_heads 
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model) 
        self.W_v = nn.Linear(d_model, d_model) 
        self.W_o = nn.Linear(d_model, d_model) 
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q)) # (batch, heads, seq_len, d_k)
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # 第一层全连接
        self.fc2 = nn.Linear(d_ff, d_model)  # 第二层全连接
        self.relu = nn.ReLU()  # 激活函数

    def forward(self, x):
        # 前馈网络的计算
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)  # 初始化位置编码矩阵
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用正弦函数
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用余弦函数
        self.register_buffer('pe', pe.unsqueeze(0))  # 注册为缓冲区
        
    def forward(self, x):
        # 将位置编码添加到输入中
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)  
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)  
        self.norm1 = nn.LayerNorm(d_model)  
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)  
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))  
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))  
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)  # 自注意力机制
        self.cross_attn = MultiHeadAttention(d_model, num_heads)  # 交叉注意力机制
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)  # 前馈网络
        self.norm1 = nn.LayerNorm(d_model)  # 层归一化
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)  # Dropout
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))  # 残差连接和层归一化
        
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))  # 残差连接和层归一化
        
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))  # 残差连接和层归一化
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)  # 编码器词嵌入
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)  # 解码器词嵌入
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)  # 位置编码

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)  # 最终的全连接层
        self.dropout = nn.Dropout(dropout)  # Dropout

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask.to(device)
        nopeak_mask = nopeak_mask.to(device)
        tgt_mask = tgt_mask & nopeak_mask 
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
        
        output = self.fc(dec_output)
        return output

class dataset(torch.utils.data.Dataset):
    def __init__(self):
        pass
    def __getitem__(self, id):
    
        return (torch.tensor([w.index(i) for i in Chinese[id]]), torch.tensor([w.index(i) for i in English[id].split(' ')]))
    def __len__(self):
        return len(Chinese)

def collate_fn(batch):

    src_batch = []
    tgt_batch = []

    for src, tgt in batch:
        src_batch.append(src)
        tgt_batch.append(tgt)

    src_batch = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)

    return src_batch, tgt_batch
    
my_dataset = dataset()
loader = torch.utils.data.DataLoader(my_dataset, shuffle = True, batch_size = batch,collate_fn=collate_fn)

# 初始化模型

criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
if os.path.exists("../model/model.pth"):
    transformer = torch.load("../model/model.pth", weights_only=False)
    transformer = transformer.to(device)
    optimizer = optim.Adam(transformer.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
    optimizer.load_state_dict(torch.load("../model/optimizer.pth"))
else:
    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
    transformer = transformer.to(device)
    optimizer = optim.Adam(transformer.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

'''
# 训练循环
transformer.train()
for i in range(0, 50):
    total_loss = 0
    progress = 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        output = transformer(x, y[:, :-1])
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), y[:, 1:].contiguous().view(-1))
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress += 1
        if progress%10 == 0:
            print("total: ", Word/batch, "Now: ", progress)
    print(f"epoch: {i}, loss: {total_loss/len(my_dataset)}")

    torch.save(transformer, "../model/model.pth")
    torch.save(optimizer.state_dict(), "../model/optimizer.pth")
'''
'''
ok = 0
for x,y in loader:
    if ok < 10:
        x,y = x.to(device), y.to(device)
        output = transformer(x, y).detach()
        output = output[0]
        
        print(''.join([w[i.item()] for i in x[0] if  w[i.item()] != "嗨" and w[i.item()] != "$" and w[i.item()] != "&"]))
        print(' '.join([w[torch.argmax(torch.nn.functional.softmax(i, dim = 0))] for i in output if w[torch.argmax(torch.nn.functional.softmax(i, dim = 0))] != '&']))
            
        ok += 1
    else:
        break
'''

y = torch.tensor([w.index('$')]).unsqueeze(0)
x = my_dataset[20000][0].unsqueeze(0)
testdata = '我想学习哈哈语言'
x = torch.tensor([w.index(i) for i in testdata]).unsqueeze(0)
ok = ['$']
max = 0
print(''.join([w[i.item()] for i in x[0] if  w[i.item()] != "嗨" and w[i.item()] != "$" and w[i.item()] != "&"]))
while max < 10:
    x,y = x.to(device), y.to(device)
    output = transformer(x, y)[0]
    result = [w[torch.argmax(torch.nn.functional.softmax(i, dim = 0))] for i in output]
    print(ok)
    ok.append(result[-1])
    if '&' in result:
        pass
    y = torch.tensor([w.index(i) for i in ok]).unsqueeze(0)
    max += 1
