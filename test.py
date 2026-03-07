'''
this file is for building a transformer with pytorch in the paper "Attention is all you need"

Embedding wordlist is within English and Chinese:
English: I you he her she him we eat go to in school talk 
'''
#super parameter
import pickle
learning_rate = 1e-4
Embedding_Depth = 512
Multi_Head = 8
with open("cmn.txt", 'r', encoding = 'utf-8') as f:
    English = [i.strip().split('\t')[0] for i in f]
with open("cmn.txt", 'r', encoding = 'utf-8') as f:
    Chinese = [i.strip().split('\t')[1] for i in f]
Chinese = Chinese[:200]
English = English[:200]
with open("w.pkl", 'rb') as f:
    w = pickle.load(f)
Word = len(w)



import torch
import torchvision
import matplotlib.pyplot
import os
import math

class PositionalEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(len(w), Embedding_Depth)

    def forward(self, x):   #x is a vector of input sequence
        positional_matrix = torch.zeros(len(x), Embedding_Depth)
        for i in range(0, len(x)):
            for j in range(0, int((Embedding_Depth-1)/2)):
                positional_matrix[i][2*j] = math.sin(i/10000**(2*j/Embedding_Depth)) 
                positional_matrix[i][2*j+1] = math.cos(i/10000**(2*j/Embedding_Depth))
        output = self.embedding(torch.tensor([w.index(i) for i in x])) + positional_matrix
        return output
        
        
class selfAttension(torch.nn.Module):
    def __init__(self, mask = False):
        super().__init__()
        self.mask = mask
        self.K = torch.nn.ModuleList([torch.nn.Linear(Embedding_Depth, int(Embedding_Depth/Multi_Head)) for i in range(Multi_Head)])
        self.Q = torch.nn.ModuleList([torch.nn.Linear(Embedding_Depth, int(Embedding_Depth/Multi_Head)) for i in range(Multi_Head)])
        self.V = torch.nn.ModuleList([torch.nn.Linear(Embedding_Depth, int(Embedding_Depth/Multi_Head)) for i in range(Multi_Head)])
        self.output_mat = torch.nn.Linear(Embedding_Depth, Embedding_Depth)
        
    def forward(self, x):
        result = []
        DK = torch.sqrt(torch.tensor(Embedding_Depth/Multi_Head))
        for i in range(0, Multi_Head):
            k =self.K[i](x)
            q =self.Q[i](x)
            v =self.V[i](x)
            if(self.mask):
                mask_mat = torch.triu(torch.full((len(x), len(x)), float('-inf')), diagonal = 1)
                result.append(torch.nn.functional.softmax((q @ k.T + mask_mat)/DK, dim = 1) @ v)
            else:
                result.append(torch.nn.functional.softmax((q @ k.T)/DK, dim = 1) @ v)
        result = torch.hstack(result)
        result = self.output_mat(result)
        return result


class EDAttension(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.K = torch.nn.ModuleList([torch.nn.Linear(Embedding_Depth, int(Embedding_Depth/Multi_Head)) for i in range(Multi_Head)])
        self.Q = torch.nn.ModuleList([torch.nn.Linear(Embedding_Depth, int(Embedding_Depth/Multi_Head)) for i in range(Multi_Head)])
        self.V = torch.nn.ModuleList([torch.nn.Linear(Embedding_Depth, int(Embedding_Depth/Multi_Head)) for i in range(Multi_Head)])
        self.output_mat = torch.nn.Linear(Embedding_Depth, Embedding_Depth)
        
    def forward(self, x, y):
        result = []
        DK = torch.sqrt(torch.tensor(Embedding_Depth/Multi_Head))
        for i in range(0, Multi_Head):
            k = self.K[i](x)
            q = self.Q[i](y) 
            v = self.V[i](x) 
            result.append(torch.nn.functional.softmax((q @ k.T)/DK, dim = 1) @ v)
        result = torch.hstack(result)
        result = self.output_mat(result) 
        return result


class FFN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(Embedding_Depth, Embedding_Depth)
        self.linear2 = torch.nn.Linear(Embedding_Depth, Embedding_Depth)
        
    def forward(self, x):
        x = self.linear2(torch.nn.functional.relu(self.linear1(x)))
        return x
        
class encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attension = selfAttension()
        self.ffn = FFN()
        
    def forward(self, x):
        x = self.ffn(self.attension(x) + x)
        return x
        
class decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attension = selfAttension(True)
        self.ffn = FFN()
        
    def forward(self, x):
        x = self.ffn(self.attension(x) + x)
        return x
        
#===============================================================================================================================================
class my_dataset(torch.utils.data.Dataset):
    def __init__(self):
        pass
    def __getitem__(self, id):
        return ([i for i in Chinese[id]], ['begin'] + English[id].split(' '))
        
    def __len__(self):
        return len(Chinese)
        
        
class MyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.en_embedding = PositionalEmbedding()
        self.encoder = torch.nn.ModuleList([encoder() for i in range(0, 8)])
        self.de_embedding = PositionalEmbedding()
        self.decoder = torch.nn.ModuleList([decoder() for i in range(0, 8)])
        self.cross = EDAttension()
        self.aftercross = torch.nn.ModuleList([encoder() for i in range(0, 8)])
        self.output = torch.nn.Linear(Embedding_Depth, Word)
        
    def forward(self, x, y):
        x = self.en_embedding(x)
        y = self.de_embedding(y)
        for i in self.encoder:
            x = i(x)
        for i in self.decoder:
            y = i(y)
        x = self.cross(x, y)
        for i in self.aftercross:
            x = i(x)
        x = self.output(x)
        return x
        
     
dataset = my_dataset()
criterion = torch.nn.CrossEntropyLoss()

if os.path.exists("model.pth"):
    model = torch.load("model.pth", weights_only=False)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    optimizer.load_state_dict(torch.load("optimizer.pth"))
else:
    model = MyTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

def getget(stringlist):
    newone = []
    for i in stringlist:
        newone.append(w.index(i))
    return torch.tensor(newone)

for i in range(0, 10):
    total_loss = 0
    for j in range(0, len(dataset)):
        output = model(dataset[j][0], dataset[j][1])
        loss = criterion(output, getget(dataset[j][1]))
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"epoch: {i}, loss: {total_loss/len(dataset)}")

    torch.save(model, "model.pth")
    torch.save(optimizer.state_dict(), "optimizer.pth")


result = torch.nn.functional.softmax(model(dataset[1][0], dataset[1][1]), dim = 1)
print(dataset[1][0])
indi = torch.argmax(result, dim = 1)
print(result)
for i in indi:
    print(w[i])

