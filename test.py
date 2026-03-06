'''
this file is for building a transformer with pytorch in the paper "Attention is all you need"

Embedding wordlist is within English and Chinese:
English: I you he her she him we eat go to in school talk 
'''
#super parameter
Embedding_Depth = 512
Multi_Head = 8
w = ["这","是","到","的","和","一个","在","那个","有","我",
            "它","为了","不","在…上","和…一起","他","作为","你","做","在",
            "这个","但是","他的","被","来自","他们","我们","说","她的","她",
            "或者","一个","将","我的","一个","全部","会","那里","他们的","什么",
            "所以","向上","向外","如果","关于","谁","得到","哪个","去","我",
            "什么时候","制作","能够","喜欢","时间","不","只是","他","知道","拿",
            "人们","进入","年","你的","好的","一些","可以","他们","看见","其他",
            "比","然后","现在","看","只","来","它的","在…之上","认为","也",
            "回来","之后","使用","二","如何","我们的","工作","第一","很好","方式",
            "甚至","新的","想要","因为","任何","这些","给","天","大多数","我们"," ","$", "$$"

            "the","be","to","of","and","a","in","that","have","I",
            "it","for","not","on","with","he","as","you","do","at",
            "this","but","his","by","from","they","we","say","her","she",
            "or","an","will","my","one","all","would","there","their","what",
            "so","up","out","if","about","who","get","which","go","me",
            "when","make","can","like","time","no","just","him","know","take",
            "people","into","year","your","good","some","could","them","see","other",
            "than","then","now","look","only","come","its","over","think","also",
            "back","after","use","two","how","our","work","first","well","way",
            "even","new","want","because","any","these","give","day","most","us"]
Word = len(w)

import torch
import torchvision
import matplotlib.pyplot

class Embedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wordlist = torch.nn.Parameter(torch.randn(len(w), Embedding_Depth))
        
    def forward(self, x):   #x is a vector of input sequence
        output = []
        #add all the vector
        for i in range(0, len(x)):
            output.append(self.wordlist[w.index(x[i])])
        #positional encoding
        positional_matrix = torch.zeros(len(x), Embedding_Depth)
        for i in range(0, len(x)):
            for j in range(0, int((Embedding_Depth-1)/2)):
                positional_matrix[i][2*j] = torch.sin(torch.tensor(i/10000**(2*j/Embedding_Depth))) 
                positional_matrix[i][2*j+1] = torch.cos(torch.tensor(i/10000**(2*j/Embedding_Depth)))
        #stack them into a tensor
        return torch.stack(output) + positional_matrix
        
        
class selfAttension(torch.nn.Module):
    def __init__(self, mask = False):
        super().__init__()
        self.mask = mask
        Head_K = [torch.randn(Embedding_Depth, int(Embedding_Depth/Multi_Head)) for i in range(Multi_Head)]
        Head_Q = [torch.randn(Embedding_Depth, int(Embedding_Depth/Multi_Head)) for i in range(Multi_Head)]
        Head_V = [torch.randn(Embedding_Depth, int(Embedding_Depth/Multi_Head)) for i in range(Multi_Head)]
        self.K = torch.nn.Parameter(torch.stack(Head_K))
        self.Q = torch.nn.Parameter(torch.stack(Head_Q))
        self.V = torch.nn.Parameter(torch.stack(Head_V))
        self.output_mat = torch.nn.Parameter(torch.randn(Embedding_Depth, Embedding_Depth))
        
    def forward(self, x):
        result = []
        DK = torch.sqrt(torch.tensor(Embedding_Depth/Multi_Head))
        for i in range(0, Multi_Head):
            k = x @ self.K[i]
            q = x @ self.Q[i]
            v = x @ self.V[i]
            if(self.mask):
                mask_mat = torch.triu(torch.full((len(x), len(x)), float('-inf')), diagonal = 1)
                result.append(torch.nn.functional.softmax((q @ k.T + mask_mat)/DK, dim = 1) @ v)
            else:
                result.append(torch.nn.functional.softmax((q @ k.T)/DK, dim = 1) @ v)
        result = torch.hstack(result)
        result = result @ self.output_mat
        result = torch.nn.functional.softmax(result, dim = 1)
        return result


class EDAttension(torch.nn.Module):
    def __init__(self):
        super().__init__()
        Head_K = [torch.randn(Embedding_Depth, int(Embedding_Depth/Multi_Head)) for i in range(Multi_Head)]
        Head_Q = [torch.randn(Embedding_Depth, int(Embedding_Depth/Multi_Head)) for i in range(Multi_Head)]
        Head_V = [torch.randn(Embedding_Depth, int(Embedding_Depth/Multi_Head)) for i in range(Multi_Head)]
        self.K = torch.nn.Parameter(torch.stack(Head_K))
        self.Q = torch.nn.Parameter(torch.stack(Head_Q))
        self.V = torch.nn.Parameter(torch.stack(Head_V))
        self.output_mat = torch.nn.Parameter(torch.randn(Embedding_Depth, Embedding_Depth))
        
    def forward(self, x, y):
        result = []
        DK = torch.sqrt(torch.tensor(Embedding_Depth/Multi_Head))
        for i in range(0, Multi_Head):
            k = x @ self.K[i]
            q = y @ self.Q[i]
            v = x @ self.V[i]
            result.append(torch.nn.functional.softmax((q @ k.T)/DK, dim = 1) @ v)
        result = torch.hstack(result)
        result = result @ self.output_mat
        result = torch.nn.functional.softmax(result, dim = 1)
        return result


class FFN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Parameter(torch.randn(Embedding_Depth, Embedding_Depth))
        self.linear2 = torch.nn.Parameter(torch.randn(Embedding_Depth, Embedding_Depth))
        #self.bias1 = torch.nn.Parameter(torch.tensor([0]))
        #self.bias2 = torch.nn.Parameter(torch.tensor([0]))
        
    def forward(self, x):
        x = torch.nn.functional.relu(x @ self.linear1) @ self.linear2 
        return x
        
        
#===============================================================================================================================================
class my_dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.X = ["$","I","know","you"]
        self.Y = ["$","我","知道","你"]
    
    def __getitem__(self, id):
        return (self.X, self.Y)
        
    def __len__(self):
        return len(self.X)
        
        
class MyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.en_embedding = Embedding()
        self.en_atten1 = selfAttension()
        self.en_ffn1 = FFN()
        self.en_atten2 = selfAttension()
        self.en_ffn2 = FFN()
        self.en_atten3 = selfAttension()
        self.en_ffn3 = FFN()
        self.en_atten4 = selfAttension()
        self.en_ffn4 = FFN()
        self.en_atten5 = selfAttension()
        self.en_ffn5 = FFN()
        self.en_atten6 = selfAttension()
        self.en_ffn6 = FFN()
        
        self.de_embedding = Embedding()
        self.de_atten1 = selfAttension(mask = True)
        self.de_ffn1 = FFN()
        self.de_atten2 = selfAttension(mask = True)
        self.de_ffn2 = FFN()
        self.de_atten3 = selfAttension(mask = True)
        self.de_ffn3 = FFN()
        self.de_atten4 = selfAttension(mask = True)
        self.de_ffn4 = FFN()
        self.de_atten5 = selfAttension(mask = True)
        self.de_ffn5 = FFN()
        self.de_atten6 = selfAttension(mask = True)
        self.de_ffn6 = FFN()
        
        self.cross_atten1 = EDAttension()
        self.cross_ffn1 = FFN()
        self.cross_atten2 = EDAttension()
        self.cross_ffn2 = FFN()
        self.cross_atten3 = EDAttension()
        self.cross_ffn3 = FFN()
        self.cross_atten4 = EDAttension()
        self.cross_ffn4 = FFN()
        self.cross_atten5 = EDAttension()
        self.cross_ffn5 = FFN()
        self.cross_atten6 = EDAttension()
        self.cross_ffn6 = FFN()
        
        self.output = torch.nn.Parameter(torch.randn(Embedding_Depth, Word))
    def forward(self, x, y):
        # ===== Encoder =====
        x = self.en_embedding(x)
        x = self.en_atten1(x)
        x = self.en_ffn1(x) + x
        x = self.en_atten2(x)
        x = self.en_ffn2(x) + x
        x = self.en_atten3(x)
        x = self.en_ffn3(x) + x
        x = self.en_atten4(x)
        x = self.en_ffn4(x) + x
        x = self.en_atten5(x)
        x = self.en_ffn5(x) + x
        x = self.en_atten6(x)
        x = self.en_ffn6(x) + x
        
    
        # ===== Decoder =====
        y = self.de_embedding(y)
        y = self.de_atten1(y)
        y = self.de_ffn1(y) + y
        y = self.de_atten2(y)
        y = self.de_ffn2(y) + y
        y = self.de_atten3(y)
        y = self.de_ffn3(y) + y
        y = self.de_atten4(y)
        y = self.de_ffn4(y) + y
        y = self.de_atten5(y)
        y = self.de_ffn5(y) + y
        y = self.de_atten6(y)
        y = self.de_ffn6(y) + y

        # ===== Cross-Attention =====
        y = self.cross_atten1(x, y)
        y = self.cross_ffn1(y) + y
        y = self.cross_atten2(x, y)
        y = self.cross_ffn2(y) + y
        y = self.cross_atten3(x, y)
        y = self.cross_ffn3(y) + y
        y = self.cross_atten4(x, y)
        y = self.cross_ffn4(y) + y
        y = self.cross_atten5(x, y)
        y = self.cross_ffn5(y) + y
        y = self.cross_atten6(x, y)
        y = self.cross_ffn6(y)
        
        y = y @ self.output
        return y
        
        
        
dataset = my_dataset()
#model = MyTransformer()
model = torch.load("model.pth", weights_only=False)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

def getget(stringlist):
    newone = []
    for i in stringlist:
        newone.append(w.index(i))
    return torch.tensor(newone)

for i in range(0, 100):
    total_loss = 0
    output = model(dataset[0][0], dataset[0][1])
    loss = criterion(output, getget(dataset[0][1]))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"epoch: {i}, loss: {loss}")
torch.save(model, "model.pth")



result = torch.nn.functional.softmax(model(dataset[0][0], dataset[0][1]), dim = 1)
print(result.shape)
indi = torch.argmax(result, dim = 1)
print(indi)
print(w[indi[0]],w[indi[1]],w[indi[2]],w[indi[3]])

