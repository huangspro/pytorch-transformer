'''
this file is for building a transformer with pytorch in the paper "Attention is all you need"

Embedding wordlist is within English and Chinese:
English: I you he her she him we eat go to in school talk 
'''
#super parameter
Embedding_Depth = 512
Multi_Head = 8
w = [
  "$", "&",
  "I", "you", "he", "she", "we", "they", "it", "me", "us", "them", "her", "and",
  "know", "like", "see", "want", "make", "take", "give", "use", "do", "can", "will",
  "go", "come", "think", "be", "used", "by", "together", "should", "because", "is", "good", "results",
  "learning", "English", "so", "that", "communicate", "how", "works", "before", "with", "for", "there", "this",
  "我", "你", "他", "她", "我们", "他们", "它",
  "知道", "喜欢", "看见", "想要", "制作", "拿", "给", "使用", "做", "可以", "会",
  "去", "来", "认为", "是", "被", "一起", "应该", "因为", "很好", "结果", "正在", "学习",
  "英语", "以便", "与", "交流", "把", "如果", "需要", "如何", "运作", "的", "所以", "工作",
  "这个", "那里", "并",
  "help", "She", "wants", "to", "We", "They", "what", "You", "He", "if", "need", "work",
  "而且", "用", "帮助", "看", "为", "在", "之前", "在做", "什么", "这", "和"
]
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
        self.K = [torch.Linear(Embedding_Depth, int(Embedding_Depth/Multi_Head)) for i in range(Multi_Head)]
        self.Q = [torch.Linear(Embedding_Depth, int(Embedding_Depth/Multi_Head)) for i in range(Multi_Head)]
        self.V = [torch.Linear(Embedding_Depth, int(Embedding_Depth/Multi_Head)) for i in range(Multi_Head)]
        
        self.output_mat = torch.Linear(Embedding_Depth, Embedding_Depth)
        
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
        self.linear1 = torch.nn.Parameter(torch.randn(Embedding_Depth, Embedding_Depth))
        self.linear2 = torch.nn.Parameter(torch.randn(Embedding_Depth, Embedding_Depth))
        #self.bias1 = torch.nn.Parameter(torch.tensor([0]))
        #self.bias2 = torch.nn.Parameter(torch.tensor([0]))
        
    def forward(self, x):
        x = torch.nn.functional.sigmoid(x @ self.linear1) @ self.linear2 
        return x
        
        
#===============================================================================================================================================
class my_dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.X = [
            ["$","I","know","you","and","I","will","help","you","with","this","&"],
            #["$","I","see","you","and","I","will","help","you","with","this","&"],
            #["$","She","wants","to","see","it","and","I","can","make","it","for","her","&"],
            #["$","We","will","go","there","and","use","it","before","they","come","&"],
            #["$","They","like","to","see","us","and","they","will","know","what","we","do","&"],
            #["$","I","think","this","can","be","used","by","you","and","me","together","&"],
            #["$","You","should","do","it","because","it","is","good","and","we","can","see","results","&"],
            #["$","He","is","learning","English","so","that","he","can","use","it","to","communicate","with","them","&"],
            #["$","We","know","it","and","we","will","give","it","to","them","if","they","need","it","&"],
            #["$","They","want","to","make","it","and","they","can","see","how","it","works","&"],
            #["$","I","know","you","and","you","know","me","so","we","can","work","together","&"]
        ]

        self.Y = [
            ["$","我","知道","你","而且","我","会","用","这个","帮助","你","&"],
            #["$","我","看见","你","而且","我","会","用","这个","帮助","你","&"],
            #["$","她","想要","看","它","而且","我","可以","为","她","制作","它","&"],
            #["$","我们","会","去","那里","并","使用","它","在","他们","来","之前","&"],
            #["$","他们","喜欢","看见","我们","而且","他们","会","知道","我们","在做","什么","&"],
            #["$","我","认为","这","可以","被","你","和","我","一起","使用","&"],
            #["$","你","应该","做","它","因为","它","很好","而且","我们","可以","看见","结果","&"],
            #["$","他","正在","学习","英语","以便","他","可以","使用","它","与","他们","交流","&"],
            #["$","我们","知道","它","而且","我们","会","把","它","给","他们","如果","他们","需要","它","&"],
            #["$","他们","想要","制作","它","而且","他们","可以","看见","它","是","如何","运作","的","&"],
            #["$","我","知道","你","而且","你","知道","我","所以","我们","可以","一起","工作","&"]
        ]
    def __getitem__(self, id):
        return (self.X[id], self.Y[id])
        
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
model = MyTransformer()
#model = torch.load("model.pth", weights_only=False)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)

def getget(stringlist):
    newone = []
    for i in stringlist:
        newone.append(w.index(i))
    return torch.tensor(newone)

for i in range(0, 10):
    for j in range(0, len(dataset)):
        total_loss = 0
        output = model(dataset[j][0], dataset[j][1])
        loss = criterion(output, getget(dataset[j][1]))
        total_loss += loss
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
    
        print(f"epoch: {i}, loss: {total_loss/len(dataset)}")

#torch.save(model, "model.pth")



result = torch.nn.functional.softmax(model(dataset[0][0], dataset[0][1]), dim = 1)
indi = torch.argmax(result, dim = 1)
print(result)
for i in indi:
    print(w[i])

