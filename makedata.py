import pickle

with open("cmn.txt", 'r', encoding = 'utf-8') as f:
    English = [i.strip().split('\t')[0] for i in f]
with open("cmn.txt", 'r', encoding = 'utf-8') as f:
    Chinese = [i.strip().split('\t')[1] for i in f]
w = []

for i in Chinese:
    for j in i:
        if j not in w:
            w.append(j)
print(1)
for i in English:
    for j in i.split(' '):
        if j not in w:
            w.append(j)
print(2)
with open("w.pkl", 'wb') as f:
    pickle.dump(w, f)
print(w)
print(len(w), len(Chinese), len(English))
