import jieba
import visdom
import csv
import pandas as pd
import re
import torch
from torchtext import data
from torchtext.vocab import Vectors
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# GPU
if torch.cuda.is_available():
    device = 'cuda:0' 
else:
    device = 'cpu'
print('GPU state:', device)
viz = visdom.Visdom(env='bdahw2')
viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))  #前面是y軸，後面是x軸
viz.line([0.], [0.], win='test_loss', opts=dict(title='test loss')) 
viz.line([0.], [0.], win='test_acc', opts=dict(title='test acc'))
'''
#開啟 CSV 檔案
text = []
label = []
dict = {
  "text" : text,
  "label" : label
    }
tra = pd.DataFrame(dict)
test = pd.DataFrame(dict)
with open("漲1.csv", newline='',encoding = "utf-8") as csvFile:
  # 轉成一個 dictionary, 讀取 CSV 檔內容，將每一列轉成字典
    rows = csv.DictReader(csvFile)
  # 迴圈輸出 指定欄位
    for row in rows:
      #print(row['content'])
      new = pd.DataFrame({"text":row['content'],"label":1},index=[1])
      tra = tra.append(new,ignore_index=True)
with open("漲2.csv", newline='',encoding = "utf-8") as csvFile:
  # 轉成一個 dictionary, 讀取 CSV 檔內容，將每一列轉成字典
    rows = csv.DictReader(csvFile)
  # 迴圈輸出 指定欄位
    for row in rows:
        #print(row['content'])
      new = pd.DataFrame({"text":row['content'],"label":1},index=[1])
      tra = tra.append(new,ignore_index=True)
with open("跌1.csv", newline='',encoding = "utf-8") as csvFile:
  # 轉成一個 dictionary, 讀取 CSV 檔內容，將每一列轉成字典
    rows = csv.DictReader(csvFile)
  # 迴圈輸出 指定欄位
    for row in rows:
        #print(row['content'])
      new = pd.DataFrame({"text":row['content'],"label":0},index=[1])
      tra = tra.append(new,ignore_index=True)    
with open("跌2.csv", newline='',encoding = "utf-8") as csvFile:
  # 轉成一個 dictionary, 讀取 CSV 檔內容，將每一列轉成字典
    rows = csv.DictReader(csvFile)
  # 迴圈輸出 指定欄位
    for row in rows:
        #print(row['content'])
      new = pd.DataFrame({"text":row['content'],"label":0},index=[1])
      tra = tra.append(new,ignore_index=True)  

      
    

tra.to_csv('train.csv',encoding = "utf-8",index = False)
test.to_csv('test.csv',encoding = "utf-8",index = False)
'''
'''
with open("train.csv", newline='',encoding="utf-8") as csvFile:
  # 轉成一個 dictionary, 讀取 CSV 檔內容，將每一列轉成字典
    rows = csv.DictReader(csvFile)
    for row in rows:
        print(row)
'''

def tokenizer(text): # create a tokenizer function
    regex = re.compile(r'[^\u4e00-\u9fa5]') #只保留中文字
    text = regex.sub(' ', text)
    return [word for word in jieba.cut(text) if word.strip()]  

def get_stop_words():
    file_object = open('stopword.txt',encoding = "utf-8")
    stop_words = []
    for line in file_object.readlines():
        line = line[:-1]
        line = line.strip()
        stop_words.append(line)
    return stop_words    
stop_words = get_stop_words()
print(stop_words)
text = data.Field(lower=True, tokenize = tokenizer,fix_length=250,batch_first = True,stop_words = stop_words)
label = data.Field(sequential=False)

train , test = data.TabularDataset.splits(
    path = './', format = 'csv', skip_header = True,
    train='train.csv',
    validation='test.csv',
    #注意下面的fields，我们的数据集格式就要包含label和text两个域。
    fields=[
        ('text', text),
        ('label', label)
    ]
)
#print(vars(train[0]))
text.build_vocab(train, test, min_freq = 5)   
label.build_vocab(train, test)
print(vars(test.examples[1]))
print(len(train.examples))
print(len(test.examples))
#print(train[0].__dict__.keys())
'''
train_iter, test_iter = data.Iterator.splits(
            (train, test),
            sort_key=lambda x: len(x.text),
            batch_sizes=(16, len(test)), # 訓練集設置batch_size,驗證集整個集合用於測試
            device= device
    )'''
train_iter = data.Iterator(dataset=train, batch_size=16, shuffle=True,
            sort_within_batch=False, repeat=False, device=device)
test_iter = data.Iterator(dataset=test, batch_size=len(test.examples), shuffle=True,
            sort_within_batch=False, repeat=False, device=device)

'''
batch = next(iter(train_iter))
print(batch.text.t())
a = batch.text.t()
for id in a[0]:
  print(text.vocab.itos[id])
print(len(batch.label))
#print(len(text.vocab))
'''
class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.fc = nn.Linear(250*embedding_dim, hidden_dim)
        self.fc2tag = nn.Linear(hidden_dim, tagset_size)
        self.dropout = nn.Dropout(p = 0.3)

    def forward(self, sentence):
        batch_size = sentence.size(0)
        sentence = sentence.long()
        embeds = self.word_embeddings(sentence)   
        #print(len(embeds),len(embeds[0]),len(embeds[0][0]))
        embeds = embeds.view(-1,len(sentence[0])*embedding_dim)    
        fc_out = self.fc(embeds)
        tag_space = self.fc2tag(fc_out)
        tag_space = self.dropout(tag_space)
        tag_space = tag_space[:,-1]     
        return tag_space 
    
    
batch_size = 16
vocab_size = len(text.vocab) # +1 for the 0 padding + our word tokens
embedding_dim = 300
hidden_dim = 128

#net = LSTMTagger(embedding_dim, hidden_dim, vocab_size, 2)
net = LSTMTagger(embedding_dim, hidden_dim, vocab_size, 1).to(device) 
lr = 0.001
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

epochs = 10
for epoch in range(epochs):
    total = 0 
    net.train()
    for times, data in enumerate(train_iter, 0):
        inputs = data.text
        labels = torch.Tensor([float(label.vocab.itos[id]) for id in data.label]).to(device)
        #print(inputs,labels)
      
        #inputs, labels = inputs.to(device), labels.to(device)
        #h = net.init_hidden(labels.size(0))
        #h = tuple([each.data for each in h])
        # zero accumulated gradients
        net.zero_grad()
        
        # get the output from the model
        output = net(inputs)
        
     #   print(output.data)
        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels)
        
        loss.backward()
       
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), 5)
        optimizer.step()
        total += labels.size(0)
    #    viz.line([loss.item()], [(times+1)/len(train_loader)], win='train_loss', update='append')
        if times % 100 == 99 or times+1 == len(train_iter):
            print('[%d/%d, %d/%d] loss: %.3f' % (epoch+1, epochs, times+1, len(train_iter),loss.item()))
            viz.line([loss.item()], [(times)/len(train_iter)+epoch], win='train_loss', update='append')
            #print(inputs,loss.item(),labels.size(0))
            
    with torch.no_grad():
      total = 0
      correct = 0 
      net.eval()
      for times, data in enumerate(test_iter, 0):
          inputs = data.text
          labels = torch.Tensor([float(label.vocab.itos[id]) for id in data.label]).to(device)
        #print(inputs,labels)
        
        #for id in inputs[0]:
          #print(text.vocab.itos[id])
        #inputs, labels = inputs.to(device), labels.to(device)
         # h = net.init_hidden(labels.size(0))
       #   h = tuple([each.data for each in h])
        # zero accumulated gradients        
        # get the output from the model
          output = net(inputs)
        # calculate the loss and perform backprop
          loss = criterion(output.squeeze(), labels)
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
          total += labels.size(0)
          predict = torch.round(torch.sigmoid(output.squeeze()))
          print(predict,labels)
          correct  += (predict == labels).sum().item()
          
            
          acc = 100 * correct/total
          viz.line([loss.item()], [(times)/len(test_iter)+epoch], win='test_loss', update='append')
    #    viz.line([loss.item()], [(times+1)/len(train_loader)], win='train_loss', update='append')
          if times % 100 == 99 or times+1 == len(test_iter):
              print('[%d/%d, %d/%d] loss: %.3f' % (epoch+1, epochs, times+1, len(test_iter),loss.item()))
            
              viz.line([acc], [(times)/len(test_iter)+epoch], win='test_acc', update='append')
              print(inputs,loss.item(),labels.size(0))
              mat = confusion_matrix(labels.cpu(),predict.cpu())
              for idx in range(labels.size(0)):
                if(predict[idx] != labels[idx]):
                  for id in inputs[idx]:
                    print(text.vocab.itos[id])
                  print(predict[idx],labels[idx])
              sns.heatmap(mat,square= True, annot=True, cbar= True,fmt="d")
              plt.xlabel("predicted value")
              plt.ylabel("true value")
              plt.show()
