import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from tqdm.auto import tqdm
import transformers
from transformers import AutoTokenizer,Trainer,TrainingArguments,AutoModelForSequenceClassification,AdamW,get_scheduler,BertConfig,DistilBertTokenizerFast,DistilBertForSequenceClassification
from datasets import load_metric

model_checkpoint = "distilbert-base-uncased"

train_data = pd.read_csv("data/SNLI/snli-train.txt",sep = '\t')
test_data = pd.read_csv("data/SNLI/snli-test.txt",sep = '\t')
train_data = train_data[:1000]
test_data = train_data[:200]
def text_batch(data):
    text_sentence1 = []
    text_sentence2 = []
    labels = []
    for text1,text2,label in zip(data['sentence1'],data['sentence2'],data['gold_label']):
        text_sentence1.append(text1)
        text_sentence2.append(text2)
        labels.append(label)
    return text_sentence1,text_sentence2,labels
train_text_sentence1 ,train_text_sentence2,train_text_label = text_batch(train_data)
test_text_sentence1 ,test_text_sentence2,test_text_label = text_batch(test_data)

tokenizer_bert = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_tokenize_dataset = tokenizer_bert(train_text_sentence1,train_text_sentence2,padding='max_length',max_length=100,truncation=True)

test_tokenize_dataset = tokenizer_bert(test_text_sentence1,test_text_sentence2,padding='max_length',max_length=100,truncation=True)

#print(train_tokenize_dataset)


class SNIL_dataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = SNIL_dataset(train_tokenize_dataset,train_text_label)
test_dataset = SNIL_dataset(test_tokenize_dataset,test_text_label)

#print(train_dataset.__getitem__(10))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        #print(loss)
        loss.backward()
        optim.step()

# #model.eval()