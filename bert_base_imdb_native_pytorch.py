from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast,DistilBertForSequenceClassification, Trainer, TrainingArguments,AdamW
import torch
from datasets import load_metric
from torch.utils.data import DataLoader
def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text(errors='ignore'))
            labels.append(0 if label_dir is "neg" else 1)

    return texts, labels

train_text, train_label = read_imdb_split('aclImdb/train')
test_text, test_label = read_imdb_split('aclImdb/test')

train_texts, train_labels = [],[]
test_texts, test_labels = [],[]
train_texts.extend(train_text[:100])
train_texts.extend(train_text[-100:-1])
train_labels.extend(train_label[:100])
train_labels.extend(train_label[-100:-1])
test_texts.extend(test_text[:20])
test_texts.extend(test_text[-20:-1])
test_labels.extend(test_label[:20])
test_labels.extend(test_label[-20:-1])

#print(train_labels)
#print (train_texts)
#print(len(train_texts))
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

#print(train_texts)
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)
#print(train_encodings)
#print(train_encodings['input_ids'])

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)

train_loarder = DataLoader(train_dataset,batch_size = 1,shuffle = True)
test_loader = DataLoader(test_dataset,batch_size=1,shuffle=True)

device =torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model =DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.to(device)
model.train()

optim = AdamW(model.parameters(),lr=5e-5)

for epoch in range(3):
    for step,batch in enumerate(train_loarder):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        # print(loss)
        loss.backward()
        optim.step()
        if step % 50 == 0 :
            print('train_dataset_loss:',loss)
            # print('/t')
            # num = 0
            # correct = 0
            # for batch in test_loader:
            #     input_ids = batch['input_ids'].to(device)
            #     attention_mask = batch['attention_mask'].to(device)
            #     labels = batch['labels'].to(device)
            #     outputs = model(input_ids, attention_mask=attention_mask)
            #     print(outputs.logits)

#model.eval()
#torch.save(model.state_dict_,'model.pth')