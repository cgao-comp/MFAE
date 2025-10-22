import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder

# 加载本地的RoBERTa模型和分词器
model_name = ""
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = 'cuda'  # 选择GPU
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

# 定义数据集类
class PostDataset(Dataset):
    def __init__(self, posts, labels, tokenizer, max_len):
        self.posts = posts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.posts)
    
    def __getitem__(self, item):
        # 确保输入文本是字符串类型
        text = str(self.posts[item])  # Convert to string if necessary
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }
train_df = pd.read_csv('')  
val_df = pd.read_csv('')  
test_df = pd.read_csv('') 

# 标签编码
label_encoder = LabelEncoder()
train_df['encoded_label'] = label_encoder.fit_transform(train_df['label'])
val_df['encoded_label'] = label_encoder.transform(val_df['label'])  
test_df['encoded_label'] = label_encoder.transform(test_df['label'])  

# 创建数据集和DataLoader
train_dataset = PostDataset(
    posts=train_df['original_post'].values,
    labels=train_df['encoded_label'].values,
    tokenizer=tokenizer,
    max_len=128
)

val_dataset = PostDataset(
    posts=val_df['original_post'].values,
    labels=val_df['encoded_label'].values,
    tokenizer=tokenizer,
    max_len=128
)

test_dataset = PostDataset(
    posts=test_df['original_post'].values,
    labels=test_df['encoded_label'].values,
    tokenizer=tokenizer,
    max_len=128
)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 冻结模型的其他部分，只训练分类器部分
for param in model.base_model.parameters():
    param.requires_grad = False

# 只训练最后的分类器层
for param in model.classifier.parameters():
    param.requires_grad = True

# 定义训练过程
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# 训练过程
for epoch in range(10):
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # 验证过程
    model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            val_preds.extend(preds)
            val_labels.extend(labels)

    val_accuracy = accuracy_score(val_labels, val_preds)
    print(f"Epoch {epoch + 1} - Validation Accuracy: {val_accuracy}")

# 测试过程
model.eval()
test_preds = []
test_labels = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].cpu().numpy()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

        test_preds.extend(preds)
        test_labels.extend(labels)

# 计算总体指标
test_accuracy = accuracy_score(test_labels, test_preds)
test_macro_f1 = f1_score(test_labels, test_preds, average='macro')
test_micro_f1 = f1_score(test_labels, test_preds, average='micro')

# 打印总体指标
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Macro F1: {test_macro_f1:.4f}")
print(f"Test Micro F1: {test_micro_f1:.4f}")

# 计算标签0和1的precision, recall, F1
# 不再使用pos_label参数，改为使用labels参数
precision_0 = precision_score(test_labels, test_preds, labels=[0], average='binary')
recall_0 = recall_score(test_labels, test_preds, labels=[0], average='binary')
f1_0 = f1_score(test_labels, test_preds, labels=[0], average='binary')

precision_1 = precision_score(test_labels, test_preds, labels=[1], average='binary')
recall_1 = recall_score(test_labels, test_preds, labels=[1], average='binary')
f1_1 = f1_score(test_labels, test_preds, labels=[1], average='binary')

print(f"\nMetrics for class 0:")
print(f"Precision (0): {precision_0:.4f}")
print(f"Recall (0): {recall_0:.4f}")
print(f"F1 (0): {f1_0:.4f}")

print(f"\nMetrics for class 1:")
print(f"Precision (1): {precision_1:.4f}")
print(f"Recall (1): {recall_1:.4f}")
print(f"F1 (1): {f1_1:.4f}")
