import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from transformers import HubertModel, AutoFeatureExtractor
import librosa
import numpy as np
from tqdm import tqdm

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载本地的HuBERT模型和特征提取器
model_name = ''
local_model_path = ''
feature_extractor = AutoFeatureExtractor.from_pretrained(local_model_path)
hubert = HubertModel.from_pretrained(local_model_path).to(device)

# 冻结HuBERT的参数
for param in hubert.parameters():
    param.requires_grad = False

class AudioDataset(Dataset):
    def __init__(self, post_ids, labels, audio_dir, max_length=160000): 
        self.post_ids = post_ids
        self.labels = labels
        self.audio_dir = audio_dir
        self.max_length = max_length
        self.feature_extractor = feature_extractor
        
    def __len__(self):
        return len(self.post_ids)
    
    def pad_or_trim(self, audio):
        # 填充或截断音频到固定长度
        if len(audio) > self.max_length:
            return audio[:self.max_length]
        else:
            return np.pad(audio, (0, self.max_length - len(audio)), 'constant')
    
    def __getitem__(self, idx):
        try:
            post_id = self.post_ids[idx]
            label = self.labels[idx]
            audio_id = str(post_id).replace('.mp3', '')
            audio_path = f"{self.audio_dir}/{audio_id}.wav"
            
            # 加载并预处理音频
            audio, sr = librosa.load(audio_path, sr=16000)
            audio = self.pad_or_trim(audio)
            
            # 使用feature_extractor处理音频
            inputs = self.feature_extractor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )
            input_values = inputs.input_values.squeeze()
            
            return {
                'audio': input_values,
                'labels': torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {str(e)}")
            # 返回一个空音频和标签
            return {
                'audio': torch.zeros(self.max_length),
                'labels': torch.tensor(label, dtype=torch.long)
            }

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(768, 256)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # 对时间维度求平均
        x = torch.mean(x, dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_epoch(model, hubert, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    preds_list = []
    labels_list = []
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        audio = batch['audio'].to(device)
        labels = batch['labels'].to(device)
        
        with torch.no_grad():
            outputs = hubert(audio)
            features = outputs.last_hidden_state
            
        logits = model(features)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        
        preds_list.extend(preds.cpu().numpy())
        labels_list.extend(labels.cpu().numpy())
    
    epoch_loss = total_loss / len(dataloader)
    epoch_acc = accuracy_score(labels_list, preds_list)
    return epoch_loss, epoch_acc

def evaluate(model, hubert, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            audio = batch['audio'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = hubert(audio)
            features = outputs.last_hidden_state
            logits = model(features)
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = total_loss / len(dataloader)
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return val_loss, val_acc, val_f1, all_preds, all_labels

# 主要训练和评估流程
def main():
    # 读取数据
    train_df = pd.read_csv('')
    val_df = pd.read_csv('')
    test_df = pd.read_csv('')
    # 标签编码
    label_encoder = LabelEncoder()
    train_df['encoded_label'] = label_encoder.fit_transform(train_df['label'])
    val_df['encoded_label'] = label_encoder.transform(val_df['label'])
    test_df['encoded_label'] = label_encoder.transform(test_df['label'])
    
    audio_dir = ''
    
    # 创建数据集
    train_dataset = AudioDataset(
        post_ids=train_df['post_id'].values,
        labels=train_df['encoded_label'].values,
        audio_dir=audio_dir
    )
    
    val_dataset = AudioDataset(
        post_ids=val_df['post_id'].values,
        labels=val_df['encoded_label'].values,
        audio_dir=audio_dir
    )
    
    test_dataset = AudioDataset(
        post_ids=test_df['post_id'].values,
        labels=test_df['encoded_label'].values,
        audio_dir=audio_dir
    )
    
    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    # 初始化模型
    num_classes = len(label_encoder.classes_)
    classifier = Classifier(num_classes).to(device)
    
    # 训练参数
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3)
    num_epochs =
    best_val_f1 =
    
    # 训练循环
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # 训练阶段
        train_loss, train_acc = train_epoch(
            classifier, hubert, train_dataloader, criterion, optimizer, device
        )
        
        # 验证阶段
        val_loss, val_acc, val_f1, _, _ = evaluate(
            classifier, hubert, val_dataloader, criterion, device
        )
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(classifier.state_dict(), 'best_model.pth')
    
    # 加载最佳模型进行测试
    classifier.load_state_dict(torch.load('best_model.pth'))
    _, test_acc, test_f1, test_preds, test_labels = evaluate(
        classifier, hubert, test_dataloader, criterion, device
    )
    
    # 计算总的Micro F1
    test_micro_f1 = f1_score(test_labels, test_preds, average='micro')
    
    # 计算详细指标
    print("\nTest Results:")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")
    print(f"Test Micro F1: {test_micro_f1:.4f}")
    
    # 计算每个类别的指标
    for label in range(num_classes):
        precision = precision_score(test_labels, test_preds, labels=[label], average='micro')
        recall = recall_score(test_labels, test_preds, labels=[label], average='micro')
        f1 = f1_score(test_labels, test_preds, labels=[label], average='micro')
        print(f"\nClass {label} metrics:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
    
    

if __name__ == "__main__":
    main()