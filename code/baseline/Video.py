import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)  # 设置启动方法为 spawn

import pandas as pd
import torch
import numpy as np
import h5py
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os
import torch.nn as nn
from transformers import TimesformerConfig, TimesformerModel, TimesformerForVideoClassification

class TimesFormerClassifier(nn.Module):
    def __init__(self, num_classes, input_dim=4096):
        super().__init__()
        self.input_dim = input_dim
    
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, 768),  # 从4096投影到768
            nn.LayerNorm(768),
            nn.GELU()
        )
        

        config = TimesformerConfig(
            image_size=24,     
            patch_size=24,     
            num_channels=1,   
            num_frames=8,      
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act='gelu',
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            layer_norm_eps=1e-6,
            qkv_bias=True,
            attention_type='divided_space_time',
            drop_path_rate=0.1
        )
        

        self.model = TimesformerModel(config)
        
        # 自定义分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, num_classes)
        )
        
        # 稳定的初始化
        self._init_weights()
    
    def _init_weights(self):
        for module in self.feature_projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        nn.init.xavier_uniform_(self.classifier[1].weight)
        nn.init.zeros_(self.classifier[1].bias)
    
    def forward(self, x):
        batch_size, num_frames, feature_dim = x.shape
        projected_features = self.feature_projection(x)  
        

        pseudo_images = projected_features.view(batch_size, num_frames, 1, 24, 32)
        
        outputs = self.model(pseudo_images)
        pooled_output = outputs.last_hidden_state[:, 0]
        
        logits = self.classifier(pooled_output)
        return logits

class VideoDataset(Dataset):
    def __init__(self, labels, post_ids, feature_folder, max_frames=8):
        self.labels = labels
        self.post_ids = post_ids
        self.feature_folder = feature_folder
        self.max_frames = max_frames
        
    def __len__(self):
        return len(self.post_ids)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        post_id = self.post_ids[idx]
        
        # 构造路径
        feature_path = os.path.join(self.feature_folder, f"{post_id}.hdf5")
        
        # 默认特征
        video_feature = torch.zeros(self.max_frames, 4096)

        if not os.path.exists(feature_path):
            print(f"[Missing File] Feature file not found for post_id: {post_id}")
        else:
            try:
                with h5py.File(feature_path, 'r') as f:
                    # 读取C3D特征
                    if str(post_id) in f and 'c3d_features' in f[str(post_id)]:
                        c3d_features = f[str(post_id)]['c3d_features'][:]
                        num_frames = min(c3d_features.shape[0], self.max_frames)
                        
                        # 随机或均匀采样帧特征
                        if c3d_features.shape[0] > self.max_frames:
                            # 均匀采样
                            indices = np.linspace(0, c3d_features.shape[0] - 1, self.max_frames, dtype=int)
                            video_feature = torch.tensor(c3d_features[indices], dtype=torch.float32)
                        else:
                            # 填充或截断
                            video_feature[:num_frames] = torch.tensor(c3d_features[:num_frames], dtype=torch.float32)
                    else:
                        print(f"[Empty or Invalid File] No 'c3d_features' found for post_id: {post_id}")
            except Exception as e:
                print(f"[Load Error] post_id: {post_id}, error: {e}")
        
        return {
            'video': video_feature,
            'labels': torch.tensor(label, dtype=torch.long)
        }

def calculate_metrics(test_df, test_preds, test_labels):
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_macro_f1 = f1_score(test_labels, test_preds, average='macro')
    test_micro_f1 = f1_score(test_labels, test_preds, average='micro')
    
    print("=== Overall Metrics ===")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Macro F1: {test_macro_f1:.4f}")
    print(f"Test Micro F1: {test_micro_f1:.4f}")
    
    print("\n=== Per Label Metrics ===")
    for label in sorted(set(test_labels)):
        precision = precision_score(test_labels, test_preds, pos_label=label, average='binary')
        recall = recall_score(test_labels, test_preds, pos_label=label, average='binary')
        f1 = f1_score(test_labels, test_preds, pos_label=label, average='binary')
        
        print(f"\nLabel {label}:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")

def train_and_evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    train_df = pd.read_csv('')
    val_df = pd.read_csv('')
    test_df = pd.read_csv('')
    
    # 编码标签
    label_encoder = LabelEncoder()
    train_df['encoded_label'] = label_encoder.fit_transform(train_df['label'])
    val_df['encoded_label'] = label_encoder.transform(val_df['label'])
    test_df['encoded_label'] = label_encoder.transform(test_df['label'])
    
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Label mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    # 视频特征文件夹路径
    feature_folder = ''
    
    # 创建数据集
    max_frames = 8
    
    train_dataset = VideoDataset(
        labels=train_df['encoded_label'].values,
        post_ids=train_df['post_id'].values,
        feature_folder=feature_folder,
        max_frames=max_frames
    )
    
    val_dataset = VideoDataset(
        labels=val_df['encoded_label'].values,
        post_ids=val_df['post_id'].values,
        feature_folder=feature_folder,
        max_frames=max_frames
    )
    
    test_dataset = VideoDataset(
        labels=test_df['encoded_label'].values,
        post_ids=test_df['post_id'].values,
        feature_folder=feature_folder,
        max_frames=max_frames
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=4)
    
    num_classes = len(label_encoder.classes_)
    model = TimesFormerClassifier(num_classes, input_dim=4096).to(device)
    
    # 优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    best_val_acc = 0
    epochs = 10
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_steps = 0
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}')
        for step, batch in progress_bar:

            videos = batch['video'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
            
            progress_bar.set_postfix({'loss': f"{train_loss/train_steps:.4f}"})
        

        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                videos = batch['video'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(videos)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
            }, 'best_model_timesformer.pth')
            print(f"  New best model saved! Val Acc: {val_acc:.4f}")
    
    # 测试最佳模型
    print("\n=== Testing Best Model ===")
    checkpoint = torch.load('best_model_timesformer.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_preds = []
    test_labels = []
    test_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            videos = batch['video'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(videos)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_labels.extend(labels.cpu().numpy())
    
    # 计算并显示测试指标
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")
    calculate_metrics(test_df, test_preds, test_labels)

if __name__ == '__main__':
    train_and_evaluate()