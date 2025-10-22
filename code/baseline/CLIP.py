import pandas as pd
import torch
import clip
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
import os

class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.clip_model = clip_model
        self.classifier = nn.Linear(512, num_classes) 

        # 更稳定的初始化
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, image, text):
        # 批量处理文本
        text = text.squeeze(1) if text.dim() == 3 else text  
        text_features = self.clip_model.encode_text(text).float()  
        image_features = self.clip_model.encode_image(image).float() 
        combined_features = image_features + text_features
        return self.classifier(combined_features)

class PostDataset(Dataset):
    def __init__(self, posts, labels, image_ids, image_folder, max_length=77):
        self.posts = posts
        self.labels = labels
        self.image_ids = image_ids
        self.image_folder = image_folder
        self.max_length = max_length
        _, self.preprocess = clip.load("ViT-B/32", device='cuda:0')  
    def __len__(self):
        return len(self.posts)

    def __getitem__(self, idx):
        text = str(self.posts[idx])
        label = self.labels[idx]
        image_id = self.image_ids[idx]

        # 文本处理保持在CPU
        text_tensor = clip.tokenize([text], truncate=True)[0] 

        # 图像处理保持在CPU
        image_path = f"{self.image_folder}/{image_id}"
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.preprocess(image).float()  
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = torch.zeros(3, 224, 224).float()  

        return {
            'text': text_tensor,
            'image': image,
            'labels': label  # 直接返回标量值
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
    for label in [0, 1]:
        precision = precision_score(test_labels, test_preds, pos_label=label, average='binary')
        recall = recall_score(test_labels, test_preds, pos_label=label, average='binary')
        f1 = f1_score(test_labels, test_preds, pos_label=label, average='binary')

        print(f"\nLabel {label}:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")

def train_and_evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载CLIP模型到指定设备，并确保为float32
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model = clip_model.float()

    train_df = pd.read_csv('')
    val_df = pd.read_csv('')
    test_df = pd.read_csv('')
    label_encoder = LabelEncoder()
    train_df['encoded_label'] = label_encoder.fit_transform(train_df['label'])
    val_df['encoded_label'] = label_encoder.transform(val_df['label'])
    test_df['encoded_label'] = label_encoder.transform(test_df['label'])

    image_folder = ''

    train_dataset = PostDataset(
        posts=train_df['original_post'].values,
        labels=train_df['encoded_label'].values,
        image_ids=train_df['image_id'].values,
        image_folder=image_folder,
        max_length=77
    )

    val_dataset = PostDataset(
        posts=val_df['original_post'].values,
        labels=val_df['encoded_label'].values,
        image_ids=val_df['image_id'].values,
        image_folder=image_folder,
        max_length=77
    )

    test_dataset = PostDataset(
        posts=test_df['original_post'].values,
        labels=test_df['encoded_label'].values,
        image_ids=test_df['image_id'].values,
        image_folder=image_folder,
        max_length=77
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, num_workers=4)

    num_classes = len(label_encoder.classes_)
    model = CLIPClassifier(clip_model, num_classes).to(device).float()  

    optimizer = torch.optim.Adam(model.parameters(), lr) 
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    for epoch in range(10):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            # 数据移动到设备
            images = batch['image'].to(device)
            texts = batch['text'].to(device)
            labels = torch.tensor(batch['labels']).to(device)

            optimizer.zero_grad()
            outputs = model(images, texts)
            loss = criterion(outputs, labels)

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=)

            loss.backward()
            optimizer.step()

            # 打印梯度和损失值以调试
            print(f"Epoch {epoch + 1}, Batch Loss: {loss.item():.4f}")

            train_loss += loss.item()

        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                # 确保数据在同一设备上
                images = batch['image'].to(device)
                texts = batch['text'].to(device)
                outputs = model(images, texts)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(batch['labels'].cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader):.4f}, Val Acc: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(model.classifier.state_dict(), 'best_model_clip.pth')

    print("\n=== Testing Best Model ===")

    state_dict = torch.load('best_model_clip.pth', weights_only=True)
    model.classifier.load_state_dict(state_dict)
    model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            images = batch['image'].to(device)
            texts = batch['text'].to(device)
            outputs = model(images, texts)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_labels.extend(batch['labels'].cpu().numpy())

    calculate_metrics(test_df, test_preds, test_labels)

if __name__ == '__main__':
    train_and_evaluate()