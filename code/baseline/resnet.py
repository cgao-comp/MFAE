import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms, models
from PIL import Image

# 加载ResNet模型
device = 'cuda'  # 选择GPU
resnet = models.resnet101(pretrained=True)
# 去掉最后的全连接层，用于提取特征
resnet = nn.Sequential(*list(resnet.children())[:-1]).to(device)
# 冻结ResNet的参数，只训练自定义分类器
for param in resnet.parameters():
    param.requires_grad = False

# 定义图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 定义数据集类
class ImageDataset(Dataset):
    def __init__(self, post_ids, labels, image_dir):
        self.post_ids = post_ids
        self.labels = labels
        self.image_dir = image_dir

    def __len__(self):
        return len(self.post_ids)

    def __getitem__(self, item):
        post_id = self.post_ids[item]
        label = self.labels[item]
        image_id = str(post_id).replace('.jpg', '')
        image_path = f"{self.image_dir}/{image_id}.jpg"
        image = Image.open(image_path).convert('RGB')
        image = transform(image)

        return {
            'image': image,
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

image_dir = ''  

# 创建数据集和DataLoader
train_dataset = ImageDataset(
    post_ids=train_df['post_id'].values,
    labels=train_df['encoded_label'].values,
    image_dir=image_dir
)

val_dataset = ImageDataset(
    post_ids=val_df['post_id'].values,
    labels=val_df['encoded_label'].values,
    image_dir=image_dir
)

test_dataset = ImageDataset(
    post_ids=test_df['post_id'].values,
    labels=test_df['encoded_label'].values,
    image_dir=image_dir
)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 定义一个简单的分类器
class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

num_classes = len(label_encoder.classes_)
classifier = Classifier(num_classes).to(device)

# 定义训练过程
optimizer = torch.optim.Adam(classifier.parameters(), lr=2e-3)
criterion = nn.CrossEntropyLoss()

# 训练过程
for epoch in range(10):
    classifier.train()
    for batch in train_dataloader:
        optimizer.zero_grad()

        image = batch['image'].to(device)
        labels = batch['labels'].to(device)

        features = resnet(image).squeeze()
        outputs = classifier(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 验证过程
    classifier.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for batch in val_dataloader:
            image = batch['image'].to(device)
            labels = batch['labels'].cpu().numpy()

            features = resnet(image).squeeze()
            outputs = classifier(features)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            val_preds.extend(preds)
            val_labels.extend(labels)

    val_accuracy = accuracy_score(val_labels, val_preds)
    print(f"Epoch {epoch + 1} - Validation Accuracy: {val_accuracy}")

# 测试过程
classifier.eval()
test_preds = []
test_labels = []

with torch.no_grad():
    for batch in test_dataloader:
        image = batch['image'].to(device)
        labels = batch['labels'].cpu().numpy()

        features = resnet(image).squeeze()
        outputs = classifier(features)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        test_preds.extend(preds)
        test_labels.extend(labels)

# 计算总体指标
test_accuracy = accuracy_score(test_labels, test_preds)
test_macro_f1 = f1_score(test_labels, test_preds, average='macro')
test_micro_f1 = f1_score(test_labels, test_preds, average='micro')

# 打印总体指标
print(f"Test Accuracy: {test_accuracy}")
print(f"Test Macro F1: {test_macro_f1}")
print(f"Test Micro F1: {test_micro_f1}")

# 计算标签0和1的precision, recall, F1
precision_0 = precision_score(test_labels, test_preds, pos_label=0)
recall_0 = recall_score(test_labels, test_preds, pos_label=0)
f1_0 = f1_score(test_labels, test_preds, pos_label=0)

precision_1 = precision_score(test_labels, test_preds, pos_label=1)
recall_1 = recall_score(test_labels, test_preds, pos_label=1)
f1_1 = f1_score(test_labels, test_preds, pos_label=1)

print(f"Precision (0): {precision_0}")
print(f"Recall (0): {recall_0}")
print(f"F1 (0): {f1_0}")

print(f"Precision (1): {precision_1}")
print(f"Recall (1): {recall_1}")
print(f"F1 (1): {f1_1}")