import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights

# ==========================================
# 0. 기본 설정
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset" / "train"
MODEL_PATH = BASE_DIR / "my_custom_model.pth"
CLASSES_PATH = BASE_DIR / "classes.json"
SUPPORTED_CLASSES = ["cat", "dog"]


def normalize_class_name(name: str) -> str:
    return name.strip().lower()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 기기: {device}")
print(f"데이터 경로: {DATA_DIR}")
if not DATA_DIR.exists():
    raise FileNotFoundError(f"학습 데이터 경로가 없습니다: {DATA_DIR}")

# ==========================================
# 1. 데이터 불러오기 및 전처리
# ==========================================
base_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 폴더에서 데이터 불러오기
# 구조: ./dataset/train/cat/ , ./dataset/train/dog/
image_dataset = datasets.ImageFolder(str(DATA_DIR), base_transform)
raw_class_names = image_dataset.classes
class_names = [normalize_class_name(name) for name in raw_class_names]

if sorted(class_names) != SUPPORTED_CLASSES:
    raise ValueError(
        f"지원 클래스는 {SUPPORTED_CLASSES} 뿐입니다. 현재 폴더 클래스: {raw_class_names}"
    )

num_classes = len(class_names)
dataloader = torch.utils.data.DataLoader(
    image_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
)

dataset_size = len(image_dataset)
class_mapping = {
    normalized: image_dataset.class_to_idx[raw]
    for raw, normalized in zip(raw_class_names, class_names)
}

print(f"학습 데이터 총 개수: {dataset_size}")
print(f"원본 클래스 폴더: {raw_class_names}")
print(f"학습 클래스: {class_names}")
print(f"클래스-인덱스 매핑: {class_mapping}")

# ==========================================
# 2. 전이 학습 모델 준비
# ==========================================
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

# ==========================================
# 3. 손실 함수와 옵티마이저
# ==========================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# ==========================================
# 4. 학습 루프
# ==========================================
num_epochs = 10

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print("-" * 10)

    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects.double() / dataset_size
    print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n")

print("학습 완료!")

torch.save(model.state_dict(), MODEL_PATH)
CLASSES_PATH.write_text(json.dumps(class_names, ensure_ascii=False, indent=2) + "\n")
print(f"✅ 모델이 {MODEL_PATH} 에 저장되었습니다.")
print(f"✅ 클래스 정보가 {CLASSES_PATH} 에 저장되었습니다.")
