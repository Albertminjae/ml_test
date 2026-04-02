import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights
import os

# ==========================================
# 0. 기본 설정
# ==========================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 기기: {device}")

data_dir = './dataset/train' # 웹앱에서 데이터가 저장되는 최상위 폴더

# ==========================================
# 1. 데이터 불러오기 및 전처리 (증강 없음)
# ==========================================
# 웹앱 백엔드에서 이미 증강(자르기/회전/반전)을 마친 상태로 저장되므로, 
# 학습 코드에서는 정직하게 크기 조정 및 텐서 변환만 수행합니다.
base_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 폴더에서 데이터 불러오기
# 구조: ./dataset/train/cat/ , ./dataset/train/dog/
image_dataset = datasets.ImageFolder(data_dir, base_transform)

# DataLoader 생성 (배치 단위 공급)
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=32, shuffle=True, num_workers=2)

dataset_size = len(image_dataset)
class_names = image_dataset.classes # 폴더명 기준 정렬 ['cat', 'dog']
num_classes = len(class_names)

print(f"학습 데이터 총 개수: {dataset_size}")
print(f"분류할 클래스: {class_names}")
print(f"클래스-인덱스 매핑: {image_dataset.class_to_idx}") # 예: {'cat': 0, 'dog': 1}

# ==========================================
# 2. 전이 학습 모델 준비
# ==========================================
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# 특징 추출기 가중치 고정
for param in model.parameters():
    param.requires_grad = False

# 마지막 분류기(FC Layer) 교체
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

# ==========================================
# 3. 손실 함수와 옵티마이저
# ==========================================
criterion = nn.CrossEntropyLoss()
# 새로 교체한 분류기(model.fc)만 학습
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# ==========================================
# 4. 학습 루프 (Training Loop)
# ==========================================
num_epochs = 10

# 검증 데이터셋(val)을 따로 나누지 않고 전체 데이터로만 학습하는 간략화 버전
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * 10)

    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects.double() / dataset_size

    print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

print('학습 완료!')

# 모델 저장 (백엔드 서버가 이 파일을 읽어서 예측합니다)
torch.save(model.state_dict(), 'my_custom_model.pth')
print('✅ 모델이 my_custom_model.pth 에 저장되었습니다. 백엔드 서버를 재시작하면 예측에 반영됩니다.')