import os
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import uuid
import webbrowser
import threading
from io import BytesIO
from PIL import Image

# --- PyTorch 관련 라이브러리 추가 ---
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms

"""
====================================================================
[🚀 실행 전 필수 준비사항]
1. 현재 파일(server.py)이 위치한 곳에 'static' 이라는 이름의 빈 폴더를 만들어주세요.
2. 앞에서 작성한 프론트엔드(웹앱) 코드를 'static' 폴더 안에 'index.html' 이라는 이름으로 저장해주세요.
3. 필요한 파이썬 라이브러리가 없다면 터미널에서 설치해주세요:
   pip install fastapi uvicorn pydantic Pillow torch torchvision

[💻 실행 방법]
사용자는 터미널에서 오직 아래 명령어 하나만 실행하면 됩니다:
> python server.py
====================================================================
"""

app = FastAPI(title="이미지 수집 및 예측 백엔드 서버")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DATA_DIR = "./dataset/train"
os.makedirs(BASE_DATA_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)

# ==========================================
# 1. 수집(업로드) 관련 모델 및 파이프라인
# ==========================================
class ImageData(BaseModel):
    className: str
    type: str = "원본"
    dataUrl: str 

class ImageUploadRequest(BaseModel):
    images: List[ImageData]

transform_base = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
transform_flip = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.RandomHorizontalFlip(p=1.0)])
transform_rotate = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.RandomRotation(degrees=(15, 15))])

@app.post("/api/upload")
async def upload_images(request: ImageUploadRequest):
    # 기존 코드와 동일
    saved_count = 0
    saved_paths = []
    for img_data in request.images:
        try:
            class_dir = os.path.join(BASE_DATA_DIR, img_data.className)
            os.makedirs(class_dir, exist_ok=True)

            header, encoded = img_data.dataUrl.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            img = Image.open(BytesIO(image_bytes)).convert("RGB")

            augmented_images = {
                "원본(크기조정)": transform_base(img),
                "좌우반전": transform_flip(img),
                "15도회전": transform_rotate(img)
            }

            base_uuid = uuid.uuid4().hex[:8]
            for aug_type, aug_img in augmented_images.items():
                filename = f"{base_uuid}_{aug_type}.jpg"
                filepath = os.path.join(class_dir, filename)
                aug_img.save(filepath, "JPEG")
                saved_count += 1
                saved_paths.append(filepath)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"이미지 저장 실패: {str(e)}")

    return {"message": f"성공적으로 증강된 {saved_count}장의 이미지를 서버에 저장했습니다.", "saved_paths": saved_paths}

# ==========================================
# 2. 예측(Inference) 관련 모델 및 파이프라인
# ==========================================
# ⚠️ 중요: PyTorch의 ImageFolder는 폴더명을 알파벳 순서대로 읽습니다.
# 'cat'이 'dog'보다 알파벳이 앞서므로 cat=0, dog=1 로 학습됩니다.
# 예측 서버에서도 이 순서를 정확히 맞춰주어야 합니다!
CLASSES = ['cat', 'dog']

# 예측을 위한 전처리 (학습 때와 동일하게 텐서 변환 및 정규화 포함)
transform_inference = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# PyTorch 모델 초기화 (ResNet18)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(CLASSES)) # 분류기 교체

# 학습된 파라미터(가중치) 불러오기 시도
MODEL_PATH = 'my_custom_model.pth'
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    print(f"✅ 학습된 모델({MODEL_PATH})을 성공적으로 불러왔습니다.")
else:
    print(f"⚠️ 경고: 학습된 모델({MODEL_PATH})을 찾을 수 없습니다. 예측 결과가 부정확할 수 있습니다.")

model = model.to(device)
model.eval() # 평가 모드로 전환 (필수!)

class PredictRequest(BaseModel):
    dataUrl: str # Base64 인코딩된 이미지

@app.post("/api/predict")
async def predict_image(request: PredictRequest):
    try:
        # 1. Base64 이미지를 PIL Image로 변환
        header, encoded = request.dataUrl.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        img = Image.open(BytesIO(image_bytes)).convert("RGB")

        # 2. 이미지 전처리 및 텐서 변환 (배치 차원 추가)
        img_tensor = transform_inference(img).unsqueeze(0).to(device)

        # 3. 모델 추론
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1) # 확률로 변환
            confidence, predicted_idx = torch.max(probs, 1)

        # 4. 결과 추출
        predicted_class = CLASSES[predicted_idx.item()]
        confidence_score = confidence.item() * 100

        return {
            "class": predicted_class,
            "confidence": f"{confidence_score:.2f}"
        }

    except Exception as e:
        print(f"예측 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# 3. 프론트엔드 서빙
# ==========================================
@app.get("/")
def serve_frontend():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "⚠️ static 폴더에 웹앱 파일(index.html)이 없습니다."}

app.mount("/", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:8000")).start()
    print("🚀 서버와 웹앱을 시작합니다. 잠시 후 브라우저가 열립니다...")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)