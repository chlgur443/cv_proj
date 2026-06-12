# 프로젝트 사전 구성

데이터셋을 다운로드 받는다.

[UTKFace Dataset](https://www.kaggle.com/datasets/moritzm00/utkface-cropped)

# 프로젝트 환경구성

aws ec2를 이용한 실행방법 로컬환경에서 실행하는 방법이 있다.

## 로컬환경

1. 가상환경 생성 및 실행
```
# 가상환경 생성 (Windows/Mac 공통)
python -m venv venv
# 가상환경 활성화 (Windows)
venv\Scripts\activate
# 가상환경 활성화 (Mac/Linux)
source venv/bin/activate
```
2. 필수 라이브러리 설치
```
# PyTorch 설치 (자신의 로컬 환경에 맞는 CUDA 버전으로 설치 권장)
pip install torch torchvision torchaudio

# 컴퓨터 비전 및 YOLO 관련 필수 라이브러리 설치 (버전 고정)
pip install "numpy<2" "opencv-python<4.10.0" ultralytics scikit-learn pillow
```

## AWS EC2환경

1. 패키지 매니저 설치 및 환경 업데이트
```
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git libgl1-mesa-glx
```

2. git clone을 통해 프로젝트 세팅

데이터셋은 용량이 크므로 깃허브에 업로드가 불가해 따로 업로드 해줘야함

3. 가상환경 생성 및 실행
```
python3 -m venv venv
source venv/bin/activate
```

4. 필수 라이브러리 설치
```
# AWS Linux 환경에 맞춘 패키지 설치
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2" "opencv-python-headless<4.10.0" ultralytics scikit-learn
```

# 프로젝트 실행

프로젝트 폴더를 아래와 같이 구성한다.

```
├─UTKFace/
├─inference.ipynb
├─model.py
├─predict.py
├─prepare_data.py
├─train.py
├─trainer.py
└─utils.py
```

# 학습

데이터셋 분할
```
python prepare_data.py
```
실행하면 데이터셋이 랜덤으로 75/15/15 | train/val/test 비율로 나뉜다.

프로젝트 파일은 아래와 같이 구성된다.

```
├─dataset
│  ├─test
│  │  ├─0_Infant
│  │  ├─1_Teen
│  │  ├─2_Adult
│  │  └─3_Senior
│  ├─train
│  │  ├─0_Infant
│  │  ├─1_Teen
│  │  ├─2_Adult
│  │  └─3_Senior
│  └─val
│      ├─0_Infant
│      ├─1_Teen
│      ├─2_Adult
│      └─3_Senior
├─UTKFace/
├─inference.ipynb
├─model.py
├─predict.py
├─prepare_data.py
├─train.py
├─trainer.py
└─utils.py
```

데이터셋의 분할 비율이나 기준 나이값을 변경하고 싶으면 prepare_data.py의 아래부분을 수정해준다.
```
SPLIT_RATIO = (0.7, 0.15, 0.15) # Train, Val, Test 비율

def get_class_name(age):
    if age <= 7: return CLASSES[0]
    elif age <= 19: return CLASSES[1]
    elif age <= 64: return CLASSES[2]
    else: return CLASSES[3]
```
데이터셋 까지 준비가 되면 train.py를 실행시켜 학습을 시작한다.

```
python train.py
```

학습 관련 설정은 train.py의 아래부분을 수정하면 된다.

```
    # ==========================================
    # [실험 설정]
    # ==========================================
    DATA_DIR = './dataset'
    BATCH_SIZE = 64
    EPOCHS = 10                  # 시간 관계상 10~15 정도로 설정
    LEARNING_RATE = 0.001
    
    # [테스트 요소 1] 'resnet' vs 'mobilenet'
    MODEL_NAME = 'resnet'        
    
    # [테스트 요소 2] True(사전학습 O) vs False(처음부터 학습)
    PRETRAINED = True            
    
    # [테스트 요소 3] True(데이터 증강 O) vs False(단순 크기조절만)
    USE_AUGMENTATION = True
```
학습 log

학습을 진행하면서 최적 모델이 나오면 저장한다.
```
[*] 실행 환경: cuda:0
[*] 모델: resnet | 사전학습: True | 증강: True
[*] 클래스: ['0_Infant', '1_Teen', '2_Adult', '3_Senior']
Epoch 1/10
---------------
Train Loss: 0.4358 | Acc: 0.8491
Val Loss: 0.4648 | Acc: 0.8420
>>> [Best Model Saved] Acc: 0.8420

Epoch 2/10
---------------
Train Loss: 0.3283 | Acc: 0.8806
Val Loss: 0.4096 | Acc: 0.8366

Epoch 3/10
---------------
Train Loss: 0.2906 | Acc: 0.8960
Val Loss: 0.3028 | Acc: 0.8875
>>> [Best Model Saved] Acc: 0.8875
```

학습이 완료되면 설정에 따라 best_model_resnet_augTrue_preTrue.pth 모델이 생성된다.

만약 설정을 다르게 변경했다면 predict.py의 아래부분을 생성된 모델과 train.py의 설정과 맞게 변경해준다.

```
    # 평가할 모델의 설정값을 입력하세요 (train.py 설정과 동일하게)
    MODEL_NAME = 'resnet'
    SAVED_MODEL_PATH = 'best_model_resnet_augTrue_preTrue.pth' # 테스트할 가중치 파일명
    DATA_DIR = './dataset'
```

predict.py를 실행하여 모델의 성능을 테스트한다.

```
python predict.py
```

실행결과
```
[*] best_model_resnet_augTrue_preTrue.pth 모델의 Test 데이터 평가를 시작합니다...

========================================
 [최종 성능 평가 결과]
========================================
1. 총 평가 이미지 수: 3557 장
2. 전체 추론 시간: 10.44 초
3. 1장당 평균 추론 시간 (Inference Time): 0.0029 초/장

[Classification Report]
              precision    recall  f1-score   support

    0_Infant       0.98      0.85      0.91       401
      1_Teen       0.66      0.72      0.69       301
     2_Adult       0.93      0.96      0.94      2573
    3_Senior       0.77      0.62      0.69       282

    accuracy                           0.90      3557
   macro avg       0.83      0.79      0.81      3557
weighted avg       0.90      0.90      0.90      3557


[Confusion Matrix (오차 행렬)]
[[ 339   58    3    1]
 [   6  216   79    0]
 [   0   54 2467   52]
 [   0    0  107  175]]

(행: 실제 정답, 열: 모델의 예측)
========================================
```
#테스트

colab에서 inference.ipynb를 실행해서 저장한 모델을 통해서 추론할 수 있다.

라이브러리를 설치후 세션 재시작을 한다.
```
!pip install "numpy<2" "opencv-python<4.10.0" ultralytics facenet-pytorch
```
세션 재시작 후에는 실행 X

라이브러리들을 임포트 해준다.
```
import os
import cv2
import torch
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
from IPython.display import display
from model import get_model
```

저장한 모델과 테스트할 사진을 업로드하여 아래 코드를 모델이름과 사진이름게 맞게 수정한다.

```
MODEL_NAME = 'resnet'
SAVED_MODEL_PATH = '/content/best_model_resnet_augTrue_preFalse.pth'
IMAGE_PATH = '/content/my_photo.jpg' # 단체 사진 파일명
```

저장결과는 colab content폴더에 final_yolo_result.jpg로 저장된다.
