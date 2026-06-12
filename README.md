# 프로젝트 사전 구성

데이터셋을 다운로드 받는다.

[UTKFace Dataset](https://www.kaggle.com/datasets/moritzm00/utkface-cropped)

프로젝트 폴더를 아래와 같이 구성한다.

```bash
root 폴더
├─UTKFace/
├─inference.ipynb
├─model.py
├─predict.py
├─prepare_data.py
├─train.py
├─trainer.py
└─utils.py
```

# 프로젝트 환경구성

aws ec2를 이용한 실행방법 로컬환경에서 실행하는 방법이 있다.

## 로컬환경

1. 가상환경 생성 및 실행
```bash
# 가상환경 생성 (Windows/Mac 공통)
python -m venv venv
# 가상환경 활성화 (Windows)
venv\Scripts\activate
# 가상환경 활성화 (Mac/Linux)
source venv/bin/activate
```
2. 필수 라이브러리 설치
```bash
# PyTorch 설치 (자신의 로컬 환경에 맞는 CUDA 버전으로 설치 권장)
pip install torch torchvision torchaudio

# 컴퓨터 비전 및 YOLO 관련 필수 라이브러리 설치 (버전 고정)
pip install "numpy<2" "opencv-python<4.10.0" ultralytics scikit-learn pillow
```

## AWS EC2환경

1. 패키지 매니저 설치 및 환경 업데이트
```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git libgl1-mesa-glx
```

2. git clone을 통해 프로젝트 세팅
3. 
