import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_dataloaders
from model import get_model
from trainer import train_model

def main():
    # ==========================================
    # [실험 설정] 이 부분의 값을 변경해가며 보고서용 테스트를 진행하세요!
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
    
    # 결과가 저장될 모델 파일 이름 설정
    SAVE_PATH = f'best_model_{MODEL_NAME}_aug{USE_AUGMENTATION}_pre{PRETRAINED}.pth'
    # ==========================================

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] 실행 환경: {device}")
    print(f"[*] 모델: {MODEL_NAME} | 사전학습: {PRETRAINED} | 증강: {USE_AUGMENTATION}")

    # 데이터 로드
    loaders, classes = get_dataloaders(DATA_DIR, BATCH_SIZE, USE_AUGMENTATION)
    print(f"[*] 클래스: {classes}")

    # 모델 초기화
    model = get_model(model_name=MODEL_NAME, num_classes=len(classes), pretrained=PRETRAINED)

    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 학습 시작
    train_model(model, loaders, criterion, optimizer, EPOCHS, device, SAVE_PATH)

if __name__ == '__main__':
    main()