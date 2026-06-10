import torch
import time
from utils import get_dataloaders
from model import get_model
from sklearn.metrics import confusion_matrix, classification_report

def main():
    # ==========================================
    # 평가할 모델의 설정값을 입력하세요 (train.py 설정과 동일하게)
    MODEL_NAME = 'resnet'
    SAVED_MODEL_PATH = 'best_model_resnet_augTrue_preTrue.pth' # 테스트할 가중치 파일명
    DATA_DIR = './dataset'
    # ==========================================

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Test 로더만 가져오기 (Augmentation 무조건 False)
    loaders, classes = get_dataloaders(DATA_DIR, batch_size=32, use_augmentation=False)
    test_loader = loaders['test']

    # 모델 준비 및 가중치 로드
    model = get_model(model_name=MODEL_NAME, num_classes=len(classes), pretrained=False)
    model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    
    print(f"[*] {SAVED_MODEL_PATH} 모델의 Test 데이터 평가를 시작합니다...")
    
    # Inference Time 측정을 위한 타이머 시작
    start_time = time.time()
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    end_time = time.time()
    
    # 결과 계산
    total_time = end_time - start_time
    total_images = len(all_labels)
    time_per_image = total_time / total_images

    # 보고서 작성용 출력
    print("\n" + "="*40)
    print(" [최종 성능 평가 결과]")
    print("="*40)
    print(f"1. 총 평가 이미지 수: {total_images} 장")
    print(f"2. 전체 추론 시간: {total_time:.2f} 초")
    print(f"3. 1장당 평균 추론 시간 (Inference Time): {time_per_image:.4f} 초/장")
    print("\n[Classification Report]")
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    print("\n[Confusion Matrix (오차 행렬)]")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    print("\n(행: 실제 정답, 열: 모델의 예측)")
    print("="*40)

if __name__ == '__main__':
    main()