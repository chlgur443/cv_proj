import os
import shutil
import random

# --- 설정 (Configuration) ---
RAW_DATA_DIR = './UTKFace'  
TARGET_DIR = './dataset'    
SPLIT_RATIO = (0.7, 0.15, 0.15) # Train, Val, Test 비율
CLASSES = ['0_Infant', '1_Teen', '2_Adult', '3_Senior']

def get_class_name(age):
    if age <= 7: return CLASSES[0]
    elif age <= 19: return CLASSES[1]
    elif age <= 64: return CLASSES[2]
    else: return CLASSES[3]

def main():
    # 1. 폴더 생성
    for split in ['train', 'val', 'test']:
        for cls in CLASSES:
            os.makedirs(os.path.join(TARGET_DIR, split, cls), exist_ok=True)

    # 2. 파일 목록 불러오기 및 셔플
    all_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.jpg')]
    random.seed(42)
    random.shuffle(all_files)

    # 3. 분할 인덱스 계산
    total = len(all_files)
    train_end = int(total * SPLIT_RATIO[0])
    val_end = train_end + int(total * SPLIT_RATIO[1])

    datasets = {
        'train': all_files[:train_end],
        'val': all_files[train_end:val_end],
        'test': all_files[val_end:]
    }

    # 4. 파일 복사 및 분류
    print("데이터 분류 작업을 시작합니다...")
    for split_name, files in datasets.items():
        count = 0
        for filename in files:
            try:
                age = int(filename.split('_')[0])
                class_name = get_class_name(age)
                
                src = os.path.join(RAW_DATA_DIR, filename)
                dst = os.path.join(TARGET_DIR, split_name, class_name, filename)
                
                shutil.copy2(src, dst)
                count += 1
            except ValueError:
                continue # 파일명 규칙에 맞지 않는 파일은 무시
        print(f"[{split_name.upper()}] {count}장 세팅 완료")

if __name__ == '__main__':
    main()