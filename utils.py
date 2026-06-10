import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size=32, use_augmentation=False):
    # Train 데이터용 변환
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)), # 200x200을 왜곡 없이 224x224로 확대
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10), # 정렬된 데이터이므로 회전은 10도 정도로 살짝만
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # Validation & Test 데이터용 변환
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 데이터셋 로드
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), test_transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), test_transform)

    # 데이터로더 생성
    loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    }

    return loaders, train_dataset.classes