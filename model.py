import torch.nn as nn
from torchvision import models

def get_model(model_name='resnet', num_classes=4, pretrained=True):
    # PyTorch 최신 버전에 맞춘 weights 파라미터 적용
    weights_param = 'DEFAULT' if pretrained else None
    
    if model_name == 'resnet':
        model = models.resnet50(weights=weights_param)
        # 마지막 분류기(FC layer)를 4개 클래스에 맞게 수정
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'mobilenet':
        model = models.mobilenet_v3_large(weights=weights_param)
        # MobileNet의 마지막 분류기 수정
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)
        
    else:
        raise ValueError("model_name은 'resnet' 또는 'mobilenet'이어야 합니다.")
        
    return model