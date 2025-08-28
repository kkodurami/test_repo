# 벌레 이미지 분류 프로세스 완벽 가이드

## 📋 프로젝트 개요
- **목적**: 4종류의 벌레 이미지를 자동으로 분류하는 AI 모델 개발
- **분류 대상**: 나방, 노린재, 담배가루이, 무당벌레
- **사용 기술**: TensorFlow/Keras, CNN, 전이학습 (MobileNetV2)

## 🔧 1단계: 환경 설정 및 준비

### Google Colab 환경 설정
```python
# Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 한글 폰트 설치
!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
```

### 필수 라이브러리 임포트
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
```

## 📁 2단계: 데이터 구조 설정

### 프로젝트 경로 설정
```python
PROJECT_NAME = "BugClassification"
PROJECT_PATH = f'/content/drive/MyDrive/{PROJECT_NAME}'
BUG_DATA_PATH = '/content/drive/MyDrive/Bug'
BUG_CLASSES = ['나방', '노린재', '담배가루이', '무당벌레']
```

### 예상 데이터 구조
```
Bug/
├── images/
│   ├── 나방/
│   ├── 노린재/
│   ├── 담배가루이/
│   └── 무당벌레/
└── labels/
```

## 📊 3단계: 데이터 전처리 및 분할

### 데이터셋 구조 확인
```python
def check_bug_dataset(base_path):
    """벌레 데이터셋 구조와 이미지 수 확인"""
    # 각 클래스별 이미지 파일 개수 확인
    # 데이터의 균형성 체크
```

### 훈련/검증/테스트 데이터 분할
- **훈련 데이터**: 70%
- **검증 데이터**: 20%
- **테스트 데이터**: 10%

```python
def create_train_val_test_split(source_path, target_path, 
                               train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    # sklearn의 train_test_split 사용
    # 각 클래스별로 균등하게 분할
```

## 🖼️ 4단계: 데이터 증강 및 제너레이터 생성

### 훈련 데이터 증강
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,          # 픽셀값 정규화
    rotation_range=20,       # 회전
    width_shift_range=0.2,   # 가로 이동
    height_shift_range=0.2,  # 세로 이동
    shear_range=0.2,         # 전단 변환
    zoom_range=0.2,          # 확대/축소
    horizontal_flip=True,    # 좌우 뒤집기
    fill_mode='nearest'      # 빈 공간 채우기
)
```

### 검증/테스트 데이터 (증강 없음)
```python
val_test_datagen = ImageDataGenerator(rescale=1./255)
```

## 🧠 5단계: 모델 아키텍처

### MobileNetV2 기반 전이학습 모델
```python
def create_bug_classification_model(num_classes=4, img_size=(224, 224)):
    # 사전 훈련된 MobileNetV2 백본
    base_model = keras.applications.MobileNetV2(
        input_shape=(*img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # 백본 고정
    base_model.trainable = False
    
    # 분류기 추가
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model
```

### 모델 구조
1. **MobileNetV2 백본**: ImageNet으로 사전 훈련된 특성 추출기
2. **Global Average Pooling**: 특성 맵을 벡터로 변환
3. **Dropout (0.2)**: 과적합 방지
4. **Dense (128)**: 완전연결층 + ReLU 활성화
5. **Batch Normalization**: 훈련 안정화
6. **Dropout (0.5)**: 추가 정규화
7. **Output Dense (4)**: 4클래스 분류 + Softmax

## 🚀 6단계: 모델 훈련

### 컴파일 및 콜백 설정
```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5, restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001
    ),
    keras.callbacks.ModelCheckpoint(
        'best_bug_model.h5', save_best_only=True, monitor='val_accuracy'
    )
]
```

### 주요 하이퍼파라미터
- **배치 크기**: 16
- **에포크**: 25
- **학습률**: 0.001 (적응적 감소)
- **조기 종료**: 검증 정확도 기준 5 에포크 patience

## 📈 7단계: 결과 분석 및 평가

### 훈련 과정 시각화
- 훈련/검증 정확도 그래프
- 훈련/검증 손실 그래프

### 모델 평가 지표
```python
def evaluate_model(model, test_generator):
    # 혼동 행렬 (Confusion Matrix)
    # 분류 리포트 (Precision, Recall, F1-score)
    # 전체 테스트 정확도
```

## 🔮 8단계: 실제 예측

### 단일 이미지 예측 함수
```python
def predict_bug_image(image_path, show_image=True):
    # 이미지 전처리 (크기 조정, 정규화)
    # 모델 예측
    # 결과 시각화 및 신뢰도 출력
```

## 🎯 핵심 특징

### 장점
1. **전이학습 활용**: ImageNet 사전 훈련 모델로 효율적 학습
2. **데이터 증강**: 제한된 데이터로도 robust한 모델 생성
3. **정규화 기법**: Dropout, BatchNorm으로 과적합 방지
4. **적응적 학습**: 학습률 감소 및 조기 종료로 최적 성능 달성

### 기술적 고려사항
1. **이미지 크기**: 224x224 (MobileNetV2 최적 크기)
2. **GPU 메모리 최적화**: 동적 메모리 할당
3. **한글 폰트 지원**: 결과 시각화를 위한 설정
4. **모델 체크포인트**: 최고 성능 모델 자동 저장

## 📊 예상 성능
- **일반적인 정확도**: 85-95% (데이터 품질에 따라)
- **훈련 시간**: GPU 기준 15-25분
- **메모리 사용량**: 배치 크기 16으로 최적화

## 🔧 사용법 요약
1. Google Drive에 벌레 이미지 데이터 업로드
2. 노트북 순차적 실행
3. 훈련된 모델로 새로운 이미지 분류
4. 결과 분석 및 모델 성능 평가

이 프로세스는 실제 농업이나 곤충학 연구에서 벌레 종 식별 자동화에 활용할 수 있는 실용적인 AI 시스템을 구축합니다.
