
# DDIM 코드 해설 (`ddim.py`)

이 문서는 `ddim.py` 스크립트의 전체 코드에 대한 한국어 해설입니다. 각 섹션은 PDF의 `프로그램 D-1` 구조를 따르며, 코드의 각 부분이 어떤 역할을 하는지 설명합니다.

---

### **1. 라이브러리 임포트 (01-06행)**

스크립트 실행에 필요한 기본 도구들을 불러옵니다.

```python
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import layers
```

- **`math`**: 수학 계산(예: `math.pi`)을 위해 필요합니다.
- **`matplotlib.pyplot`**: 학습 중 생성된 이미지를 시각화하여 보여주는 데 사용됩니다.
- **`tensorflow`**: 구글에서 개발한 핵심 딥러닝 프레임워크입니다.
- **`tensorflow_datasets`**: `oxford_flowers102`와 같은 표준 데이터셋을 쉽게 불러올 수 있게 해줍니다.
- **`keras`**: 텐서플로의 고수준 API로, 신경망을 더 쉽고 직관적으로 만들 수 있도록 도와줍니다.

---

### **2. 하이퍼파라미터 설정 (08-23행)**

모델의 학습 방식과 구조를 결정하는 주요 변수들을 미리 정의합니다.

```python
dataset_name = "oxford_flowers102"
image_size = 64
batch_size = 64
...
```

- **`dataset_name`**: 학습에 사용할 데이터셋의 이름입니다.
- **`image_size`**: 모델이 다룰 이미지의 크기를 64x64 픽셀로 지정합니다.
- **`batch_size`**: 한 번에 처리할 이미지의 개수입니다.
- **`kid_...`**: 생성된 이미지의 품질을 평가하는 지표인 KID(Kernel Inception Distance) 계산에 사용될 값들입니다.
- **`min/max_signal_rate`**: 확산 과정(이미지에 노이즈를 섞는 과정)의 강도를 조절하는 핵심 파라미터입니다.
- **`embedding_dims`, `widths`, `block_depth`**: U-Net 신경망의 구조(깊이, 너비 등)를 결정합니다.

---

### **3. 데이터 파이프라인 (25-40행)**

데이터셋을 불러와 모델이 학습할 수 있는 형태로 가공하는 과정입니다.

```python
def preprocess_image(data):
    # ... 이미지를 정사각형으로 자르고 크기 조절 ...
    image = tf.image.resize(image, size=[image_size, image_size], antialias=True)
    # ... 픽셀 값을 0.0 ~ 1.0 사이로 정규화 ...
    return ops.clip(image / 255.0, 0.0, 1.0)

def prepare_dataset(split):
    # ... 데이터셋을 불러와 전처리, 셔플, 배치화 등 수행 ...

# 데이터셋을 80%는 학습용, 20%는 검증용으로 분리
train_dataset = prepare_dataset("train[:80%]+validation[:80%]+test[:80%]")
val_dataset = prepare_dataset("train[80%:]+validation[80%:]+test[80%:]")
```

- **`preprocess_image` 함수**: 개별 이미지를 받아 중앙을 잘라내고, 지정된 크기(64x64)로 맞춘 뒤, 픽셀 값을 0과 1 사이로 정규화합니다.
- **`prepare_dataset` 함수**: 데이터셋 전체에 `preprocess_image` 함수를 적용하고, 학습 효율을 높이기 위해 데이터를 섞고(`shuffle`), 배치(`batch`)로 묶어줍니다.
- **데이터셋 분리**: `oxford_flowers102` 데이터셋의 모든 이미지를 합친 후, 80%를 학습에, 20%를 모델 성능 검증에 사용하도록 나눕니다.

---

### **4. KID(Kernel Inception Distance) 클래스 (42-77행)**

생성된 이미지의 품질을 정량적으로 평가하기 위한 지표를 정의하는 클래스입니다.

```python
class KID(keras.metrics.Metric):
    def __init__(self, name, **kwargs):
        # ... InceptionV3 모델을 특징 추출기로 로드 ...
        self.encoder = keras.Sequential([...])

    def update_state(self, real_images, generated_images, sample_weight=None):
        # ... 실제 이미지와 생성된 이미지에서 특징을 추출 ...
        # ... 두 특징 분포 간의 거리를 계산하여 KID 점수 업데이트 ...

    def result(self):
        # ... 평균 KID 점수 반환 ...
```

- `KID` 클래스는 케라스의 `Metric`을 상속받아 만들어집니다.
- 미리 학습된 InceptionV3 모델을 이용해 실제 이미지와 생성된 이미지로부터 특징(feature)을 추출합니다.
- 두 특징 벡터 집합 간의 거리(Maximum Mean Discrepancy)를 계산하여 이미지의 품질을 평가합니다. 점수가 낮을수록 실제 이미지와 유사하다는 의미입니다.

---

### **5. U-Net 네트워크 아키텍처 (79-141행)**

DDIM의 핵심인 U-Net 모델의 구조를 정의합니다. U-Net은 노이즈 섞인 이미지를 입력받아 원본 이미지를 예측(또는 노이즈를 예측)하는 역할을 합니다.

```python
def sinusoidal_embedding(x):
    # ... 확산 시간(noise level)을 sin, cos 함수를 이용해 벡터로 변환 ...

def ResidualBlock(width):
    # ... 잔차 연결(skip connection)을 포함하는 기본 블록 ...

def DownBlock(width, block_depth):
    # ... U-Net의 인코더 부분(해상도를 낮춤) ...

def UpBlock(width, block_depth):
    # ... U-Net의 디코더 부분(해상도를 높임) ...

def get_network(image_size, widths, block_depth):
    # ... 위의 블록들을 조립하여 전체 U-Net 모델을 생성 ...
    return keras.Model(...)
```

- **`sinusoidal_embedding`**: 확산 모델은 현재 노이즈가 얼마나 섞였는지(diffusion time) 알아야 합니다. 이 함수는 그 시간 정보를 신경망이 이해하기 쉬운 벡터 형태로 변환해줍니다.
- **`ResidualBlock`, `DownBlock`, `UpBlock`**: U-Net을 구성하는 기본 빌딩 블록입니다. `DownBlock`에서 해상도를 줄여가며 이미지의 특징을 압축하고, `UpBlock`에서 다시 해상도를 높여가며 원본 이미지를 복원합니다. 이 과정에서 `DownBlock`의 출력을 `UpBlock`에 직접 연결하는 "skip connection"이 U-Net의 핵심입니다.
- **`get_network`**: 이 함수는 위의 블록들을 순서대로 조립하여 최종 U-Net 모델을 완성합니다.

---

### **6. `DiffusionModel` 클래스 (143-258행)**

모델의 학습, 평가, 이미지 생성 등 확산 모델의 모든 과정을 총괄하는 메인 클래스입니다.

```python
class DiffusionModel(keras.Model):
    def __init__(self, image_size, widths, block_depth):
        # ... U-Net(network)과 EMA U-Net(ema_network) 생성 ...

    def diffusion_schedule(self, diffusion_times):
        # ... 코사인 스케줄에 따라 노이즈와 원본 이미지의 비율을 계산 ...
        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        # ... U-Net을 이용해 노이즈 섞인 이미지에서 원본 이미지와 노이즈를 분리 ...
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        # ... 순수 노이즈에서 시작해 점진적으로 노이즈를 제거하며 이미지 생성 (샘플링) ...
        return pred_images

    def generate(self, num_images, diffusion_steps):
        # ... 전체 이미지 생성 과정을 편리하게 호출하는 함수 ...

    def train_step(self, images):
        # 1. 원본 이미지에 랜덤 노이즈를 섞음
        # 2. U-Net이 노이즈를 예측하도록 학습
        # 3. 손실(loss)을 계산하고 역전파를 통해 모델 가중치 업데이트
        # ...

    def test_step(self, images):
        # ... 학습 중인 모델의 성능을 검증 데이터셋으로 평가 (KID 점수 계산) ...

    def plot_images(self, ...):
        # ... 생성된 이미지를 화면에 그려서 보여줌 ...
```

- **`__init__`**: 학습에 사용될 `network`와, 안정적인 이미지 생성을 위한 `ema_network`(가중치를 지수이동평균으로 업데이트)를 만듭니다.
- **`diffusion_schedule`**: 확산 과정을 정의합니다. 특정 시간 `t`에서 이미지에 노이즈를 얼마나 섞을지 결정합니다.
- **`denoise`**: U-Net을 사용하여 노이즈 섞인 이미지(`noisy_images`)로부터 원본 이미지(`pred_images`)와 노이즈(`pred_noises`)를 예측합니다.
- **`reverse_diffusion`**: **이미지 생성(샘플링)** 과정입니다. 순수 노이즈에서 시작하여, `denoise`를 반복적으로 호출하며 점차 노이즈를 제거해 최종 이미지를 만듭니다.
- **`train_step`**: **학습**의 핵심입니다. 원본 이미지에 `diffusion_schedule`에 따라 노이즈를 섞고, `denoise`가 이 노이즈를 잘 예측하도록 U-Net을 훈련시킵니다.
- **`test_step`**: 학습 중간중간 모델의 성능을 평가합니다. `generate` 함수로 이미지를 생성하고 `KID` 점수를 계산합니다.
- **`plot_images`**: 에폭이 끝날 때마다 이미지를 생성하여 보여줌으로써 학습 진행 상황을 시각적으로 확인할 수 있게 합니다.

---

### **7. 모델 학습 및 추론 (260-271행)**

스크립트의 마지막 부분으로, 위에서 정의한 모든 것을 실행하여 실제 모델을 학습시키고 결과를 확인합니다.

```python
# 1. 모델 생성
model = DiffusionModel(image_size, widths, block_depth)

# 2. 체크포인트 설정 (최적의 모델 가중치를 저장하기 위함)
checkpoint_callback = keras.callbacks.ModelCheckpoint(...)

# 3. 데이터 정규화 값 계산
model.normalizer.adapt(train_dataset)

# 4. 모델 컴파일 (옵티마이저와 손실 함수 설정)
model.compile(...)

# 5. 모델 학습 시작
model.fit(train_dataset, epochs=num_epochs, ...)

# 6. 학습 완료 후, 저장된 최적의 가중치를 불러와 최종 이미지 생성
model.load_weights(checkpoint_path)
model.plot_images()
```

- **모델 생성**: `DiffusionModel` 클래스의 인스턴스를 만듭니다.
- **체크포인트**: 학습 중 `val_kid` 점수가 가장 낮을 때(가장 성능이 좋을 때)의 모델 가중치를 파일로 저장하는 규칙을 설정합니다.
- **정규화**: 학습 데이터 전체의 평균과 분산을 계산하여 `Normalization` 레이어에 저장합니다.
- **컴파일**: 모델이 어떻게 학습할지(어떤 옵티마이저와 손실 함수를 쓸지)를 설정합니다.
- **학습**: `model.fit()` 함수를 호출하여 정의된 에폭(epoch)만큼 학습을 시작합니다.
- **추론**: 학습이 끝나면, 저장된 가장 좋은 모델의 가중치를 불러와 최종 결과 이미지를 생성하고 출력합니다.
