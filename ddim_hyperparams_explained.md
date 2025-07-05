# DDIM 홀더 예제 하이퍼파래터 해석 (Keras)

## 데이터 관련

```python
dataset_name = "oxford_flowers102"
```
- Keras Datasets에서 제공하는 Oxford Flowers 102 (꽃 102종) 데이터셋 사용

```python
dataset_repetitions = 5
```
- 데이터셋을 학습에 더 많이 사용하기 위해 반복 횟수를 지정합니다.
- 데이터셋이 상대적으로 작기 때문에, 한 epoch 내에서 여러 번 반복하여 효과적으로 더 많이 학습시키는 역할을 합니다.
```python
num_epochs = 1
```
- epoch 수
- 좋은 성능을 위해서는 50 이상 권장

```python
image_size = 64
```
- 입력 이미지 크기
- 64x64 크기로 재조정

## KID (Kernel Inception Distance) 관련

```python
kid_image_size = 75
```
- KID를 계산할 때 사용하는 InceptionNet 입력 이미지 크기입니다.
- 일반적으로 Inception은 75x75 이상 이미지를 사용하므로 75로 지정.

```python
kid_diffusion_steps = 5
```
- KID 평가를 위해 **reverse diffusion(샘플링)**을 할 때 몇 step만 사용할지 지정합니다.
- 낮을수록 빠르지만 정확도 낮고, 높을수록 평가 품질 높아짐.

```python
plot_diffusion_steps = 20
```
- 시각화를 위해 샘플링 과정에서 몇 개의 중간 이미지를 그릴지 결정합니다.

## 사크 값

```python
min_signal_rate = 0.02
max_signal_rate = 0.95
```
- Noise Schedule 의 최소/최대 signal rate
- signal rate = 1 - noise rate
- 낮은 rate일수록 더 큰 노이즈

## 네트워크 구조

```python
embedding_dims = 32
```
- timestep t의 sinusoidal positional embedding 차수

```python
embedding_max_frequency = 1000.0
```
- positional embedding 에서 사용하는 최대 주화수

```python
widths = [32, 64, 96, 128]
```
- UNet 계층 매 단계별 channel 값

```python
block_depth = 2
```
- UNet block 단계 내 계층 수

## 최적화

```python
batch_size = 64
```
- 하나의 batch 다른 이미지 수

```python
ema = 0.999
```
- 지수 이동 평균(EMA) 계수

```python
learning_rate = 1e-3
```
- 학습 율

```python
weight_decay = 1e-4
```
- L2 regularization 값

