# CNN, RNN, LSTM 정리

---

# 1. CNN (Convolutional Neural Network)

## 1-1. CNN이란?

CNN(합성곱 신경망)은 이미지 데이터를 처리하기 위해 사용하는 딥러닝 모델이다.  
기존의 Dense Layer(완전연결층) 방식은 이미지의 공간적인 특징을 제대로 학습하지 못한다는 문제가 있었다.

예를 들어 이미지 데이터를 단순히 1차원으로 평탄화(flatten)하게 되면:
- 픽셀 간의 위치 정보 손실
- 공간적 특징(spatial feature) 손실
- 이미지 구조 파악 어려움

즉, 이미지의 중요한 특징인:
- 모서리(edge)
- 선(line)
- 패턴(pattern)

등을 효과적으로 추출하기 어렵다.

---

# 1-2. Convolution(합성곱) 연산

이러한 문제를 해결하기 위해 CNN에서는 Convolution 연산을 사용한다.

합성곱은:
- 원본 이미지에 필터(Filter)를 적용하여
- 새로운 특징 맵(feature map)을 생성하는 과정이다.

즉:

\[
\text{입력 이미지} * \text{필터} = \text{특징 맵}
\]

필터는 이미지 위를 이동하면서:
- 수직선
- 수평선
- 모서리

같은 특징들을 추출한다.

---

## 1-3. 필터(Filter)

필터는 작은 행렬 형태이다.

예:

\[
3 \times 3,\ 5 \times 5
\]

필터 크기가 커질수록:
- 더 넓은 특징 추출 가능
- 가중치(weight)와 bias 증가
- 계산량 증가

Dense Layer는 하나의 큰 weight 연결 구조이지만,  
CNN은 여러 개의 필터를 통해 다수의 weight/bias를 학습한다.

또한 필터 값은 처음에는 랜덤값으로 시작하지만,
학습 과정에서 최적화된다.

즉:
- Forward propagation
- Loss 계산
- Backpropagation

과정을 통해 필터가 점점 이미지 특징에 맞게 수정된다.

---

# 1-4. Stride와 Padding

## (1) Stride

Stride는 필터가 이동하는 거리이다.

예:
- Stride = 1 → 한 칸씩 이동
- Stride = 2 → 두 칸씩 이동

Stride가 커질수록:
- 출력 크기 감소
- 계산량 감소

---

## (2) Padding

Padding은 이미지 경계 부분을 처리하기 위해 주변에 값을 추가하는 것이다.

목적:
- 가장자리 정보 보존
- 출력 크기 조절

종류:
- Valid Padding
- Same Padding

---

# 1-5. Pooling Layer

Convolution 연산을 수행하면 특징 정보의 양이 매우 증가한다.

이를 해결하기 위해 Pooling을 사용한다.

Pooling은 특징 맵의 크기를 줄여:
- 계산량 감소
- 과적합 완화
- 중요한 특징 유지

를 수행한다.

---

## Pooling 종류

### (1) Max Pooling

영역 내 최대값 선택

특징:
- 가장 강한 특징 유지

---

### (2) Average Pooling

영역 내 평균값 계산

특징:
- 전체적인 특징 유지

---

# 1-6. Pooling의 역할

Pooling은 크게 두 가지 기능을 가진다.

## (1) 정보 요약

중요한 특징만 남기고 정보 압축

---

## (2) 이동 불변성(Translation Invariance)

이미지가 약간 이동해도 비슷한 특징을 유지

즉:
- 위치가 조금 변해도 같은 객체로 인식 가능

---

# 1-7. Activation Function

CNN에서는 보통 ReLU 함수를 사용한다.

\[
ReLU(x)=\max(0,x)
\]

역할:
- 비선형성 추가
- 학습 속도 향상
- Gradient Vanishing 완화

---

# 1-8. Fully Connected Layer

CNN 마지막 부분에서는 Fully Connected Layer(Dense Layer)를 사용하여 최종 분류를 수행한다.

예:
- 고양이
- 강아지
- 자동차

등의 클래스를 분류한다.

---

# 1-9. CNN 장점과 단점

## 장점
- 이미지 특징 자동 추출
- 공간적 특징 학습 가능
- 높은 이미지 인식 성능

---

## 단점
- 계산량 많음
- GPU 자원 필요
- 데이터 많이 필요

---

# 2. RNN (Recurrent Neural Network)

## 2-1. RNN이란?

RNN(순환 신경망)은 순서(sequence)가 중요한 데이터를 처리하는 모델이다.

대표 예시:
- 자연어 처리(NLP)
- 음성 인식
- 번역
- 시계열 데이터 분석

---

## 2-2. RNN의 핵심 특징

RNN은 이전 상태(hidden state)를 기억한다.

즉:

\[
h_t=f(x_t,h_{t-1})
\]

- \(x_t\): 현재 입력
- \(h_{t-1}\): 이전 상태
- \(h_t\): 현재 상태

이전 정보를 다음 단계에 전달하기 때문에:
- 문맥(context)
- 시간 흐름

을 학습할 수 있다.

---

## 2-3. RNN 장점

- 순차 데이터 처리 가능
- 문맥 정보 기억 가능
- 시계열 분석 가능

---

## 2-4. Gradient Vanishing 문제

RNN은 역전파 과정에서 gradient를 반복적으로 곱하게 된다.

이때:
- gradient가 매우 작아지거나
- 매우 커질 수 있다.

특히 gradient가 0에 가까워지는 현상을:

\[
\text{Gradient Vanishing}
\]

이라고 한다.

문제점:
- 오래된 정보 학습 어려움
- 긴 문장 기억 어려움

---

# 3. LSTM (Long Short-Term Memory)

## 3-1. LSTM이란?

LSTM은 RNN의 Gradient Vanishing 문제를 해결하기 위해 등장한 모델이다.

긴 시퀀스 데이터도 효과적으로 기억할 수 있다.

---

# 3-2. LSTM 핵심 구조

LSTM은:
- Cell State
- Gate 구조

를 사용한다.

---

## (1) Forget Gate

불필요한 정보 제거

---

## (2) Input Gate

새로운 정보 저장

---

## (3) Output Gate

출력할 정보 결정

---

# 3-3. LSTM 장점

- 장기 의존성 문제 해결
- 긴 문장 처리 가능
- RNN보다 성능 우수

---

# 3-4. LSTM 단점

- 구조 복잡
- 연산량 증가
- 학습 속도 느림

---

# 4. CNN vs RNN vs LSTM 비교

| 모델 | 데이터 종류 | 특징 | 장점 | 단점 |
|---|---|---|---|---|
| CNN | 이미지 | 공간 특징 추출 | 이미지 처리 우수 | 계산량 큼 |
| RNN | 시퀀스 | 이전 정보 기억 | 순차 데이터 처리 | Gradient Vanishing |
| LSTM | 시퀀스 | Gate 구조 | 장기 기억 가능 | 구조 복잡 |
