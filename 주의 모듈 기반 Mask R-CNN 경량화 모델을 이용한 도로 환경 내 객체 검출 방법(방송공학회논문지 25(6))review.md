## 주의 모듈 기반 Mask R-CNN 경량화 모델을 이용한 도로 환경 내 객체 검출 방법

### 1. 서론

CNN을 이용한 다양한 객체인식 및 검출방법들 중에서 Mask R-CNN은 객체별 분할(Instance Segmentation) 결과까지 제공함으로써 자율주행을 위한 도로 환경 객체 검출에 널리 사용

Mask R-CNN은 영역 제안 신경망(RPN)을 통해 영상 내 객체가 존재할 가능성이 있는 영역 후보를 생성하고 이에 대해 Latent Feature(잠재 특징)을 추출-> 추출된 잠재특징은 classification, Bounding Box Regression 및 Segmentation에 공통적으로 사용 

이 2단계 신경망 구조는 복잡한 영상에서도 객체 별로 독립적으로 분할 가능하기 때문에 사각 영역 기반 검출기 대비 차종, 도로표지판, 보행자 인식에 유용하게 적용 가능(YOLO)

그러나 많은 수의 신경망 매개변수 및 GPU 메모리 요구에 따라 실제 자율주행을 위한 자동차에 서버와의 통신없이 탑재되기 어려움

-> 이 한계점을 극복하기 위해 Efficient Net구조를 Backbone신경망으로 적용

+다중 스케일 정보를 효과적으로 사용하여 특징을 추출하기 위해 BiFPN적용

+주의 모듈을 각 작업별 가지 신경망에 삽입함으로써 중요도가 각 작업의 목적에 맞게 재조정 되도록

=매개변수 수를 약 절반으로 감소시킬 수 있으며 성능 또한 효과적으로 유지 할 수 있음



### 2. 제안하는 방법

**1. Mask R-CNN 경량화 모델의 구조**

![image-20210719224502349](C:\Users\chaeh\AppData\Roaming\Typora\typora-user-images\image-20210719224502349.png)

EfficientNet 구조를 기반으로 영상 특징을 다중 스케일에서 추출하는 Backbone으로 사용

: Efficient Net은 모델의 해상도, 깊이, 너비를 동시에 고려하는 복합 스케일링 기법을 사용하여 효과적으로 특징을 압축 

- 해상도: 입력영상의 크기, 깊이: 신경망 계층의 개수, 너비: 필터(채널의 개수)

- 기존 Mask RCNN에서 사용하는 ResNet-101구조는 영상특징이 압축되면서 채널의 개수가 2048개까지 증가하지만 , EfficientNet 구조는 224개 채널 사용

- Backbone 구조의 각 스케일마다 특징맵을 추출하여 5개의 특징맵을 얻고, BiFPN구조를 사용

  ![image-20210719225322747](C:\Users\chaeh\AppData\Roaming\Typora\typora-user-images\image-20210719225322747.png)

- 높은수준에서 낮은수준으로 방향만 연결선이 존재하는 FPN구조와 달리, 상향식 방식의 경로가 추가되어 모든 스케일에서 추출된 특징들을 풍성하게 결합

- BiFPN을 여러번 반복하여 객체의 다양한 변이를 효과적으로 고려

- RPN은 BiFPN을 통하여 추출된 잠재 특징 전체 영역에서 Anchor box를 이용하여 관심영역을 ROI Pooling을 통해 일정한 해상도의 지역적 특징맵을 생성

- 지역적 특징맵은 주의 모듈을 통해 세 개의 가지 신경망을 거쳐 객체인식과 객체 위치 예측 및 영역분할에 사용

  ![image-20210719230002833](C:\Users\chaeh\AppData\Roaming\Typora\typora-user-images\image-20210719230002833.png)



궁금증 : Attention module이 ROI 풀링 바로 다음에 3가지 task수행 전 일어나는 것인지?



**2. 주의 모듈을 이용한 특징 결합 재조정**

기존 Mask R-CNN에서는 객체 인식(Classification), 객체 위치 예측(Bounding Box Regression) 및 영역 분할 (Segmentation)의 세 가지 작업을 위해 동일한 지역적 특징 사용하기 때문에 각 작업별 중요도가 고려되지 않음

-> 각 작업을 위한 가지 신경망마다 주의 모듈(Attention Module)을 삽입함으로써 객체 영역 내 지역적 특징 값이 해당 작업의 목적에 맞게 재조정 될 수 있도록 구성

-> SE block을 사용

SE Block 기반 주의모듈: 입력된 지역적 특징을 전역 평균 풀링(Global Average Pooling)을 통해 채널 단위로 1x1 값이 되도록 특징 정보를 압축, 그후 Fully Connected Layer, ReLU, Sigmoid를 거쳐 중요도를 채널단위로 계산

-> 중요도를 이용하여 작업별 가지 신경망에 입력되기 전 잠재 특징을 재조정



**3. 객체 검출을 위한 손실 함수 설계**

객체인식에 사용되는 손실함수: Cross Entropy

객체 위치 예측에 이용되는 손실함수 : 클래스에 해당하는 실제 객체 상자의 위치정보와 예측된 객체 상자의 위치정보를 비교(L1손실함수)

객체별 분할에 사용되는 손실함수: 이진 교차 엔트로피 함수



### 3. 실험 결과 및 분석

MS COCO 2017 datset 사용(1920x1440 화소)

![image-20210719232436431](C:\Users\chaeh\AppData\Roaming\Typora\typora-user-images\image-20210719232436431.png)

pytorch에서 EfficientNet을 사용, 신경망의 학습 매개변수들은 ILSVRC 데이터셋을 이용하여 미리 학습된 모델의 매개변수 값으로 초기화, 정규화 계층은 Group Normalization 기법을 사용, 배치 크기는 1, 손실 함수 AdamW 사용 파워(Power)와 가속도(Momentum) 값은 각각 0.9와 0.999로 설정, 가중치 감쇠 (Weight Decay) 값은 Backbone 신경망에서는 0.0005로 설정, 학습 속도(Learning Rate)는 10^-4부터 10^-5까지 감소시키며 학습



mAP를 활용하여 효율성을 검증-> 값이 클수록 좋은성능을 의미

![image-20210719233321317](C:\Users\chaeh\AppData\Roaming\Typora\typora-user-images\image-20210719233321317.png)



AP50은 정답이라고 판단되는 IoU가 0.5이상일 때의 mAP를 의미

AP75은 정답이라고 판단되는 IoU가 0.75이상일 때의 mAP를 의미

APmask는 객체영역 분할결과에 대한 평균 mAP를 의미

제안하는 방법에서 사용된 BiFPN 구조는 FPN과 비교하여 1.7M의 매개변수 증가 비용이 있지만 mAP 성능을 1.1만큼 크게 향상시킴

모든 데이터셋 에서 주의 모듈을 적용하였을 때가 그렇지 않을 때보다 적은 매개변수 증가 비용으로 큰 검출 성능 향상을 보임

![image-20210719233610725](C:\Users\chaeh\AppData\Roaming\Typora\typora-user-images\image-20210719233610725.png)



### 4. 결론

효율적인 특징 압축을 위해 

Backbone 신경망 구조를 수정 + BiFPN 으로 대체

주의 모듈을 적용하여 다중 작업 신경망에 입력되는 객체 영역 특징을 각 작업의 목적에 맞게 적응적으로 재조정시켜 효과적인 객체 검출 학습 달성

-> 다양한 실험을 통해 제안하는 방법이 기존 방법 대비 신경망 모델의 매개변수를 대폭 감소시키고 검출 성능 또한 향상시킴