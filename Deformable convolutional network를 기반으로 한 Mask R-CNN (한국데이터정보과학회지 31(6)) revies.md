## Deformable convolutional network를 기반으로 한 Mask R-CNN

### 1. 서론

딥러닝 기술의 발달로 객체 탐색 방법론들이 발전

-> 영역제안 알고리즘(region proposal)을 연결하여 객체 탐색 수행하는 **RCNN** 

-> RCNN의 복잡한 학습단계 대신, 한번의 학습을 위한 CNN을 구축하고 객체 탐색을 수행하는 **Fast RCNN**

-> Fast R-CNN에 영역제안 네트워크(RPN)를 추가하여 GPU에서 계산을 가능하게 한 **Faster RCNN**

-> 이미지 내의 경계박스와 클래스 확률을 단일회귀문제로 묶어 속도를 향상시킨 YOLO를 제안

-> Faster RCNN에서 이진마스크와 관심영역정렬(ROI)을 사용하여 성능을 개선한 **Mask RCNN**



Deformable convolutional network : 객체의 크기에 맞는 필터를 학습시킴 (사람의 시야처럼)



**DC+Mask RCNN을 제안**



### 2. 객체 탐색 기존 연구

**2.1 개념 및 용어**

객체탐색: 이미지 내에서 어떤 물체인지 분류 (classification) + 그 물체가 어디있는지 경계박스를 통해 위치 정보를  나타내는 국소화(localization)

영역제안방법: 입력값에서 '객체가 있을 법한'영역을 빠른 속도로 찾아내는 알고리즘 

-> 임의의 모든 영역을 조사하기 때문에 오래 걸림

-> 완전 탐색 방법의 단점을 보완하여 데이터의 특성에 기반하여 후보가 될 영역을 바운딩 박스로 찾아내는 분할(segmentation) 제안 (색깔, 패턴으로 후보영역 찾음)

-> 완전탐색+분할 = 선택적 탐색(selective search)을 제안 (seed를 설정하고, 그 seed에 대해 완전탐색 방식으로 찾음)



**2.2 R-CNN**

영역제안+CNN=RCNN기법

![image-20210718214004151](C:\Users\chaeh\AppData\Roaming\Typora\typora-user-images\image-20210718214004151.png)



1. input image로 부터 selective search를 이용하여 객체후보 영역 약 2000개정도의 영역제안을 추출
2. 각각의 영역들은 CNN의 입력값으로 들어가기 위해 균일한 크기로 맞추고 affine warping을 수행
3. CNN으로 특징 추출
4. 추출된 특징들은 선형 SVM을 통해 최종적으로 객체종류 인식

-> 객체 종류를 수행한 후 경계박스 회귀분석을 수행하면서 localization error를 줄이는 과정 수행



**2.3 Fast RCNN**

RCNN은 입력값으로 부터 선택적 탐색을 이용하여 객체 후보 영역인 수천개의 영역 제안을 추출 

-> 하지만 추출된 수천개의 영역제안 값들은 모두 CNN의 입력값으로 들어가기 때문에 시간이 오래 걸린다는 단점 

![image-20210718214809674](C:\Users\chaeh\AppData\Roaming\Typora\typora-user-images\image-20210718214809674.png)

-> RCNN에 영역제안마다 합성곱연산을 하는 대신, 입력이미지 한번에 CNN을 적용하고, 객체를 판별하는 관심 영역 풀링을 도입한 **Fast RCNN제안**

pooling : Convolution을 거쳐서 나온 activation maps이 있을 때, 이를 이루는 convolution layer을 resizing하여 새로운 layer를 얻는 것

또한 SVM과 선형회귀모델을 모두 하나의 네트워크에 포함시켜 훈련을 시킴, 

소프트맥스층과 동일하게 CNN뒤에 추가하여 성능 개선





**2.4 Faster RCNN**

객체영역 후보 검출을 위한 별도의 과정을 거치지 않고, CNN+RPN 네트워크를 추가하는 **Faster RCNN 제안**

![image-20210718220238382](C:\Users\chaeh\AppData\Roaming\Typora\typora-user-images\image-20210718220238382.png)

-> 따라서, Faster RCNN은 RPN자체를 학습 , 즉 선택적 탐색(selective search)없이 RPN을 학습하는 구조이고, RPN은 CNN으로부터 나온 feature map을 input, 영역제안을 output으로 하는 네트워크이다.

RPN은 객체후보 영역을 탐색하기 위해 sliding window마다 다양한 크기와 가로세로비율을 갖는 anchor들을 추출 -> anchor는 미리 정의된 reference 경계박스

후보로 k개의 anchor을 미리 정해넣고 k x k 필터로 슬라이딩 윈도우 할  때, 슬라이딩 마다 k개의 경계박스 후보를 생성

 슬라이딩 윈도우마다 객체인식을 위한 2k개의 score와  + 회귀를 위한 4k개의 좌표가 생성되어 => anchor마다 p/n라벨로 넣고

이를 트레이닝 셋으로 분류기와 regressor를 학습( 128 × 128, 256 × 256, 512 × 512 3개의 크기와 2:1, 1:1, 1:2 3개의 비율을 이용하여 9개의 사전 정의된 앵커를 사용)![image-20210718220249215](C:\Users\chaeh\AppData\Roaming\Typora\typora-user-images\image-20210718220249215.png)



**2.5 YOLO**

객체탐색문제를 영상으로부터 경계박스좌표와 객체클래스 확률값을 추정하는 단순한 구조의 회귀문제로 해결한 실시간 객체 탐색기법인 **YOLO 제안**

![image-20210718220456440](C:\Users\chaeh\AppData\Roaming\Typora\typora-user-images\image-20210718220456440.png)

입력영상을 S*S그리드로 나눈후, 각 그리드 셀에서 경계박스와 각 박스에 대해 confidence 값을 추정

각 경계박스는 중심위치의 좌표 (x,y)+너비와 높이(w,h)로 구성

신뢰도(confidence)= 정답박스와 예측박스 간의 겹치는 공간인 IOU

각 그리드 셀은 객체클래스에 확률값(신뢰도값)을 추정하게 되고, 특정 객체를 얼마나 포함하는지 정도에 따라 조건부 확률값을 계산

-> Faster RCNN에서의 객체후보 영역 검출을 CNN네트워크와 통합 



궁금증 :RPN네트워크 뒤에서 이루어 진다는것인지? 



**2.6 Mask RCNN **

Faster RCNN 픽셀이 객체인지 아닌지를 판단하는 이진 마스크를 추가+관심영역 풀링대신 관심영역 정렬을 사용

![image-20210718221521522](C:\Users\chaeh\AppData\Roaming\Typora\typora-user-images\image-20210718221521522.png)

mask RCNN은 RPN으로 부터 얻은 관심영역을 클래스 레이블, 경계박스, 마스크 3개의 브랜치로 나누어 처리

-> 마스크 브랜치 : Faster RCNN이 가지지 못했던 pixel to pixel alignment가능하게 함

-> 관심영역 풀링: 반올림을 사용하지 않고 이중선형보간법을 이용해 feature map의 관심영역을 정확하게 정렬





### 3. 제안하는 방법

필터의 모양을 변형시켜 객체에 맞는 필터를 사용하는 방법  **Deformable convolution network**

![image-20210718222422509](C:\Users\chaeh\AppData\Roaming\Typora\typora-user-images\image-20210718222422509.png)

deformable convolution : 일반적인 합성곱층과는 다른 합성곱층이 하나 더 존재

![image-20210718222610841](C:\Users\chaeh\AppData\Roaming\Typora\typora-user-images\image-20210718222610841.png)

conv: 각 입력의 2d offset을 학습하기 위한것 ,계산은 선형보간법으로 이루어짐

offset은 가중치와 함께 역전파 과정에서 같이 학습됨



Mask R-CNN + DC 모형

![image-20210718223038314](C:\Users\chaeh\AppData\Roaming\Typora\typora-user-images\image-20210718223038314.png)

vgg16같은 모형에서 어느층에 offset에 학습시키는지를 정하는것에 따라 결과값이 달라질 수 있음- > 오프셋을 어느 층에 학습시키고 얼마나 오프셋을 학습시키는 것이 성능을 좋게 하는지 확인

(원래의 CNN에서 Receptive Field 에는 정해진 개수의 픽셀들이 입력 이미지를 커버하기 위해서 존재하는데, 이 픽셀들의 위치를 유동적으로 변경하기 위하여 Offset들을 학습하여 적용)



궁금증: offset이 정확히 무엇인지 .. 기준이 되는  bounding box의 크기에 더해지는 값? 

### 4. 실험 결과 및 결과

**4.1 데이터 설명 및 실험설계**

COCO데이터와 Pascal VOC를 사용하여 실험

Vgg16,Vgg19를 사용하여 각각의 모형마다 층별로 DC 네트워크를 적용시킴, IOU는 각각 0.5, 0.75로 설정, epoch은 각각 500, 600을 사용, 학습률 (learning rate)는 각각 0.01, 0.0001, 가중치 감소 (weight decay)는 각각 0.0001, 0.001

필터값은 3 × 3, RPN 앵 커 stride는 1,  모멘텀 (momentum)은 0.9, optimizer는 Adam을 사용

모형 성능 비교방법인 MAP=1개 객체 당 1개 average precision값을 구하고 여러객체의 ap값을 평균 낸 값



**4.2 실험 결과**

![image-20210718223839685](C:\Users\chaeh\AppData\Roaming\Typora\typora-user-images\image-20210718223839685.png)

![image-20210718223846007](C:\Users\chaeh\AppData\Roaming\Typora\typora-user-images\image-20210718223846007.png)

![image-20210718223859554](C:\Users\chaeh\AppData\Roaming\Typora\typora-user-images\image-20210718223859554.png)

![image-20210718223914394](C:\Users\chaeh\AppData\Roaming\Typora\typora-user-images\image-20210718223914394.png)

CNN 모형을 VGG16보다 VGG19를 사용했을 때 결과가 더 좋음을 알 수 있었고, 이를 통 해 실험을 설계할 때 어떤 모형을 사용할지 선택하는 것도 실험 결과에 중요한 영향을 끼친다는 것을 알 수 있었다. 또한 기존 Mask R-CNN 방법보다 본 연구에서 제안한 DC+Mask R-CNN 방법이 더 좋은 성능을 가지고 있음을 확인하였다.



### 5. 결론

 VGG16보다 VGG19에서 더 좋은 성능을 확인할 수 있었다. 이를 통해 실험설계에 있어서 CNN 모형 선택이 중요 한 역할을 한다는 것을 확인할 수 있었다. 그리고 Pascal VOC나 COCO와 같은 데이터를 이용한 실 험을 통해 본 연구에서 제안한 방법이 기존 방법보다 성능이 향상됨을 보였다. DC+Mask R-CNN 방 법은 오프셋을 학습시키기 때문에 어느 층에 오프셋을 학습하는지에 따라 결과가 달라지게 된다. 따라서 오프셋을 어디에 적용하는지를 찾는 것이 문제가 되고 오프셋을 학습시키기 위한 새로운 매개변수가 생기기 때문에 속도가 느려지는 단점이 있다.

