**Vision**

1) Classification

2) Object Detection

3) Image Segmentation

4) Visual relationship

![image-20210706123540752](C:\Users\chaeh\AppData\Roaming\Typora\typora-user-images\image-20210706123540752.png)

- **Classification :** Single object에 대해서 object의 클래스를 분류하는 문제이다.
- **Classification + Localization :** Single object에 대해서 object의 위치를 bounding box로 찾고 (Localization) +클래스를 분류하는 문제이다. (Classification)
- **Object Detection :** Multiple objects에서 각각의 object에 대해 Classification + Localization을 수행하는 것이다.
- **Instance Segmentation :** Object Detection과 유사하지만, 다른점은 object의 위치를
  bounding box가 아닌 실제 edge로 찾는 것이다.

![image-20210708095134127](C:\Users\chaeh\AppData\Roaming\Typora\typora-user-images\image-20210708095134127.png)

**Object detection에는 1-stage detector, 2-stage detector가 있다.**

<1-stage detector>

![image-20210706124402281](C:\Users\chaeh\AppData\Roaming\Typora\typora-user-images\image-20210706124402281.png)

- 전체 image에 대해서 convolution network로 classification, box regression(localization)을 수행

- 여러 noise 즉, 여러 object가 섞여있는 이미지에서 정확도는 떨어짐
- 간단하고 쉬운만큼 속도가 빠름



<2-stage detector>

![image-20210706124554217](C:\Users\chaeh\AppData\Roaming\Typora\typora-user-images\image-20210706124554217.png)

- Selective search, Region proposal network와 같은 알고리즘을 및 네트워크를 통해 object가 있을만한 영역을 추출 -> RoI(Region of Interest)
- 각 영역들을 convolution network를 통해 classification, box regression(localization)을 수행



## R-CNN 

Image classification을 수행하는 CNN +  localization을 위한 regional proposal 알고리즘 

**Object detection을 위해 고안 **

**R-CNN 프로세스**

1. Image를 입력받는다.

2. Selective search알고리즘에 의해 regional proposal output 약 2000개를 추출한다.

   추출한 regional proposal output을 모두 동일 input size로 만들어주기 위해 warp해준다.

3. 2000개의 warped image를 각각 CNN 모델에 넣는다.

4. 각각의 Convolution 결과에 대해 classification을 진행하여 결과를 얻는다.



**결국, 3가지 모듈로 진행됨 **

- **1. Region Proposal :**

  "Object가 있을법한 영역"을 찾는 모듈 (기존의 Sliding window방식의 비효율성 극복) 

  Selective search 알고리즘을 사용하여, 2000개의 region proposal이 생성되면 이들을 모두 CNN에 넣기 전에 같은 사이즈로 warp

  < Selective search>

  ![image-20210706131146260](C:\Users\chaeh\AppData\Roaming\Typora\typora-user-images\image-20210706131146260.png)    	1. 색상, 질감, 영역크기 등.. 을 이용해 non-object-based segmentation을 수행한다.

  이 작업을 통해 좌측 제일 하단 그림과 같이 많은 small segmented areas들을 얻을 수 있다.

  2. Bottom-up 방식으로 small segmented areas들을 합쳐서 더 큰 segmented areas들을 만든다.

  3. (2)작업을 반복하여 최종적으로 2000개의 region proposal을 생성한다.

     

- **2. CNN** : 각각의 영역으로부터 고정된 크기의 Feature Vector를 추출(224x224)
   AlexNet의 구조를 차용

  

- **3. SVM** : Classification을 위한 선형 지도학습 모델 

  CNN모델로부터 feature가 추출되면 Linear SVM을 통해 classification을 진행

  (SVM은 CNN으로부터 추출된 각각의 feature vector들의 점수를 class별로 매기고, 객체인지 아닌지, 객체라면 어떤 객체인지 등을 판별하는 역할을 하는 Classifier)

  +Selective search로 만든 bounding box는 정확하지 않기 때문에 물체를 정확히 감싸도록 조정해주는 bounding box regression(선형회귀 모델)이 존재



**단점**

1. 여기서 selective search로 2000개의 region proposal을 뽑고 각 영역마다 CNN을 수행하기 때문에 CNN연산 \* 2000 만큼의 시간이 걸려 수행시간이 매우 느리다. 

2. CNN, SVM, Bounding Box Regression 총 세가지의 모델이 multi-stage pipelines으로 한 번에 학습되지 않는다. 따라서 SVM, bounding box regression에서 학습한 결과가 CNN을 업데이트 시키지 못한다.

    (추후 Fast R-CNN  나옴)

   

## 이후 R-CNN

- Fast R-CNN: RoI Pooling을 하나 추가함으로써 CNN후에 region proposal 연산 - 2000xCNN연산 → 1번의 CNN연산으로 가능하게 하였고, 변경된 feature vector가 결국 기존의 region proposal을 projection시킨 후 연산한 것이므로 해당 output으로 classification과 bounding box regression도 학습을 가능하게 함

- ![image-20210706133306476](C:\Users\chaeh\AppData\Roaming\Typora\typora-user-images\image-20210706133306476.png)

   

  그러나 RoI를 생성하는 Selective search 알고리즘은 CNN외부 에서 진행되므로 여전히 속도가 느림

- Faster R-CNN :Region Proposal도 Selective search 쓰지말고 CNN - (classification | bounding box regression)  이 네트워크 안에서 해결하는 것이 base

  RPN ( selective search의 역할을 온전히 대체)+ Fast R-CNN (Fast R-CNN구조에서 conv feature map과 RoI Pooling사이에 RoI를 생성하는Region Proposal Network가 추가)

  - RPN
    - 다양한 사이즈의 이미지를 입력 값으로 object score과 obeject proposal을 출력한다.
    - Fast r-cnn과 합성공 신경망을 공유한다.
    - feature map의 마지막 conv 층을 작은 네트워크가 sliding하여 저차원으로 매핑한다.
    - Regression과 classification을 수행한다.

## Mask R-CNN

**Image Segmentation을 위해 고안**

Fast RCNN과 달라진 점

1) Fast R-CNN의 classification, localization(bounding box regression) branch에 새롭게 mask branch가 추가됐다.

2) RPN 전에 FPN(feature pyramid network)가 추가됐다.

3) Image segmentation의 masking을 위해 RoI align이 RoI pooling을 대신하게 됐다.

![image-20210706135015525](C:\Users\chaeh\AppData\Roaming\Typora\typora-user-images\image-20210706135015525.png)

**<Mask R-CNN 과정>**



###  compute_backbone_shapes로 backbone 네트워크 스테이지의 width, height를 계산

identity_block()과 conv_block()으로 이루어진 Resnet network

1. 800~1024 사이즈로 이미지를 resize해준다. (using bilinear interpolation)(ResNet 네트워크에서는 이미지 input size가 800~1024일때 성능이 좋다고 알려져있다) 

   

2. Backbone network의 인풋으로 들어가기 위해 1024 x 1024의 인풋사이즈로 맞춰준다. (using padding)

3. ResNet-101을 통해 각 layer(stage)에서 feature map (C1, C2, C3, C4, C5)를 생성한다.

4. FPN을 통해 이전에 생성된 feature map에서 P2, P3, P4, P5, P6 feature map을 생성한다.

FPN에서는 마지막 layer의 feature map에서 점점 이전의 중간 feature map들을 더하면서 이전 정보까지 유지할 수 있도록 한다. 

5. 최종 생성된 feature map에 각각 RPN을 적용하여 classification, bounding box regression output값을 도출한다.

6. output으로 얻은 bounding box regression값을 원래 이미지로 projection시켜서 anchor box를 생성한다.

7. Non-max-suppression을 통해 생성된 anchor box 중 score가 가장 높은 anchor box를 제외하고 모두 삭제한다.

8. 각각 크기가 서로다른 anchor box들을 RoI align을 통해 size를 맞춰준다.

9. Fast R-CNN에서의 classification, bounding box regression branch와 더불어 mask branch에 anchor box값을 통과시킨다.

