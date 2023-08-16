# AIFFEL Campus Online 5th Code Peer Review Templete
- 코더 : 이수봉
- 리뷰어 : 조준규


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [ ] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  > 코드가 resnet34까지는 완성이 되었지만 resnet50이 완성되지 못해서 문제가 온전히 해결되지는 않았다.
- [ ] 주석을 보고 작성자의 코드가 이해되었나요?
  > 코드는 잘 이해되었지만 주석이 많이 작성되어 있지는 않았다.
- [X] 코드가 에러를 유발할 가능성이 없나요?
  > 팀 코드에서 plain net을 build 하는 과정에 오류가 있긴 했지만,
  > 전체적으로 block 단위로 네트워크를 구현하여 다른 큰 오류가 나타나지 않았다.
- [X] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > resnet의 skip connection 등을 잘 해결한 것 등으로 코드를 제대로 이해했다고 생각했다.
- [X] 코드가 간결한가요?
  > 하나의 함수를 만들어서 plain과 resnet을 쉽게 build 할 수 있도록 하였다.

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.
```python
# 코드 이해도 예시
# resnet_block이라는 함수를 잘 구현하여 알맞게 사용함.
def build_resnet_34(plain=False):
    
    inputs = layers.Input(shape=(224, 224, 3))
    x = layers.Rescaling(1./255)(inputs)
    
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    ...    
    
    x = resnet_block(x, filters=512, short_cut=True, plain=plain)
    x = resnet_block(x, filters=512, plain=plain)
    x = resnet_block(x, filters=512, plain=plain)
    
    
    x = layers.AveragePooling2D()(x)
    x = layers.Flatten()(x)
    out = layers.Dense(1, activation='sigmoid')(x) 
    
    model = keras.Model(inputs=inputs, outputs=out)  # 모델 생성
    
    return model
```
```python
# 코드 간결성 예시
# build_resnet이라는 함수를 통해 plain과 resnet을 쉽게 build함.
resnet_34 = build_resnet_34(plain=False)
resnet_34_plain = build_resnet_34(plain=True)
```

# 참고 링크 및 코드 개선
```python
# build 함수에 50 layers에 대한 모델 생성 부분도 있으면 좋겠습니다.
# 팀 코드에서 문제 생기는 건 average pooling할 때 stride로 인해 음수 shape가 생기는 것 같은데
# 정확히 어디서 문제가 생기는지는 모르겠네요 ㅠㅠ
# 한 번 수정해보시는 것도 좋을 것 같습니다. 👍
```
