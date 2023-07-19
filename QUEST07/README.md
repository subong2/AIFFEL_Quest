# AIFFEL Campus Online 5th Code Peer Review Templete
- 코더 : 이수봉
- 리뷰어 : 황규빈


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [ ] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  > 네
- [ ] 주석을 보고 작성자의 코드가 이해되었나요?
  > 친절하게 잘 적혀 있어서, 몰랐던 부분도 이해가 되었습니다.
  > 추가적으로 이미지를 넣어서 이해가 더 잘 되었습니다.
  > 
  > ![image](https://github.com/HGyubin/subong_AIFFEL_Quest/assets/137243622/b23a5d2f-e457-47f3-b040-176491854c75)

- [ ] 코드가 에러를 유발할 가능성이 없나요?
  > 없을걸로 보여집니다.
- [ ] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 네, 전체적으로 로직에 대한 이해가 쉽게 작성하여서 코드 읽기가 수월했습니다.
- [ ] 코드가 간결한가요?
  ```python
    class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.block1 = layers.Concatenate()
        self.block2 = DiscBlock(n_filters=64, stride=2, custom_pad=False, use_bn=False, act=True)
        self.block3 = DiscBlock(n_filters=128, stride=2, custom_pad=False, use_bn=True, act=True)
        self.block4 = DiscBlock(n_filters=256, stride=2, custom_pad=False, use_bn=True, act=True)
        self.block5 = DiscBlock(n_filters=512, stride=1, custom_pad=True, use_bn=True, act=True)
        self.block6 = DiscBlock(n_filters=1, stride=1, custom_pad=True, use_bn=False, act=False)
        self.sigmoid = layers.Activation("sigmoid")

    위 코드를 아래처럼 간결하게 작성할 수 있을거 같습니다.
    하지만, 위 코드가 더 가독성이 더 좋다고 느껴집니다.
        
        filters = [64,128,256,512,1]
        self.blocks = [layers.Concatenate()]
       
        for i, f in enumerate(filters):
            self.blocks.append(DiscBlock(f, stride=2 if i<3 else 1, custom_pad=i>=2, use_bn=i not in [0,4], act=i<4))
        self.sigmoid = layers.Activation('sigmoid')
        
      

# 참고 링크 및 코드 개선

  ```python
    # 학습을 완료하고 시각적으로 학습이 잘 되었는지 확인하는 코드를 넣으면 더 좋을거 같습니다!
  
    # epoch loop 과정
    # ..
    history = {'gen_loss':[], 'l1_loss':[], 'disc_loss':[]}
    history['gen_loss'].append(g_loss)
    history['l1_loss'].append(l1_loss)
    history['disc_loss'].append(d_loss)
    #...
  
    plt.figure(figsize=(16,10))
    
    plt.subplot(311)
    plt.plot(history['gen_loss'])
    plt.title('Generator Loss')
    
    plt.subplot(312)
    plt.plot(history['l1_loss'])
    plt.title('L1 Loss')
    
    plt.subplot(313)
    plt.plot(history['disc_loss'])
    plt.title('Discriminator Loss')
    
    plt.show()
