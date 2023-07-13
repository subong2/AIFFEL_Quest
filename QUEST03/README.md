# AIFFEL Campus Online 5th Code Peer Review Templete
- 코더 : 이수봉
- 리뷰어 : 김범준


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [V] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  > 여러 사진들을 시험해보는 코드들을 확인할 수 있었는데, 다소 어둡거나 얼굴이 정면을 바라보고 있지 않을 때 모델이 얼굴을 인식하지 못하는 결과를 볼 수 있었습니다. 다만, 이 부분을 모델의 문제라 해석하였기 때문에 코드가 정상적으로 동작하고 문제를 해결했는지에 대한 답변으로 "그렇다"를 체크하게 되었습니다.
- [V] 주석을 보고 작성자의 코드가 이해되었나요?
  > 주석을 통해 각각의 코드들이 어떤 작동을 하는지 이해할 수 있었습니다.
- [V] 코드가 에러를 유발할 가능성이 없나요?
  > if 문의 경우, else를 통해 다른 변수가 생겼을 때를 대비하셨던 부분을 보고 에러가 발생할 가능성이 적다고 판단하였습니다.
- [V] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 피어 리뷰 시간 동안 각각의 코드에 대해 이해하고 있는 모습을 보여주셨습니다.
- [V] 코드가 간결한가요?
  > self.status를 이례적인 경우에 대비해 다시 한번 써 주셨던 부분이 있지만, 그 부분을 제외하면 전반적으로 간결하게 코드가 구성된 것 같습니다.

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.
```python
# 사칙 연산 계산기
class calculator:
    # 예) init의 역할과 각 매서드의 의미를 서술
    def __init__(self, first, second):
        self.first = first
        self.second = second
    
    # 예) 덧셈과 연산 작동 방식에 대한 서술
    def add(self):
        result = self.first + self.second
        return result

a = float(input('첫번째 값을 입력하세요.')) 
b = float(input('두번째 값을 입력하세요.')) 
c = calculator(a, b)
print('덧셈', c.add()) 
```

# 참고 링크 및 코드 개선
```python
# 편의상 코드 내에 적혀있던 주석들은 제외하였습니다.
# 코드의 길이를 고려해 pred_face 함수의 if문 부분을 일부 생략했습니다.
# 아래 코드 이전에 self.status가 0으로 이미 할당되어 있었기 때문에 이후 코드에서 추가로 self.status = 0을 적어두지 않아도 괜찮지 않을까 생각했습니다.
# 다만, 이례적인 경우를 대비해 언제나 확실하게 해 두는 습관도 중요하다 생각해 위의 체크박스에는 따로 이 부분을 감안하지 않았습니다. 

class sticker_photo:
  def __init__(self , img):
    self.detector_hog = dlib.get_frontal_face_detector()
    self.landmark_predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    img = cv2.imread(img)
    self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    self.status = 0

 def pred_face(self):
    cp_img = copy.deepcopy(self.img)
    self.list_landmarks = []
    self.dlib_rects = self.detector_hog(cp_img, 1)

    if self.dlib_rects :
      self.status = 1 

      ...............(if 문 content)................

    else :
      print('검출 된 얼굴이 없습니다. 다른 사진으로 시도해주세요')
      self.status = 0

    plt.imshow(cp_img)
    plt.show()
```
