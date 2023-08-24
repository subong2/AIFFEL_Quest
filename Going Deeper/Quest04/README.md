# AIFFEL Campus Online 5th Code Peer Review Templete
- 코더 : 이수봉
- 리뷰어 : 최연석


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [X] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  > def test_system(func) 으로 결과값을 잘 도출해 내었습니다.
- [X] 주석을 보고 작성자의 코드가 이해되었나요?
  > 주석과 추가 설명을 통해 코드를 잘 설명하였습니다.
  > ```
  > ex)
  Object Detection의 라벨은 class와 box로 이루어지므로 각각을 추론하는 부분이 필요(head)
  Backbone에 해당하는 네트워크와 FPN을 통해 pyramid layer가 추출되고 나면 그 feature들을 바탕으로 class와 box도 예상한다
  class와 box가 모두 맞을 수도, class와 box 중 하나만 맞을 수도, 둘 다 틀릴 수도 있다 class를 예측하는 head와 box를 예측하는 head가 별도로 존재한다는 것이 중요
  그래서 각각의 head를 만들어 줍니다.
  > ```
- [X] 코드가 에러를 유발할 가능성이 없나요?
  > 오류없이 잘 작동 하였습니다.
- [X] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 함수 윗 부분에 필요한 설명을 첨부하였습니다.
  > '''
  > ex)
  입력으로 이미지 경로를 받습니다.
  정지조건에 맞는 경우 "Stop" 아닌 경우 "Go"를 반환합니다.
  조건은 다음과 같습니다.
  
  사람이 한 명 이상 있는 경우
  차량의 크기(width or height)가 300px이상인 경우
  > '''
- [X] 코드가 간결한가요?
  > 함수 및 클래스를 사용하여 간결하게 작성 하였습니다.
  > '''
  > class DecodePredictions, RetinaNetBoxLoss, RetinaNetClassificationLoss, RetinaNetLoss 등..
  > def prepare_image, visualize_detections 등..
  > '''

# 참고 링크 및 코드 개선
```
모델 성능 그래프가 없어 시각화가 아쉬웠습니다.
```
