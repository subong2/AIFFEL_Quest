# AIFFEL Campus Online 5th Code Peer Review Templete
- 코더 : 이수봉
- 리뷰어 : 김연수


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [O] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  > 네, 모든 코드가 정상적으로 동작하고 과제에서 요구하는 결과물을 도출했습니다.
- [O] 주석을 보고 작성자의 코드가 이해되었나요?
  > 단계별로 주석이 적혀있어서 전체적인 코드를 이해하는 데에 도움이 되었습니다.
- [O] 코드가 에러를 유발할 가능성이 없나요?
  > 에러 없이 잘 작동하는 것으로 보입니다.
- [O] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 작성한 코드에 대해 설명하는 시간을 가졌는데, 학습 내용에 대한 완전한 이해를 바탕으로 코드를 작성했다는 것을 알 수 있었습니다.
- [O] 코드가 간결한가요?
  > 네, 불필요한 부분 없이 간결합니다.

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.

### 1) 3가지 이상의 모델을 성공적으로 시도(LSTM, CNN, GRU)
```python

# 예시)LSTM 모델
vocab_size = 10000    # 어휘 사전의 크기입니다(10,000개의 단어)
word_vector_dim = 16  # 워드 벡터의 차원 수 (변경 가능한 하이퍼파라미터)

# model 설계 - 딥러닝 모델 코드를 직접 작성해 주세요.
model_lstm = tf.keras.Sequential()
# [[YOUR CODE]]
model_lstm.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=word_vector_dim, mask_zero=True))
model_lstm.add(tf.keras.layers.LSTM(10))   # 가장 널리 쓰이는 RNN인 LSTM 레이어를 사용하였습니다. 이때 LSTM state 벡터의 차원수는 8로 하였습니다. (변경 가능)
model_lstm.add(tf.keras.layers.Dense(8, activation='relu'))
model_lstm.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # 최종 출력은 긍정/부정을 나타내는 1dim 입니다.
model_lstm.summary()

model_lstm.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
              
epochs=20  # 몇 epoch를 훈련하면 좋을지 결과를 보면서 바꾸어 봅시다. 

history_lstm = model_lstm.fit(X_train,
                    y_train,
                    epochs=epochs,
                    batch_size=512,
                    validation_data=(X_val, y_val),
                    verbose=1)

```
### 2) 단계별로 주석이 적혀있어서 전체적인 코드를 이해하는 데에 도움이 되었습니다.
```python
def load_data(train_data, test_data, num_words=10000):
    
    # 1) 데이터 중복 제거
    train_data.drop_duplicates(subset=['document'], inplace=True)
    test_data.drop_duplicates(subset=['document'], inplace=True)
    
    # 2) NaN 결측치 제거
    train_data = train_data.dropna(how = 'any') 
    test_data = test_data.dropna(how = 'any')
    
    # 3) 한국어 토크나이저로 토큰화, 불용어 제거
    
    X_train = []
    for sentence in train_data['document']:
        temp_X = tokenizer.morphs(sentence) # 토큰화
        temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
        X_train.append(temp_X)
        
    X_test = []
    for sentence in test_data['document']:
        temp_X = tokenizer.morphs(sentence) # 토큰화
        temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
        X_test.append(temp_X)
```
# 참고 링크 및 코드 개선
```python
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```
gensim 참고링크  
https://radimrehurek.com/gensim/models/keyedvectors.html 
https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#storing-and-loading-models
