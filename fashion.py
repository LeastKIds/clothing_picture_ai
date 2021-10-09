import tensorflow as tf
import matplotlib.pyplot as plt

(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

# # 데이터를 사진화 시키기
# plt.imshow(trainX[0])
# # 흑백화
# plt.gray()
# # 컬러 바
# plt.colorbar()
# plt.show()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(28, 28), activation='relu'), # input_shape : 들어갈 데이터 형식
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Flatten(),  # 여러차원의 배열을 1차원 배열로 바꿔줌
    tf.keras.layers.Dense(10, activation='softmax'),  # 마지막이 10개면 확률로
#     sigmoid :
#        결과를 0~1로 압축
#        binary 예측 문제 ex) 대학원 붙는다 / 안 붙는다
#        마지막 노드는 1개
#     softmax :
#           결과를 0~1로 압축
#           카테고리 예측 문제에 사용(10개의 정답 중 하나를 선택해 그게 정답일 확률)
#           노드의 개수는 카테고리 개수
])

model.summary() # 데이터를 돌리기 전에 요약본을 미리 봄. input_shape 가 설정되어 있어야함

# sparse_categorical_crossentropy : trainX 가 0,1,2,3... 등 정수로 되어있을 때
# categorical_crossentropy : trainY가 원핫인코딩 되어있을 때
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainY, epochs=5)
