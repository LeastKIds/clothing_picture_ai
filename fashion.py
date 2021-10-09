import tensorflow as tf
import matplotlib.pyplot as plt

(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

# 데이터를 사진화 시키기
plt.imshow(trainX[0])
# 흑백화
plt.gray()
# 컬러 바
plt.colorbar()
plt.show()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']