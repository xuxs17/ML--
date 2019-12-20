# Keras 多层感知机神经网络样例
from keras.models import Sequential
from keras.layers import Dense
import numpy
# 加载数据
dataset = numpy.loadtxt("data.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]
# 1. 定义网络
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# 2. 编译网络
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 3. 训练网络
history = model.fit(X, Y, epochs=100, batch_size=10)
# 4. 评价网络
loss, accuracy = model.evaluate(X, Y)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
# 5. 进行预测
probabilities = model.predict(X)
predictions = [float(round(x)) for x in probabilities]
accuracy = numpy.mean(predictions == Y)
print("Prediction Accuracy: %.2f%%" % (accuracy*100))

