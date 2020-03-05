import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import pandas as pd
import numpy as np
from keras.models import load_model
import cv2


class CNNModel:

	def model(self):
		self.dataset = pd.read_csv('mnist.csv', header = None)
		self.x = np.array((self.dataset.iloc[:, 1:].values).reshape(len(self.dataset), 28, 28, 1))
		self.y = np.array(keras.utils.to_categorical(self.dataset.iloc[:, 0].values, num_classes = 10))

		self.model = Sequential()
		self.model.add(Convolution2D(32, 3, data_format = 'channels_last', activation = 'relu', input_shape = (28, 28, 1)))
		self.model.add(MaxPooling2D(pool_size = (2, 2)))
		self.model.add(Flatten())
		self.model.add(Dense(100))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(10))
		self.model.add(Activation('softmax'))
		self.model.compile(loss = 'categorical_crossentropy', optimizer = 'adadelta', metrics = ['accuracy'])
		self.model.fit(self.x, self.y, epochs = 15, batch_size = 128,  verbose = 1, shuffle = 1)

		self.model.save('mnist.h5')

class ImageTesting:
	def __init__(self, image, model):
		self.image = image
		self.model = load_model(model)

	def prediction(self):
		self.image = np.asarray(cv2.imread(self.image))
		self.resize = cv2.resize(self.image, (28, 28))
		self.gray = cv2.cvtColor(self.resize, cv2.COLOR_BGR2GRAY)
		self.image = (cv2.bitwise_not(self.gray)).reshape(1,28, 28, 1)
		return self.model.predict_classes(self.image)[0]

class LiveModel:
	def __init__(self, model):
		self.model = load_model(model)

	def activate(self):
		self.capture = cv2.VideoCapture(0)

		while True:
			_, self.frame = self.capture.read()
			self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
			self.blur = cv2.GaussianBlur(self.gray, (5, 5), 0)
			self.edges = cv2.Canny(self.blur, 50, 200, 255)
			self.retrieve, self.treshed = cv2.threshold(self.edges, 90, 255, cv2.THRESH_BINARY_INV)
			self.contours, _ = cv2.findContours(self.edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
			self.rectangles = [cv2.boundingRect(i) for i in self.contours]

			for r in self.rectangles:
				for contour in self.contours:
					self.area = cv2.contourArea(contour)
					if self.area > 60: #minimum is 2000
						cv2.rectangle(self.frame, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (0, 255, 0), 3)
						self.image = np.asarray(self.blur)
						self.image = cv2.resize(self.image, (28, 28))
						self.image = (cv2.bitwise_not(self.image)).reshape(1,28, 28, 1)
						self.prediction = self.model.predict_classes(self.image)

						if self.prediction[0] in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
							cv2.putText(self.frame, str(int(self.prediction[0])), (r[0], r[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
					# print ("predicted: ", str(self.prediction))
			cv2.imshow('Frame', self.frame)

			self.key = cv2.waitKey(1)

			if self.key == 27:
				break
		self.capture.release()
		cv2.destroyAllWindows()

LiveModel(model = 'mnist.h5').activate()


# CNNModel().model()
# prediction = ImageTesting(image = 'eight.jpg', model = 'mnist.h5').prediction()
# print ('Prediction made by model : ', prediction)