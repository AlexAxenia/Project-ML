import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Dense, Dropout

# Se incarca datele din MNIST
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizarea setului de date
X_train = X_train/255.0
X_test = X_test/255.0

# Se creeaza reteaua neurala
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(Dropout(0.2))

model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(Dropout(0.2))

model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

# Compilarea si optimizarea modelului
model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Antrenarea modelului
model.fit(X_train, y_train, epochs=3)

# Evaluarea modelului
val_loss, val_acc = model.evaluate(X_test, y_test)
print(val_loss)
print(val_acc)

# Salvarea modelului
model.save('handwritten_digits.model')