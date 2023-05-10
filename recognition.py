import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Dense, Dropout

# Incarcarea modelului
model = tf.keras.models.load_model('handwritten_digits.model')

# Prezicerea imaginilor din fisier
image_number = 1
while os.path.isfile('digits/digit{}.png'.format(image_number)):
    try:
        img = cv2.imread('digits/digit{}.png'.format(image_number))[:, :, 0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print("Cifra este {}".format(np.argmax(prediction)))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        image_number += 1

    except:
        print("Nu s-a putut citi imaginea! Se trece la urmatoarea imagine...")
        image_number += 1
