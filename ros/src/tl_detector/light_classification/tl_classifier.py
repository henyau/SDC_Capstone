from styx_msgs.msg import TrafficLight

from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers.core import Flatten, Dense, Lambda,  Dropout
from tensorflow.contrib.keras.python.keras.layers.convolutional import Conv2D
import tensorflow as tf
import cv2
import time
import numpy as np

LOGITS = "logits"

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed

class TLClassifier(object):

    def __init__(self):
        reg = tf.contrib.layers.l2_regularizer(1e-3)
        model = Sequential()
        model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(114, 52, 3)))
        model.add(
            Conv2D(filters=24, kernel_size=5, strides=1, padding='same', activation='relu', kernel_regularizer=reg))
        model.add(
            Conv2D(filters=36, kernel_size=5, strides=1, padding='same', activation='relu', kernel_regularizer=reg))
        model.add(
            Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=reg))
        model.add(Flatten())
        model.add(Dense(50))
        model.add(Dense(10))
        model.add(Dense(6, activation='softmax', name=LOGITS))
        model.load_weights('/home/workspace/SDC_Capstone/ros/src/tl_detector/light_classification/weights.h5')
        self.model = model


    @timeit
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO: resize
        resized = cv2.resize(image, (52, 114), interpolation=cv2.INTER_AREA)
        # # Blur
        # blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        #TODO: convert
        images = np.array([cv2.cvtColor(resized, cv2.COLOR_BGR2YUV)])
        #Predict
        label = self.model.predict(images)[0]
        if label == 0:
            res = TrafficLight.RED
        elif label == 1:
            res = TrafficLight.YELLOW
        elif label == 2:
            res = TrafficLight.GREEN
        else:
            res = TrafficLight.UNKNOWN
        return res
