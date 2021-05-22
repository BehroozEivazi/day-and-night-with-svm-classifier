import cv2 as cv2
import numpy as np
import os


def histogram(source, isTest=False):
    if (isTest):
        image = source
        chans = cv2.split(image)
        colors = ('b', 'g', 'r')
        features = []
        feature_data = ''
        counter = 0
        for (chan, color) in zip(chans, colors):
            counter = counter + 1
            hist = cv2.calcHist([chan], [0], None, [256], [0, 32])
            features.extend(hist)
            elem = np.argmax(hist)
            if counter == 1:
                blue = str(elem)
            elif counter == 2:
                green = str(elem)
            elif counter == 3:
                red = str(elem)
                feature_data = red + ',' + green + ',' + blue
        with open('testDayAndNight.data', 'w') as myfile:
            myfile.write(feature_data)
    else:
        if 'day' in source:
            data_source = 'day'
        elif "night" in source:
            data_source = 'night'
        image = cv2.imread(source)
        chans = cv2.split(image)
        colors = ('b', 'g', 'r')
        features = []
        feature_data = ''
        counter = 0
        for (chan, color) in zip(chans, colors):
            counter = counter + 1
            hist = cv2.calcHist([chan], [0], None, [256], [0, 32])
            features.extend(hist)
            elem = np.argmax(hist)
            if counter == 1:
                blue = str(elem)
            elif counter == 2:
                green = str(elem)
            elif counter == 3:
                red = str(elem)
                feature_data = red + ',' + green + ',' + blue

        with open('TrainDayAndNight.data', 'a') as myfile:
            myfile.write(feature_data + ',' + data_source + '\n')


path = "D:/dataset/dsn/training/"


def training():
    for f in os.listdir(path + 'day'):
        histogram(path + 'day/' + f)

    for f in os.listdir(path + "night"):
        histogram(path + 'night/' + f)
