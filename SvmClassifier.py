import numpy as np
import math
import random
import operator
import cv2
import csv


def loadDataset(
        filename,
        filename2,
        training_feature_vector=[],
        label_vectors=[],
        test_feature_vector=[], ):
    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(3):
                dataset[x][y] = float(dataset[x][y])
            training_feature_vector.append(dataset[x])
            label_vectors.append(str(dataset[x][3]))
        for i in training_feature_vector:
            if("day" in i):
                i.remove('day')
            else:
                i.remove('night')
    with open(filename2) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(3):
                dataset[x][y] = float(dataset[x][y])
            test_feature_vector.append(dataset[x])


def main(training_data, test_data):
    training_feature_vector = []
    test_feature_vector = []
    label_vectors = []
    loadDataset(training_data, test_data, training_feature_vector, label_vectors, test_feature_vector)
    from sklearn import svm
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(training_feature_vector, label_vectors)
    return clf.predict(test_feature_vector)