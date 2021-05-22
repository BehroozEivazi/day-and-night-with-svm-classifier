import os

import cv2
import histogram_days
import SvmClassifier

source_image = cv2.imread('D:/dataset/dsn/test/night/20151102_011134.jpg')
prediction = 'n.a.'


PATH = './TrainDayAndNight.data'

if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
    print ('training data is ready, classifier is loading...')
else:
    histogram_days.training()

histogram_days.histogram(source_image,True)

prediction = SvmClassifier.main('TrainDayAndNight.data', 'testDayAndNight.data')
cv2.putText(
    source_image,
    'Prediction: ' + prediction[0],
    (15, 45),
    cv2.FONT_HERSHEY_PLAIN,
    3,
    200,
    )

cv2.imshow("{} image".format(prediction[0]),source_image)
cv2.waitKey(0)
