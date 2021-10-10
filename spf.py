import cv2
import numpy as np

class spectre(object):

    def __init__(self):
        pass

    def LiveFeed(self,):
        winName = "SR-1 Live Feed"
        cv2.namedWindow(winName)
        cap = cv2.VideoCapture(1)

        if cap.isOpened():
            ret, frame = cap.read()
        else:
            ret = False

        while ret:
            ret, frame = cap.read()
            frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
            cv2.imshow(winName, frame)

            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

    def detRed(self):
        winName = "SR-1 Red Detection"
        cv2.namedWindow(winName)
        cap = cv2.VideoCapture(1)

        if cap.isOpened():
            ret, frame = cap.read()
        else:
            ret = False

        while ret:
            ret, frame = cap.read()
            frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            redUpper = np.array([190, 255, 255])  #BRIGHTER
            redLower = np.array([140, 150, 0])   #DARKER

            redMask = cv2.inRange(hsv, redLower, redUpper)
            rMask = cv2.bitwise_and(frame, frame, mask=redMask)

            cv2.imshow("SR-1 LiveFeed", frame)
            cv2.imshow(winName, rMask)

            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()