import cv2
import numpy as np
import imutils
import heapq


class Spectre(object):

    def __init__(self):
        pass

    def LiveFeed(self):
        winName = "SR-1 Live Feed"
        cv2.namedWindow(winName)
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
            cv2.imshow(winName, frame)

            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

    def detRed(self):
        winName = "SR-1 Red Detection"
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            frame = imutils.resize(frame, width=1000)
            blur = cv2.GaussianBlur(frame, (11,11), 0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            redUpper = np.array([190, 255, 255])  # BRIGHTER
            redLower = np.array([140, 150, 0])   # DARKER

            # Making masks
            mask = cv2.inRange(hsv, redLower, redUpper)
            mask = cv2.erode(mask, None, iterations=2) # Removes small blobs
            mask = cv2.dilate(mask, None, iterations=2) # Remobes small blobs
            redMask = cv2.bitwise_and(frame, frame, mask=mask)

            # Making contours
            contr = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contr = imutils.grab_contours(contr)
            center = None #Center of ball

            if len(contr) > 0:
                bigc = heapq.nlargest(3, contr, key=cv2.contourArea) # Biggest contour
                cnt = 0  # Count of detections
                for c in bigc:
                    ((x, y), radius) = cv2.minEnclosingCircle(c)
                    M = cv2.moments(c)
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    if radius > 5: #Only draw for contours with min radius
                        cnt = cnt + 1
                        cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 20), 4)
                        cv2.circle(frame, center, 5, (0, 255, 255), -1)
                        cv2.putText(frame, "Detections: ", (22, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 1)
                        cv2.rectangle(frame, (12, 15), (195, 60), (0, 255, 255), 2)
                cv2.putText(frame, str(int(cnt)), (172, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 1)

            else:
                cv2.putText(frame, "No Detections", (22, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
                cv2.rectangle(frame, (12, 15), (215, 60), (0, 0, 255), 2)

            if cv2.waitKey(1) == 27:
                break

            cv2.imshow("SR-1 Live Feed", frame)
            #cv2.imshow(winName, redMask)

        cap.release()
        cv2.destroyAllWindows()