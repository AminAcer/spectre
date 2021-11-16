import cv2
import imutils
import heapq
import time
import numpy as np

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
        redUpper = np.array([170, 255, 255])  # BRIGHTER
        redLower = np.array([160, 70, 0])  # DARKER

        #redUpper = np.array([190, 255, 255])  # BRIGHTER
        #redLower = np.array([140, 150, 0])  # DARKER
        cap = cv2.VideoCapture(1)
        print("Camera warming up...")
        time.sleep(2.0) #Let camera warm up

        while True:
            ret, frame = cap.read()
            frame = imutils.resize(frame, width=1000)
            blur = cv2.GaussianBlur(frame, (11, 11), 0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

            # Making masks
            mask = cv2.inRange(hsv, redLower, redUpper)
            # mask = cv2.erode(mask, None, iterations=2) # Removes small blobs
            mask = cv2.dilate(mask, None, iterations=2)  # Remobes small blobs
            redMask = cv2.bitwise_and(frame, frame, mask=mask)

            # Making contours
            contr = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contr = imutils.grab_contours(contr)
            center = None  # Center of ball

            #Only proceed if there is atleast 1 contour
            if len(contr) > 0:
                max_balls = 2     # Max number of balls tracked
                bigc = heapq.nlargest(max_balls, contr, key=cv2.contourArea)
                cnt = 0  # Count of detection
                for c in bigc:
                    ((x, y), radius) = cv2.minEnclosingCircle(c)
                    M = cv2.moments(c)
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    if radius > 1:  # Only draw for contours with min radius
                        cnt = cnt + 1
                        cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 2)
                        cv2.circle(frame, center, 5, (0, 255, 255), -1)
                cv2.putText(frame, "Detected:", (22, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (12, 15), (185, 56), (0, 255, 0), 2)
                cv2.putText(frame, str(int(cnt)), (158, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            else:
                cv2.putText(frame, "Detected:", (22, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (12, 15), (185, 56), (0, 0, 255), 2)
                cv2.putText(frame, "0", (158, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            if cv2.waitKey(1) == 27:
                break

            #self.telemetry(frame)

            cv2.imshow("SR-1 Live Feed", frame)

            #cv2.imshow(winName, redMask)

        cap.release()
        cv2.destroyAllWindows()

    def telemetry(self, frame):
        #serialport = serial.Serial('COM3', baudrate=9600, timeout=2)
        #serial.flushIn
        #data = serialport.readline().decode('ascii')
        #print(data)

        with open("data.txt", "r") as file:
            rdl = file.readlines()
            lastline = rdl[-2]
            lastline.strip()
            pitch = lastline.strip()[0:3].strip()
            roll = lastline.strip()[-3:].strip()

        pi = abs(float(pitch))
        ro = abs(float(roll))
        # Yellow BGR = [0,255,255], Red BGR = [0,0,255]
        pico = abs(int(round(pi / 90 * 255) - 255))
        roco = abs(int(round(ro / 90 * 255) - 255))

        cv2.putText(frame, "Pitch:", (22, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, pico, 240), 2)
        cv2.rectangle(frame, (12, 65), (185, 106), (0, pico, 240), 2)
        cv2.putText(frame, pitch, (114, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, pico, 240), 2)

        cv2.putText(frame, "Roll:", (22, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, roco, 240), 2)
        cv2.rectangle(frame, (12, 114), (185, 156), (0, roco, 240), 2)
        cv2.putText(frame, roll, (114, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, roco, 240), 2)


