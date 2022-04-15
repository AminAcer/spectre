import cv2
import imutils
import heapq
import time
from datetime import datetime, date
import numpy as np
import keyboard

def main():
    winName = "PR-1 Live Feed"
    #redUpper = np.array([170, 255, 255])  # BRIGHTER
    #redLower = np.array([160, 70, 0])  # DARKER

    redUpper = np.array([190, 255, 255])  # BRIGHTER
    redLower = np.array([140, 150, 0])  # DARKER
    cap = cv2.VideoCapture(1) #2
    ftime = datetime.now()
    fdate = date.today()
    ft = ftime.strftime("%I:%M:%S %p")
    fd = fdate.strftime("%b-%d-%Y")
    print("Camera warming up...")
    time.sleep(2.0) #Let camera warm up

    try:
        with open("telemetry.txt", "a") as f:
            f.write("NA  NA" + "\n")
            f.write("NA  NA" + "\n")
            f.write("NA  NA" + "\n")
    except PermissionError:
        pass
    maxb = 2  # Total number of balls on field
    mainc = 0 # Duration of entire flight
    detc = 0 # Total duration of detection
    maxc = 0 # Duration of over detections
    misd = [] # Amount of misdetections (Uses length of list of time stamps)
    armed = False
    dettime = []

    while True:
        ret, frame = cap.read()
        #frame = cv2.resize(frame, None, fx=1, fy=1.1, interpolation=cv2.INTER_AREA)
        frame = cv2.resize(frame, None, fx=2.4, fy=2.1, interpolation=cv2.INTER_AREA)
        frame = imutils.resize(frame, width=1100)
        blur = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # Making masks
        gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
        #gm = cv2.erode(gray, None, iterations=2)  # Removes small blobs
        gm = cv2.dilate(gray, None, iterations=2)  # Remobes small blobs

        mask = cv2.inRange(hsv, redLower, redUpper)
        mask = cv2.erode(mask, None, iterations=2) # Removes small blobs
        mask = cv2.dilate(mask, None, iterations=2)  # Remobes small blobs
        redMask = cv2.bitwise_and(frame, frame, mask=mask)
        circles = cv2.HoughCircles(gm, cv2.HOUGH_GRADIENT, 1.2, 100, minRadius=5)
        # if circles is not None:
        #     # convert the (x, y) coordinates and radius of the circles to integers
        #     circles = np.round(circles[0, :]).astype("int")
        #     cirList = []
        #     for (x, y, r) in circles:
        #         lst = [x,y,r]
        #         cirList.append(lst)
        #         print(cirList)
        #         # draw the circle in the output image, then draw a rectangle
        #         # corresponding to the center of the circle
        #         #cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
        #         #cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        # Making contours
        contr = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contr = imutils.grab_contours(contr)
        center = None  # Center of ball

        if keyboard.is_pressed('s'):
            armed = True
        if keyboard.is_pressed('p'):
            armed = False
        if keyboard.is_pressed('space') and armed is True: # Save timestamp for detections
            time.sleep(0.1)
            if keyboard.is_pressed('space') is False:
                tstamp = datetime.now().strftime("%I:%M:%S %p")
                dettime += [tstamp]
        if keyboard.is_pressed('x') and armed is True: #Add to misdetections counter
            time.sleep(0.1)
            if keyboard.is_pressed('x') is False:
                mtime = datetime.now().strftime("%I:%M:%S %p")
                misd += [mtime]
        if armed is False:
            armc = (0, 0, 255)
        else:
            armc = (0, 255, 0)
        #Only proceed if there is atleast 1 contour
        if len(contr) > 0 and armed is True: #Draw outlines and positive detections hud
            max_balls = maxb + 1     # Max number of balls tracked
            bigc = heapq.nlargest(max_balls, contr, key=cv2.contourArea)
            cnt = 0  # Count of detection
            for c in bigc:
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                if radius > 5:  # Only draw for contours with min radius
                    # for u in range(len(cirList)):
                    #     if abs(cirList[u][0] - int(x)) < 15:
                            cnt = cnt + 1
                            cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 2)
                            cv2.circle(frame, center, 5, (0, 255, 255), -1)
            if armed is True:
                detc += 1
            if cnt > maxb and armed is True:
                maxc += 1
                armc = (0, 100, 0)
            hudDisp(frame, "", str(int(cnt)), (0, 0), (18, 146), (12, 65), (90, 156), armc, 2, 3.3)  # Detections HUD
        else: #Draw 0 detections hud
            hudDisp(frame, "", "0", (0, 0), (18, 146), (12, 65), (90, 156), armc, 2, 3.3) #0 Detections HUD

        if armed is True:
            mainc += 1

        now = datetime.now()
        ct = now.strftime("%I:%M:%S %p")
        hudDisp(frame, "", str(ct), (0, 0), (20, 45), (12, 12), (208, 56), (30, 30, 30), 2, 0.9) #Time HUD

        # hudDisp(frame, "Armed", "", (20, 195), (0, 0), (12, 165), (208, 206), armc, 2, 0.9) #Arming HUD
        telemetry(frame)  # Read and display telemetry data

        cv2.imshow(winName, frame)
        #cv2.imshow(winName, redMask)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    flightSum(fd, ft, mainc, detc, maxc, misd, dettime)

def telemetry(frame):
    with open("telemetry.txt", "r") as file:
        rdl = file.readlines()
        try:
            lastline = rdl[-2]
        except IndexError:
            lastline = "NA  NA"
        lastline.strip()
        pitch = lastline.strip()[0:3].strip()
        roll = lastline.strip()[-3:].strip()
    try:
        pi = abs(float(pitch))
        ro = abs(float(roll))
    except ValueError:
        pi = 0
        ro = 0
    # Yellow BGR = [0,255,255], Red BGR = [0,0,255]
    pico = abs(int(round(pi / 90 * 255) - 255))
    roco = abs(int(round(ro / 90 * 255) - 255))

    hudDisp(frame, "P:", pitch, (108, 95), (140, 95), (100, 65), (208, 106), (0, pico, 240), 2, 0.9)
    hudDisp(frame, "R:", roll, (108, 145), (140, 145), (100, 115), (208, 156), (0, roco, 240), 2, 0.9)

def hudDisp(frame, name, data, nc, dc, rtl, rbr, bgr, thick, fs):
    #Display data with box around it
    #frame = frame to display on, name = name of data, data = data displayed after name
    #nc = name cords for text, dc = data cords for text
    #rtl = top left rect cords, rbr = bottom right rect cords
    #bgr = BGR values, thick = thickness of text, fs = font size
    cv2.putText(frame, name, nc, cv2.FONT_HERSHEY_SIMPLEX, fs, bgr, thick)
    cv2.rectangle(frame, rtl, rbr, bgr, thick)
    cv2.putText(frame, data, dc, cv2.FONT_HERSHEY_SIMPLEX, fs, bgr, thick)

def flightSum(fdate, ftime, mainc, detc, maxc, misd, dettime):
    # Flight Summary
    ms = int(round(mainc/18))
    ds = int(round(detc/18))
    xs = int(round(maxc/18))
    mm = int(ms/60)
    dm = int(ds/60)
    xm = int(xs/60)

    tstamp = []
    miscount = []
    for i in dettime: # Remove duplicated time stamps
        if i not in tstamp:
            tstamp.append(i)
    tsint = len(tstamp)
    for i in misd:  # Remove duplicated time stamps
        if i not in miscount:
            miscount.append(i)
    misint = len(miscount)
    if len(miscount) > len(tstamp):
        dif = len(miscount) - len(tstamp)
        for i in range(dif):
            tstamp += ["           "]
    elif len(tstamp) > len(miscount):
        dif = len(tstamp) - len(miscount)
        for i in range(dif):
            miscount += ["           "]

    print("")
    print("")
    print("")

    print("\033[1m" + "Flight Debrief:", fdate, "|", ftime + "\033[0m")
    print("==============================================")
    print("Summary:")

    if mm > 0:
        print("   Total flight time:", str(mm), "min", str(ms - 60*mm), "sec")
    else:
        print("   Total flight time:", str(ms), "sec")
    if dm > 0:
        print("   Total detection time:", str(dm), "min", str(ds - 60*dm), "sec")
    else:
        print("   Total detection time:", str(ds), "sec")
    print("----------------------------------------------")
    print("Errors:")
    print("   Misdetections:", str(misint))
    if xm > 0:
        print("   Total time of excess detections:", str(xm), "min", str(xs - 60*xm), "sec")
    else:
        print("   Total time of excess detections:", str(xs), "sec")
    print("==============================================")
    print("Timestamps:", str(tsint))
    print("")

    save = input("Would you like to save this flight? Y/N: ")
    if save == "Y" or save == "y":
        notes = input("Notes for flight: ")
        with open('FlightDebriefs.txt', 'a') as f:
            f.write("Flight Debrief: " + fdate + " | " + ftime + "\n")
            if len(notes) > 0:
                f.write("Notes: " + notes + "\n")
            f.write("==============================================" + "\n")
            f.write("Summary:" + "\n")
            if mm > 0:
                f.write("   Total flight time: " + str(mm) + " min " + str(ms - 60 * mm) + " sec" + "\n")
            else:
                f.write("   Total flight time: " + str(ms) + " sec" + "\n")
            if dm > 0:
                f.write("   Total detection time: " + str(dm) + " min " + str(ds - 60 * dm) + " sec" + "\n")
            else:
                f.write("   Total detection time: " + str(ds) + " sec" + "\n")
            f.write("----------------------------------------------" + "\n")
            f.write("Errors:" + "\n")
            f.write("   Misdetections: " + str(misint) + "\n")
            if xm > 0:
                f.write("   Total time of excess detections: " + str(xm) + " min " + str(xs - 60 * xm) + " sec" + "\n")
            else:
                f.write("   Total time of excess detections: " + str(xs) + " sec" + "\n")
            f.write("==============================================" + "\n")
            if len(tstamp) > 0 or len(miscount) > 0:
                big = max(len(tstamp), len(miscount))
                if len(dettime) == 0:
                    f.write("Misdetections:" + "\n")
                    for i in range(big):
                        f.write("   " + miscount[i] + "\n")
                elif len(misd) == 0:
                    f.write("Timestamps:" + "\n")
                    for i in range(big):
                        f.write("   " + tstamp[i] + "\n")
                else:
                    f.write("Misdetections:            Timestamps:" + "\n")
                    for i in range(big):
                        f.write("   " + miscount[i] + "               " + tstamp[i] + "\n")
            else:
                f.write("No misdetections or timestamps" + "\n")
            f.write("\n")
            f.write("##############################################################################" + "\n")
            f.write("\n")
        print("Flight saved")
    elif save == "N" or save == "n":
        print("Flight discarded")
    else:
        print("Flight discarded")

def liveFeed():
    """"""
    winName = "PR-1 Live Feed"
    cv2.namedWindow(winName)
    cap = cv2.VideoCapture(2)
    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=1100)
        #frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
        cv2.imshow(winName, frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
