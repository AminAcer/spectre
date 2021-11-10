import cv2
import numpy as np
from spf import Spectre as sp
#from spf import Gui as gui
from imutils.video import VideoStream
import argparse
import time

sr1 = sp()

# sr1.LiveFeed()
sr1.detRed()