import cv2 as cv

capture = cv.VideoCapture(0)
fps = capture.get(cv.CAP_PROP_FPS)

SUBJECT = "01-01"

HR_WINDOW_SIZE = 300
SPO2_WINDOW_SIZE = 60
SPO2_PROCESSING_DELAY = 660 

FREQ_MIN = 0.7
FREQ_MAX = 2.7