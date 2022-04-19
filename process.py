from ast import Return
from mimetypes import init
from typing import final
import numpy as np
import dlib
import cv2 as cv
import time
import glob
from libraries.roi_processing import ROI
from libraries.heart_rate import HeartRate
from libraries.spo2 import SPO2
from libraries.json_processing import getTS
from config import SUBJECT, HR_WINDOW_SIZE, SPO2_WINDOW_SIZE, SPO2_PROCESSING_DELAY

class Process:
    def __init__(self) -> None:
        self.times_hr = []
        self.time_spo2 = []
        self.signal_hr = []
        self.HR = []
        self.BLUE1_AVG = []
        self.GREEN1_AVG = []
        self.RED1_AVG = []
        self.BLUE2_AVG = []
        self.GREEN2_AVG = []
        self.RED2_AVG = []
        self.OX_SAT = []
        self.seconds_to_spo2 = 0
        self.json_index = 0
        self.t0 = time.time()
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.roi: ROI = ROI()
        self.heart_rate: HeartRate = HeartRate()
        self.spo2: SPO2 = SPO2()
        self.TSfr, self.TShr, self.JsonHR, self.JsonSPO2 = getTS()
    
    def process(self, frame):
        #for filename in glob.glob(f'C:/Users/baett/OneDrive/Desktop/Proyecto final/Dataset proyecto/{SUBJECT}/*.png'):
        #self.frame = cv.imread(self.filename)
        grayf = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        face = self.detector(grayf, 0)
        if len(face) > 0:
            self.times_hr.append(time.time() - self.t0)
            self.time_spo2.append(time.time() -self.t0)
            shape = self.predictor(grayf, face[0])
            shape = self.roi.shape_to_np(shape)
            ROI1, ROI2 = self.roi.get_cheeks(frame, shape)  

            # Getting HR signal    
            green_hr_signal = self.roi.get_hr_signal(ROI1, ROI2)
            self.signal_hr.append(green_hr_signal)

            # Gettign SPO2 signal
            b1_avg, g1_avg, r1_avg, b2_avg, g2_avg, r2_avg = self.roi.get_average_values(ROI1, ROI2)

            self.BLUE1_AVG.append(b1_avg)
            self.GREEN1_AVG.append(g1_avg)
            self.RED1_AVG.append(r1_avg)
            
            self.BLUE2_AVG.append(b2_avg)
            self.GREEN2_AVG.append(g2_avg)
            self.RED2_AVG.append(r2_avg)

            self.seconds_to_spo2 += 1

        # SPO2 measurement
        if len(self.GREEN1_AVG) == SPO2_WINDOW_SIZE:
            if self.seconds_to_spo2 >= SPO2_PROCESSING_DELAY: 
                ROI1_psd, ROI2_psd = self.roi.get_psd_roi(self.BLUE1_AVG, self.GREEN1_AVG, self.RED1_AVG, self.BLUE2_AVG, self.GREEN2_AVG, self.RED2_AVG, self.time_spo2)
                V1_bgr = np.stack((self.BLUE1_AVG, self.GREEN1_AVG, self.RED1_AVG), axis=-1)
                V2_bgr = np.stack((self.BLUE2_AVG, self.GREEN2_AVG, self.RED2_AVG), axis=-1)

                ROI_max = max([ROI1_psd, ROI2_psd])

                if ROI1_psd == ROI_max:
                    V_bgr = V1_bgr
                if ROI2_psd == ROI_max:
                    V_bgr = V2_bgr

                oxygen_saturation = self.spo2.get_spo2(V_bgr)
                self.OX_SAT.append(oxygen_saturation)

            for i in range(round(SPO2_WINDOW_SIZE/10)):
                self.BLUE1_AVG.pop(0) 
                self.GREEN1_AVG.pop(0)
                self.RED1_AVG.pop(0)
                self.BLUE2_AVG.pop(0) 
                self.GREEN2_AVG.pop(0)
                self.RED2_AVG.pop(0)
                self.time_spo2.pop(0)

        # HR measurement
        if len(self.signal_hr) == HR_WINDOW_SIZE:
            signal_processed_hr = self.roi.process_hr_signal(self.signal_hr, self.times_hr)
            heartrate = self.heart_rate.get_hr(signal_processed_hr)
            self.HR.append(heartrate)
            for i in range(round(HR_WINDOW_SIZE/10)):
                self.signal_hr.pop(0) 
                self.times_hr.pop(0)

        if len(self.HR) > 20:
            for i in range(2):
                self.HR.pop(0)

        if len(self.OX_SAT) > 40:
            for i in range(4):        
                self.OX_SAT.pop(0)

        self.json_index+=1

        if len(self.HR) > 10:
            final_HR = np.mean(self.HR)
            final_SPO2 = np.mean(self.OX_SAT)
            
            return final_HR, final_SPO2

        return 0, 0
        #cv.putText(self.frame, '{:.0f}bpm'.format(np.mean(self.HR)), (200, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        #    cv.putText(self.frame, '{:.0f}%'.format(np.mean(self.OX_SAT)), (200, 90), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        #    try:    
        #        cv.putText(self.frame, '{:.0f}bpm'.format(self.JsonHR[json_index]), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
        #        cv.putText(self.frame, '{:.0f}%'.format(self.JsonSPO2[json_index]), (50, 90), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        #    except:
        #        return
#
#
        #cv.imshow('frame', self.frame)
        ##frame_array.append(frame)
        #if cv.waitKey(1) & 0xFF == ord('q'):
        #    return

    #if __name__ == '__main__':
    #    main()