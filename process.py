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


def main():
    times_hr = []
    time_spo2 = []
    signal_hr = []
    HR = []
    BLUE1_AVG = []
    GREEN1_AVG = []
    RED1_AVG = []
    BLUE2_AVG = []
    GREEN2_AVG = []
    RED2_AVG = []
    OX_SAT = []
    seconds_to_spo2 = 0
    json_index = 0
    t0 = time.time()
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    roi: ROI = ROI()
    heart_rate: HeartRate = HeartRate()
    spo2: SPO2 = SPO2()
    TSfr, TShr, JsonHR, JsonSPO2 = getTS()

    for filename in glob.glob(f'C:/Users/baett/OneDrive/Desktop/Proyecto final/Dataset proyecto/{SUBJECT}/*.png'):
        frame = cv.imread(filename)
        grayf = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        face = detector(grayf, 0)
        if len(face) > 0:
            times_hr.append(time.time() - t0)
            time_spo2.append(time.time() - t0)
            shape = predictor(grayf, face[0])
            shape = roi.shape_to_np(shape)
            ROI1, ROI2 = roi.get_cheeks(frame, shape)  

            # Getting HR signal    
            green_hr_signal = roi.get_hr_signal(ROI1, ROI2)
            signal_hr.append(green_hr_signal)

            # Gettign SPO2 signal
            b1_avg, g1_avg, r1_avg, b2_avg, g2_avg, r2_avg = roi.get_average_values(ROI1, ROI2)

            BLUE1_AVG.append(b1_avg)
            GREEN1_AVG.append(g1_avg)
            RED1_AVG.append(r1_avg)

            BLUE2_AVG.append(b2_avg)
            GREEN2_AVG.append(g2_avg)
            RED2_AVG.append(r2_avg)

            seconds_to_spo2 += 1

        # SPO2 measurement
        if len(GREEN1_AVG) == SPO2_WINDOW_SIZE:
            if seconds_to_spo2 >= SPO2_PROCESSING_DELAY: 
                ROI1_psd, ROI2_psd = roi.get_psd_roi(BLUE1_AVG, GREEN1_AVG, RED1_AVG, BLUE2_AVG, GREEN2_AVG, RED2_AVG, time_spo2)
                V1_bgr = np.stack((BLUE1_AVG, GREEN1_AVG, RED1_AVG), axis=-1)
                V2_bgr = np.stack((BLUE2_AVG, GREEN2_AVG, RED2_AVG), axis=-1)

                ROI_max = max([ROI1_psd, ROI2_psd])

                if ROI1_psd == ROI_max:
                    V_bgr = V1_bgr
                if ROI2_psd == ROI_max:
                    V_bgr = V2_bgr

                oxygen_saturation = spo2.get_spo2(V_bgr)
                OX_SAT.append(oxygen_saturation)

            for i in range(round(SPO2_WINDOW_SIZE/10)):
                BLUE1_AVG.pop(0) 
                GREEN1_AVG.pop(0)
                RED1_AVG.pop(0)
                BLUE2_AVG.pop(0) 
                GREEN2_AVG.pop(0)
                RED2_AVG.pop(0)
                time_spo2.pop(0)

        # HR measurement
        if len(signal_hr) == HR_WINDOW_SIZE:
            signal_processed_hr = roi.process_hr_signal(signal_hr, times_hr)
            heartrate = heart_rate.get_hr(signal_processed_hr)
            HR.append(heartrate)
            for i in range(round(HR_WINDOW_SIZE/10)):
                signal_hr.pop(0) 
                times_hr.pop(0)

        if len(HR) > 10:
            cv.putText(frame, '{:.0f}bpm'.format(np.mean(HR)), (200, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            cv.putText(frame, '{:.0f}%'.format(np.mean(OX_SAT)), (200, 90), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            try:    
                cv.putText(frame, '{:.0f}bpm'.format(JsonHR[json_index]), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
                cv.putText(frame, '{:.0f}%'.format(JsonSPO2[json_index]), (50, 90), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            except:
                continue

        if len(HR) > 20:
            for i in range(2):
                HR.pop(0)

        if len(OX_SAT) > 40:
            for i in range(4):        
                OX_SAT.pop(0)
                
        json_index+=1
        cv.imshow('frame', frame)
        #frame_array.append(frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()