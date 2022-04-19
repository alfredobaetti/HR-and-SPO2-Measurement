import cv2 as cv
import numpy as np
from scipy import signal
from libraries.signal_processing import SignalProcessing
from libraries.moving_average_filter import MovingAverageFilter
from config import fps


class ROI:
    def __init__(self) -> None:
        self.moving_average: MovingAverageFilter = MovingAverageFilter()
        self.signal_processing: SignalProcessing = SignalProcessing()
        self.green = []
    
    def average(self, image):
        sum = 0
        width = image.shape[1]
        height = image.shape[0]
        for i in range(0,height):
            for j in range(0,width):
                sum = sum +image[i][j]
        return sum/(width*height)

    def shape_to_np(self,shape,dtype="int"):
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)    
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def get_cheeks(self, frame, shape):
        #cv.rectangle(frame,(shape[54][0], shape[29][1]), #draw rectangle on right and left cheeks
        #            (shape[25][0],shape[33][1]), (0,255,0), 0)
        #cv.rectangle(frame, (shape[5][0], shape[29][1]), 
        #        (shape[48][0],shape[33][1]), (0,255,0), 0) 
        #cv.rectangle(frame, (shape[20][0], shape[71][1]), 
        #        (shape[23][0],shape[74][1]), (0,255,0), 0) 
        ROI1 = frame[shape[29][1]:shape[33][1], #right cheek
                shape[54][0]:shape[25][0]]  
        ROI2 =  frame[shape[29][1]:shape[33][1], #left cheek
                shape[5][0]:shape[48][0]]  
        #ROI3 = frame[shape[71][1]:shape[74][1], #left cheek
        #        shape[20][0]:shape[23][0]] 
        # 
        #cv.rectangle(frame,(shape[54][0], shape[29][1]), #draw rectangle on right and left cheeks
        #            (shape[12][0],shape[33][1]), (0,255,0), 0)
        #cv.rectangle(frame, (shape[4][0], shape[29][1]), 
        #        (shape[48][0],shape[33][1]), (0,255,0), 0) 
        #ROI1 = frame[shape[29][1]:shape[33][1], #right cheek
        #        shape[54][0]:shape[12][0]]  
        #ROI2 =  frame[shape[29][1]:shape[33][1], #left cheek
        #        shape[4][0]:shape[48][0]] 
        #for (x, y) in shape:
        #    cv.circle(frame, (x, y), 1, (0, 0, 255), -1) #draw facial landmarks 
        return ROI1, ROI2

    def process_hr_signal(self, signal_hr, times_hr):
        t = np.arange(0, len(signal_hr)/fps, len(signal_hr)/fps/300)

        """FILTRO DETREND"""
        signal = self.signal_processing.signal_detrending(signal_hr)

        """INTERPOLATION"""
        signal = self.signal_processing.interpolation(signal, times_hr)

        """NORMALIZACIÓN"""
        signal = self.signal_processing.normalization(signal)

        """FILTRO PARA SUAVIZAR CURVA"""
        for i in range(len(signal)):
            signal[i] = self.moving_average.start(signal[i])
        
        return signal

    def get_hr_signal(self, ROI1, ROI2):
        b1, g1, r1 = self.signal_processing.get_channel_signal(ROI1)
        b2, g2, r2 = self.signal_processing.get_channel_signal(ROI2)
        g = (g1+g2)/2

        return g

    def get_average_values(self, ROI1, ROI2):
        (B1, G1, R1) = cv.split(ROI1)
        (B2, G2, R2) = cv.split(ROI2)

        b1_avg = self.average(B1)
        g1_avg = self.average(G1)
        r1_avg = self.average(R1)
        b2_avg = self.average(B2)
        g2_avg = self.average(G2)
        r2_avg = self.average(R2)

        return b1_avg, g1_avg, r1_avg, b2_avg, g2_avg, r2_avg

    def get_psd_roi(self, BLUE1_AVG, GREEN1_AVG, RED1_AVG, BLUE2_AVG, GREEN2_AVG, RED2_AVG, time_spo2):
        BLUE1_AVG = self.signal_processing.fir_filter(BLUE1_AVG)
        GREEN1_AVG = self.signal_processing.fir_filter(GREEN1_AVG)
        RED1_AVG = self.signal_processing.fir_filter(RED1_AVG)

        BLUE2_AVG = self.signal_processing.fir_filter(BLUE2_AVG)
        GREEN2_AVG = self.signal_processing.fir_filter(GREEN2_AVG)
        RED2_AVG = self.signal_processing.fir_filter(RED2_AVG)

        Xs1 = 3 * np.array(RED1_AVG) - 2 * np.array(GREEN1_AVG)
        Ys1 = 1.5 * np.array(RED1_AVG) + np.array(GREEN1_AVG) - 1.5 * np.array(BLUE1_AVG)

        S1 = Xs1 / Ys1 - 1

        Xs2 = 3 * np.array(RED2_AVG) - 2 * np.array(GREEN2_AVG)
        Ys2 = 1.5 * np.array(RED2_AVG) + np.array(GREEN2_AVG) - 1.5 * np.array(BLUE2_AVG)

        S2 = Xs2 / Ys2 - 1

        #INTERPOLATION
        S1 = self.signal_processing.interpolation(S1, time_spo2)
        S2 = self.signal_processing.interpolation(S2, time_spo2)

        for i in range(len(S1)):
            S1[i] = self.moving_average.start(S1[i])
            S2[i] = self.moving_average.start(S2[i])

        _, S1_psd = signal.welch(S1)
        _, S2_psd = signal.welch(S2)

        ROI1_psd = max(S1_psd)
        ROI2_psd = max(S2_psd)

        return ROI1_psd, ROI2_psd