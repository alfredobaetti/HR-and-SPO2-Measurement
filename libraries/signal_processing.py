import cv2
import numpy as np
from scipy import signal
import scipy.fftpack as fftpack
from scipy.signal import firwin, lfilter, kaiserord, hanning, filtfilt, butter, freqz
from config import fps


class SignalProcessing():
    def __init__(self):
        pass
    
    def get_channel_signal(self, ROI):
        b, g, r = cv2.split(ROI)

        g = np.mean(g)
        r = np.mean(r)
        b = np.mean(b)

        return b, g, r    

    def normalization(self, data_buffer):
        '''
        normalize the input data buffer
        '''
        
        #normalized_data = (data_buffer - np.mean(data_buffer))/np.std(data_buffer)
        normalized_data = data_buffer/np.linalg.norm(data_buffer)
        
        return normalized_data
    
    def signal_detrending(self, data_buffer):
        '''
        remove overall trending
        
        '''
        detrended_data = signal.detrend(data_buffer)
        
        return detrended_data
        
    def interpolation(self, data_buffer, times):
        '''
        interpolation data buffer to make the signal become more periodic (advoid spectral leakage)
        '''
        L = len(data_buffer)
        
        even_times = np.linspace(times[0], times[-1], L)
        
        interp = np.interp(even_times, times, data_buffer[:])
        interpolated_data = np.hamming(L) * interp
        return interpolated_data

    def fft_filter(self, signal, freq_min, freq_max, fps):
        fft = fftpack.fft(signal, axis=0)
        frequencies = fftpack.fftfreq(signal.shape[0], d=1.0 / fps)
        bound_low = (np.abs(frequencies - freq_min)).argmin()
        bound_high = (np.abs(frequencies - freq_max)).argmin()
        fft[:bound_low] = 0
        fft[bound_high:-bound_high] = 0
        fft[-bound_low:] = 0

        up_cutoff = (np.abs(frequencies - 7.5)).argmin() #prueba para eliminar influencia de respiración
        low_cutoff = (np.abs(frequencies - 3)).argmin()

        fft[low_cutoff:up_cutoff] = 0
        fft[-low_cutoff:-up_cutoff] = 0

        return fft, frequencies

    def butter_lowpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def fir_filter(self, x):
        order = 6
        fs = fps      # sample rate, Hz
        cutoff = 4 
        filtered_x = self.butter_lowpass_filter(x, cutoff, fs, order)
        return filtered_x
        
    
        
    
        
        
        
        
        
        
        
        