import numpy as np


class SPO2:
    def __init__(self) -> None:
        pass

    def get_spo2(self, V_bgr):
        #(blue, green, red) = cv.split(V_bgr)
        blue = V_bgr[:,0]
        red = V_bgr[:,2]

        average_blue = np.mean(blue)
        std_blue = np.std(blue)

        average_red = np.mean(red)
        std_red = np.std(red)

        spo2 = 102-5*(std_red/average_red)/(std_blue/average_blue)

        return spo2