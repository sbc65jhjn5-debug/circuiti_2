import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    C = 10e-9 # F
    R = 4.673e3 # Ohm
 
    # Tensione misurata ai capi di R

    tempo_R = np.loadtxt ("RC_tempi_R.txt") * 1e-6
    V_R = np.loadtxt ("RC_V_R.txt")

    sigma_V_R = np.ones (len (V_R)) * 0.3 # V

    # Tensione misurata ai capi di C

    tempo_C = np.loadtxt ("RC_tempi_C.txt") * 1e-6 
    V_C = np.loadtxt ("RC_V_C.txt")
    sigma_V_C = np.ones (len (V_C)) * 0.04

















































