import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

if __name__ == "__main__":

    L_stima = 42.6e-3 # Henry
    C = 10e-9 # F
    delta_V = 0.04 # V

    # configurazione A: smorzamento critico

    R_A = 1210 # Ohm

    tempi_A = np.loadtxt ("tempi_A.txt")
    V_A = np.loadtxt ("V_A.txt")

    # configurazione B: sottosmorzato

    R_B = 200 # ohm

    tempi_B = np.loadtxt("tempi_B.txt")
    V_B = np.loadtxt ("V_B.txt")

    Delta_B =  np.ones (len (tempi_B)) * 0.008

    # configurazione C: sovrasmorzato

    R_C = 5000 # ohm

    tempi_C = np.loadtxt ("tempi_C.txt")
    V_C = np.loadtxt ("V_C.txt")