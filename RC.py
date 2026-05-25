import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

# Andamento tensione ai capi di C:
def V_C_fit (t, t_0, V_0, RC, offset):
    return V_0 * (1 - np.exp (- (t - t_0) / (RC))) + offset

# Andamento tensione ai capi di R: 

if __name__ == "__main__":

    C = 10e-9 # F
    R = 4.673e3 # Ohm
    RC = R * C

    # Tensione misurata ai capi di R

    tempo_R = np.loadtxt ("RC_tempi_R.txt") * 1e-6
    V_R = np.loadtxt ("RC_V_R.txt")

    Delta_V_R = np.ones (len (V_R)) * 0.3 # V
    sigma_V_R = (2 * Delta_V_R) / np.sqrt (12)

    # Tensione misurata ai capi di C

    tempo_C = np.loadtxt ("RC_tempi_C.txt") * 1e-6 
    V_C = np.loadtxt ("RC_V_C.txt")
    Delta_V_C = np.ones (len (V_C)) * 0.04
    sigma_V_C = (2 * Delta_V_C) / np.sqrt (12)


    # Fit della tensione ai capi di C:
    ls = LeastSquares (tempo_C, V_C, sigma_V_C, V_C_fit)
    m = Minuit (ls, t_0 = 0, V_0 = 5, RC = RC, offset = -4)
    m.migrad ()
    p_value_C = chi2.sf(m.fval, m.ndof)
    V_0_fit = m.values["V_0"]
    offset_fit = m.values["offset"]
    t_0_fit = m.values["t_0"]
    RC_fit = m.values["RC"]
    print (f"Chi2: {m.fval}\nndof: {m.ndof}\nchi2/ndof: {m.fval / m.ndof}\np-value: {p_value_C}")
    print (f"Offset temporale: {t_0_fit} s")
    print (f"Costante di tempo ($\\tau$): {RC_fit} s")
    print (f"V_0: {V_0_fit} V")

    # Grafico della tensione ai capi di C:

    fig, ax = plt.subplots ()
    x_axis = np.linspace (min (tempo_C), max (tempo_C), 5000)
    ax.errorbar (tempo_C, V_C, yerr = sigma_V_C, fmt = "o", label = "Tensione ai capi di C", capsize = 4, color = "darkslateblue")
    ax.plot (x_axis, [V_C_fit (t, t_0_fit, V_0_fit, RC_fit, offset_fit) for t in x_axis], color = "darkorchid",label = "Fit tensione ai capi di C")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Tensione (V)")
    ax.set_title ("Tensione ai capi di R e C in un circuito RC")
    ax.legend ()
    plt.show ()

    # residui normalizzati del fit:
    residui_normalizzati = (V_C - V_C_fit (tempo_C, t_0_fit, V_0_fit, RC_fit, offset_fit)) / sigma_V_C
    fig, ax = plt.subplots ()
    ax.errorbar (tempo_C, residui_normalizzati, yerr = np.ones_like(residui_normalizzati), fmt = "^", color = "darkslateblue", capsize = 4)
    ax.axhline (0, color = "red", linestyle = "--")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Residui normalizzati")
    ax.set_title ("Residui normalizzati del fit della tensione ai capi di C")
    plt.show ()
















































