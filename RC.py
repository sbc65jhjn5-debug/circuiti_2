import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

# Andamento tensione ai capi di C:
def V_C_fit (t, t_0, V_0, RC, offset):
    return V_0 * (1 - np.exp (- (t - t_0) / RC)) + offset

def V_C_fit2 (t, t_0, V_0, RC, offset, A, omega, phi, B, eta):
    esponenziale = V_0 * (1 - np.exp (- (t - t_0) / RC))
    sinusoide = A * np.sin (2 * np.pi * omega * t + phi) + B * np.cos (2 * np.pi * omega * t + eta)
    return esponenziale + sinusoide + offset# Andamento tensione ai capi di R: 

if __name__ == "__main__":

    C_misurata = 10e-9 # F
    R_misurata = 4.673e3 # Ohm
    RC_misurata = R_misurata * C_misurata

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
    m = Minuit (ls, t_0 = 0, V_0 = 5, RC = RC_misurata, offset = -4)
    m.migrad ()
    p_value_C = chi2.sf(m.fval, m.ndof)
    V_0_fit = m.values["V_0"]
    offset_fit = m.values["offset"]
    t_0_fit = m.values["t_0"]
    RC_fit = m.values["RC"]
    print (f"Chi2: {m.fval}\nndof: {m.ndof}\nchi2/ndof: {m.fval / m.ndof}\np-value: {p_value_C}")
    print (f"Offset temporale: {t_0_fit} s")
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

    # residui normalizzati del fit: (CONTROLLARE NON VANNO ANCORA BENE!!!)
    residui_normalizzati = (V_C - V_C_fit (tempo_C, t_0_fit, V_0_fit, RC_fit, offset_fit)) / sigma_V_C
    fig, ax = plt.subplots ()
    ax.errorbar (tempo_C, residui_normalizzati, yerr = np.ones_like(residui_normalizzati), fmt = "^", color = "darkslateblue", capsize = 4)
    ax.axhline (0, color = "red", linestyle = "--")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Residui normalizzati")
    ax.set_title ("Residui normalizzati del fit della tensione ai capi di C")
    plt.show ()




    # COONSIDERANDO ANCHE SINUSOIDE DELLA CORRENTE:

    ls2 = LeastSquares (tempo_C, V_C, sigma_V_C, V_C_fit2)
    m2 = Minuit (ls2, t_0 = 0, V_0 = 5, RC = RC_misurata, offset = -4, A = 7, omega = 70, phi = 0)
    #m2.limits["omega"] = (47, 53)
    m2.migrad ()
    p_value_C2 = chi2.sf(m2.fval, m2.ndof)
    V_0_fit2 = m2.values["V_0"]
    offset_fit2 = m2.values["offset"]
    t_0_fit2 = m2.values["t_0"]
    RC_fit2 = m2.values["RC"]
    A_fit2 = m2.values["A"]
    omega_fit2 = m2.values["omega"]
    phi_fit2 = m2.values["phi"]
    print (f"Chi2: {m2.fval}\nndof: {m2.ndof}\nchi2/ndof: {m2.fval / m2.ndof}\np-value: {p_value_C2}")
    print (f"Offset temporale: {t_0_fit2} s")
    print (f"V_0: {V_0_fit2} V")
    print (f"A: {A_fit2} V")
    print (f"omega: {omega_fit2} Hz")    
    print (f"phi: {phi_fit2} rad")

    # Grafico della tensione ai capi di C con la sinusoide:
    fig, ax = plt.subplots ()
    x_axis = np.linspace (min (tempo_C), max (tempo_C), 5000)
    ax.errorbar (tempo_C, V_C, yerr = sigma_V_C, fmt = "o", label = "Tensione ai capi di C", capsize = 4, color = "darkslateblue")
    ax.plot (x_axis, [V_C_fit2 (t, t_0_fit2, V_0_fit2, RC_fit2, offset_fit2, A_fit2, omega_fit2, phi_fit2) for t in x_axis], color = "darkorchid",label = "Fit tensione ai capi di C con sinusoide")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Tensione (V)")
    ax.set_title ("Tensione ai capi di R e C in un circuito RC con sinusoide")
    ax.legend ()
    plt.show ()

    # residui normalizzati del fit con sinusoide:
    residui_normalizzati2 = (V_C - V_C_fit2 (tempo_C, t_0_fit2, V_0_fit2, RC_fit2, offset_fit2, A_fit2, omega_fit2, phi_fit2)) / sigma_V_C
    fig, ax = plt.subplots ()
    ax.errorbar (tempo_C, residui_normalizzati2, yerr = np.ones_like(residui_normalizzati2), fmt = "^", color = "darkslateblue", capsize = 4)
    ax.axhline (0, color = "red", linestyle = "--")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Residui normalizzati")
    ax.set_title ("Residui normalizzati del fit della tensione ai capi di C con sinusoide")
    plt.show ()













































