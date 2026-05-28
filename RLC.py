import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

def fit_A (t, A, B, C, gamma, offset, f, phi):
    critico = (A + B * t) * np.exp (- gamma * t) + offset
    sinusoide = C * np.sin (2 * np.pi * f * t + phi)
    return critico + sinusoide

def fit_B (t, A, gamma, f, phi):
    return A * np.exp (- gamma * t) * np.cos (2 * np.pi * f * t + phi)

if __name__ == "__main__":

    L_stima = 42.6e-3 # Henry
    C = 10e-9 # F
    delta_V = 0.04 # V

    # CONFIGURAZIONE A: smorzamento critico

    R_A = 1210 # Ohm
    gamma_misurato = R_A / (2 * L_stima)
    gamma_critico = 1 / np.sqrt (L_stima * C)
    R_critica = 2 * np.sqrt (L_stima / C)

    tempi_A = np.loadtxt ("tempi_A.txt") * 1e-6
    V_A = np.loadtxt ("V_A.txt")

    # fit A
    lsA = LeastSquares (tempi_A, V_A, delta_V, fit_A)
    mA = Minuit (lsA, A = -0.2, B = 2.5e6, C = 0.04, gamma = gamma_critico, offset = 0.0, f = 3.3e4, phi = 0)
    mA.limits["gamma"] = (0, None)
    mA.limits["C"] = (-0.1, 0.1)
    mA.fixed["f"] = True

    mA.migrad ()
    p_value_A = chi2.sf (mA.fval, mA.ndof)

    for par, val, err in zip (mA.parameters, mA.values, mA.errors):
        print (f"{par} = {val:.3e} +/- {err:.3e}")
    print (f"Chi2 A: {mA.fval:.3f}")
    print (f"ndof A: {mA.ndof}")
    print (f"chi2/ndof A: {mA.fval / mA.ndof:.3f}")
    print (f"p-value A: {p_value_A:.3f}")
    print (f"gamma critico teorico: {gamma_critico:.3e} 1/s")
    print (f"gamma da R_A: {gamma_misurato:.3e} 1/s")
    print (f"R critica teorica: {R_critica:.3e} Ohm")

    # plot A
    fig, ax = plt.subplots ()

    ax.errorbar (tempi_A, V_A, yerr = delta_V, marker = "o", linestyle = "none", label = "Dati A", color = "darkslateblue", capsize = 4)
    t_fit = np.linspace (tempi_A[0], tempi_A[-1], 1000)
    V_fit = fit_A (t_fit, *mA.values)
    ax.plot (t_fit, V_fit, label = "Oscillatore criticamente smorzato con sinusoide", color = "orangered")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Tensione (V)")
    ax.set_title ("Configurazione A: smorzamento critico")
    ax.legend ()
    ax.grid (True)
    plt.show ()

    # residui normalizzati A
    residui_A = (V_A - fit_A (tempi_A, *mA.values)) / delta_V
    fig, ax = plt.subplots ()
    ax.errorbar (tempi_A, residui_A, yerr = np.ones_like (residui_A), marker = "^", linestyle = "none", label = "Residui A", color = "darkslateblue", capsize = 4)
    ax.axhline (0, color = "red", linestyle = "--")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Residui normalizzati")
    ax.set_title ("Residui normalizzati configurazione A")
    ax.legend ()
    ax.grid (True)
    plt.show ()

    # CONFIGURAZIONE B: sottosmorzato

    R_B = 200 # ohm

    tempi_B = np.loadtxt("tempi_B.txt")
    V_B = np.loadtxt ("V_B.txt")

    Delta_B =  np.ones (len (tempi_B)) * 0.008

    # CONFIGURAZIONE C: sovrasmorzato

    R_C = 5000 # ohm

    tempi_C = np.loadtxt ("tempi_C.txt")
    V_C = np.loadtxt ("V_C.txt")
