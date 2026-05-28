import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

def fit_A (t, A, B, C, gamma, offset, f, phi):
    critico = (A + B * t) * np.exp (- gamma * t) + offset
    sinusoide = C * np.sin (2 * np.pi * f * t + phi)
    return critico + sinusoide

def fit_B (t, A, gamma, f, phi, offset, B, f_sin, phi_sin):
    sottos = A * np.exp (- gamma * t) * np.cos (2 * np.pi * f * t + phi)
    sinusoide = B * np.sin (2 * np.pi * f_sin * t + phi_sin)
    return sottos + offset + sinusoide

def fit_C (t, t_0, A, B, gamma_lento, gamma_veloce, offset, D, f_sin, phi_sin):
    t_shift = t - t_0
    sovras = A * np.exp (- gamma_lento * t_shift) + B * np.exp (- gamma_veloce * t_shift)
    sinusoide = D * np.sin (2 * np.pi * f_sin * t + phi_sin)
    return sovras + offset + sinusoide


if __name__ == "__main__":

    L_stima = 42.6e-3 # Henry
    C = 10e-9 # F
    delta_V = 0.04 # V
    sigma_V = (2 * delta_V) / np.sqrt (12) # V, errore standard di una misura con risoluzione delta_V

    # CONFIGURAZIONE A: smorzamento critico

    R_A = 1210 # Ohm
    gamma_misurato = R_A / (2 * L_stima)
    gamma_critico = 1 / np.sqrt (L_stima * C)
    R_critica = 2 * np.sqrt (L_stima / C)

    tempi_A = np.loadtxt ("tempi_A.txt") * 1e-6
    V_A = np.loadtxt ("V_A.txt")

    # fit A
    lsA = LeastSquares (tempi_A, V_A, sigma_V, fit_A)
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
    ax.plot (t_fit, V_fit, label = "Oscillatore criticamente smorzato con sinusoide", color = "dodgerblue")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Tensione (V)")
    ax.set_title ("Configurazione di smorzamento critico")
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
    ax.set_title ("Residui normalizzati per oscillatore criticamente smorzato")
    ax.legend ()
    ax.grid (True)
    plt.show ()

    # CONFIGURAZIONE B: sottosmorzato

    R_B = 200 # ohm

    tempi_B = np.loadtxt("tempi_B.txt") * 1e-6
    V_B = np.loadtxt ("V_B.txt")

    Delta_B =  np.ones (len (tempi_B)) * 0.008
    sigma_B = 2 * Delta_B / np.sqrt (12)

    # fit B
    lsB = LeastSquares (tempi_B, V_B, sigma_B, fit_B)
    mB = Minuit (lsB, A = 3, gamma = 2.8e4, f = 2.3e4, phi = -1.6, offset = 0, B = 0.008, f_sin = 3.0e4, phi_sin = 0)
    mB.limits["gamma"] = (0, None)
    mB.limits["B"] = (-0.1, 0.1)
    mB.limits["f_sin"] = (0, None)


    mB.migrad ()

    p_value_B = chi2.sf (mB.fval, mB.ndof)

    for par, val, err in zip (mB.parameters, mB.values, mB.errors):
        print (f"{par} = {val:.3e} +/- {err:.3e}")
    print (f"Chi2 B: {mB.fval:.3f}")
    print (f"ndof B: {mB.ndof}")
    print (f"chi2/ndof B: {mB.fval / mB.ndof:.3f}")
    print (f"p-value B: {p_value_B:.3f}")
    print (f"gamma da R_B: {R_B / (2 * L_stima):.3e} 1/s")

    # grafico B
    fig, ax = plt.subplots ()

    ax.errorbar (tempi_B, V_B, yerr = Delta_B, marker = "o", linestyle = "none", label = "Dati B", color = "darkslateblue", capsize = 4)
    t_fit_B = np.linspace (tempi_B[0], tempi_B[-1], 1000)
    V_fit_B = fit_B (t_fit_B, *mB.values)
    ax.plot (t_fit_B, V_fit_B, label = "Oscillatore sottosmorzato", color = "hotpink")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Tensione (V)")
    ax.set_title ("Configurazione di sottosmorzamento")
    ax.legend ()
    ax.grid (True)
    plt.show ()

    # residui normalizzati B
    residui_B = (V_B - fit_B (tempi_B, *mB.values)) / sigma_B
    fig, ax = plt.subplots ()
    ax.errorbar (tempi_B, residui_B, yerr = np.ones_like (residui_B), marker = "^", linestyle = "none", label = "Residui B", color = "darkslateblue", capsize = 4)
    ax.axhline (0, color = "red", linestyle = "--")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Residui normalizzati")
    ax.set_title ("Residui normalizzati per oscillatore sottosmorzato")
    ax.legend ()
    ax.grid (True)
    plt.show ()

    # CONFIGURAZIONE C: sovrasmorzato

    R_C = 5000 # ohm

    tempi_C = np.loadtxt ("tempi_C.txt") * 1e-6
    V_C = np.loadtxt ("V_C.txt")
    Delta_C = np.ones (len (tempi_C)) * 0.012
    sigma_C = 2 * Delta_C / np.sqrt (12)
    gamma_C = R_C / (2 * L_stima)
    omega_0 = 1 / np.sqrt (L_stima * C)
    gamma_lento_teorico = gamma_C - np.sqrt (gamma_C ** 2 - omega_0 ** 2)
    gamma_veloce_teorico = gamma_C + np.sqrt (gamma_C ** 2 - omega_0 ** 2)

    # fit C
    lsC = LeastSquares (tempi_C, V_C, sigma_C, fit_C)
    mC = Minuit (lsC, t_0 = 9.4e-6, A = 8.2, B = 0.2, gamma_lento = 2.0e4, gamma_veloce = 3.4e4, offset = 0, D = 0.012, f_sin = 2.2e4, phi_sin = 6.3)
    mC.limits["t_0"] = (tempi_C[0] - 5e-6, tempi_C[0] + 5e-6)
    mC.limits["gamma_lento"] = (0, None)
    mC.limits["gamma_veloce"] = (0, None)
    mC.limits["D"] = (-0.1, 0.1)
    mC.limits["f_sin"] = (0, None)

    mC.migrad (ncall = 10000)

    p_value_C = chi2.sf (mC.fval, mC.ndof)

    for par, val, err in zip (mC.parameters, mC.values, mC.errors):
        print (f"{par} = {val:.3e} +/- {err:.3e}")
    print (f"Chi2 C: {mC.fval:.3f}")
    print (f"ndof C: {mC.ndof}")
    print (f"chi2/ndof C: {mC.fval / mC.ndof:.3f}")
    print (f"p-value C: {p_value_C:.3f}")
    print (f"gamma lento teorico: {gamma_lento_teorico:.3e} 1/s")
    print (f"gamma veloce teorico: {gamma_veloce_teorico:.3e} 1/s")

    # grafico C
    fig, ax = plt.subplots ()

    ax.errorbar (tempi_C, V_C, yerr = Delta_C, marker = "o", linestyle = "none", label = "Dati C", color = "darkslateblue", capsize = 4)
    t_fit_C = np.linspace (tempi_C[0], tempi_C[-1], 1000)
    V_fit_C = fit_C (t_fit_C, *mC.values)
    ax.plot (t_fit_C, V_fit_C, label = "Oscillatore sovrasmorzato", color = "hotpink")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Tensione (V)")
    ax.set_title ("Configurazione di sovrasmorzamento")
    ax.legend ()
    ax.grid (True)
    plt.show ()

    # residui normalizzati C
    residui_C = (V_C - fit_C (tempi_C, *mC.values)) / sigma_C
    fig, ax = plt.subplots ()
    ax.errorbar (tempi_C, residui_C, yerr = np.ones_like (residui_C), marker = "^", linestyle = "none", label = "Residui C", color = "darkslateblue", capsize = 4)
    ax.axhline (0, color = "red", linestyle = "--")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Residui normalizzati")
    ax.set_title ("Residui normalizzati per oscillatore sovrasmorzato")
    ax.legend ()
    ax.grid (True)
    plt.show ()
