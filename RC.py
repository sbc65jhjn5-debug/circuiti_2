import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2


# Andamento tensione ai capi di C:
def V_C_fit (t, t_0, V_0, RC, offset):
    return V_0 * (1 - np.exp (- (t - t_0) / RC)) + offset


def V_C_fit2 (t, t_0, V_0, RC, offset, A, omega, phi):
    esponenziale = V_0 * (1 - np.exp (- (t - t_0) / RC))
    sinusoide = A * np.sin (2 * np.pi * omega * t + phi)
    return esponenziale + sinusoide + offset


def V_C_fit2_mask_sin (t, t_0, V_0, RC, offset, A, omega, phi, mask_sin):
    esponenziale = V_0 * (1 - np.exp (- (t - t_0) / RC))
    sinusoide = A * np.sin (2 * np.pi * omega * t + phi) * mask_sin
    return esponenziale + sinusoide + offset


def V_C_fit3_mask_sin_cos (t, t_0, V_0, RC, offset, A, omega_sin, phi, B, omega_cos, eta, mask_sin):
    esponenziale = V_0 * (1 - np.exp (- (t - t_0) / RC))
    sinusoide = A * np.sin (2 * np.pi * omega_sin * t + phi)
    cosinusoide = B * np.cos (2 * np.pi * omega_cos * t + eta)
    return esponenziale + (sinusoide + cosinusoide) * mask_sin + offset


# Andamento tensione ai capi di R:
def V_R_fit (t, t_0, V_0, RC, offset):
    return V_0 * np.exp (- (t - t_0) / RC) + offset


def V_R_fit2 (t, t_0, V_0, RC, offset, A, omega, phi):
    esponenziale = V_0 * np.exp (- (t - t_0) / RC)
    sinusoide = A * np.sin (2 * np.pi * omega * t + phi)
    return esponenziale + sinusoide + offset


def V_R_fit2_mask_sin (t, t_0, V_0, RC, offset, A, omega, phi, mask_sin):
    esponenziale = V_0 * np.exp (- (t - t_0) / RC)
    sinusoide = A * np.sin (2 * np.pi * omega * t + phi) * mask_sin
    return esponenziale + sinusoide + offset


def V_R_fit3_mask_sin_cos (t, t_0, V_0, RC, offset, A, omega_sin, phi, B, omega_cos, eta, mask_sin):
    esponenziale = V_0 * np.exp (- (t - t_0) / RC)
    sinusoide = A * np.sin (2 * np.pi * omega_sin * t + phi)
    cosinusoide = B * np.cos (2 * np.pi * omega_cos * t + eta)
    return esponenziale + (sinusoide + cosinusoide) * mask_sin + offset


if __name__ == "__main__":

    C_misurata = 10e-9 # F
    R_misurata = 4.673e3 # Ohm
    RC_misurata = R_misurata * C_misurata

    # Tensione misurata ai capi di C:
    tempo_C = np.loadtxt ("RC_tempi_C.txt") * 1e-6
    V_C = np.loadtxt ("RC_V_C.txt")
    Delta_V_C = np.ones (len (V_C)) * 0.04
    sigma_V_C = (2 * Delta_V_C) / np.sqrt (12)

    # Tensione misurata ai capi di R:
    tempo_R = np.loadtxt ("RC_tempi_R.txt") * 1e-6
    V_R = np.loadtxt ("RC_V_R.txt")
    Delta_V_R = np.ones (len (V_R)) * 0.1 # V
    sigma_V_R = (2 * Delta_V_R) / np.sqrt (12)


    # TENSIONE AI CAPI DI C

    # Fit della tensione ai capi di C:
    ls = LeastSquares (tempo_C, V_C, sigma_V_C, V_C_fit)
    m = Minuit (ls, t_0 = 0, V_0 = 5, RC = RC_misurata, offset = -4)
    m.migrad ()
    p_value_C = chi2.sf (m.fval, m.ndof)
    V_0_fit = m.values["V_0"]
    offset_fit = m.values["offset"]
    t_0_fit = m.values["t_0"]
    RC_fit = m.values["RC"]
    print (f"Chi2 C: {m.fval}\nndof C: {m.ndof}\nchi2/ndof C: {m.fval / m.ndof}\np-value C: {p_value_C}")
    print (f"Offset temporale C: {t_0_fit} s")
    print (f"V_0 C: {V_0_fit} V")

    # Grafico della tensione ai capi di C:
    fig, ax = plt.subplots ()
    x_axis = np.linspace (min (tempo_C), max (tempo_C), 5000)
    ax.errorbar (tempo_C, V_C, yerr = sigma_V_C, fmt = "o", label = "Tensione ai capi di C", capsize = 4, color = "darkslateblue")
    ax.plot (x_axis, V_C_fit (x_axis, t_0_fit, V_0_fit, RC_fit, offset_fit), color = "darkorchid", label = "Fit tensione ai capi di C")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Tensione (V)")
    ax.set_title ("Tensione ai capi di C in un circuito RC")
    ax.grid (True, alpha = 0.4)
    ax.legend ()
    plt.show ()

    # Residui normalizzati del fit:
    residui_normalizzati = (V_C - V_C_fit (tempo_C, t_0_fit, V_0_fit, RC_fit, offset_fit)) / sigma_V_C
    fig, ax = plt.subplots ()
    ax.errorbar (tempo_C, residui_normalizzati, yerr = np.ones_like (residui_normalizzati), fmt = "^", color = "darkorchid", capsize = 4)
    ax.axhline (0, color = "red", linestyle = "--")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Residui normalizzati")
    ax.set_title ("Residui normalizzati del fit della tensione ai capi di C")
    ax.grid (True, alpha = 0.4)
    plt.show ()


    # CONSIDERANDO ANCHE SINUSOIDE DELLA CORRENTE:

    mask_sin_fit2 = np.ones_like (tempo_C, dtype = bool)
    mask_sin_fit2[-9:] = False

    def V_C_fit2_con_mask_sin (t, t_0, V_0, RC, offset, A, omega, phi):
        return V_C_fit2_mask_sin (t, t_0, V_0, RC, offset, A, omega, phi, mask_sin_fit2)

    ls2 = LeastSquares (tempo_C, V_C, sigma_V_C, V_C_fit2_con_mask_sin)
    m2 = Minuit (ls2, t_0 = 0, V_0 = 5, RC = RC_misurata, offset = -4, A = 7, omega = 70, phi = 0)
    m2.migrad ()

    p_value_C2 = chi2.sf (m2.fval, m2.ndof)
    V_0_fit2 = m2.values["V_0"]
    offset_fit2 = m2.values["offset"]
    t_0_fit2 = m2.values["t_0"]
    RC_fit2 = m2.values["RC"]
    A_fit2 = m2.values["A"]
    omega_fit2 = m2.values["omega"]
    phi_fit2 = m2.values["phi"]

    print (f"Chi2 C con sinusoide: {m2.fval}\nndof C con sinusoide: {m2.ndof}\nchi2/ndof C con sinusoide: {m2.fval / m2.ndof}\np-value C con sinusoide: {p_value_C2}")
    print (f"Offset temporale C con sinusoide: {t_0_fit2} s")
    print (f"V_0 C con sinusoide: {V_0_fit2} V")
    print (f"A C con sinusoide: {A_fit2} V")
    print (f"omega C con sinusoide: {omega_fit2} Hz")
    print (f"phi C con sinusoide: {phi_fit2} rad")

    # Grafico della tensione ai capi di C con la sinusoide:
    fig, ax = plt.subplots ()
    x_axis = np.linspace (min (tempo_C), max (tempo_C), 5000)
    x_axis_mask_sin = x_axis < tempo_C[-9]
    ax.errorbar (tempo_C, V_C, yerr = sigma_V_C, fmt = "o", label = "Tensione ai capi di C", capsize = 4, color = "darkslateblue")
    ax.plot (x_axis, V_C_fit2_mask_sin (x_axis, t_0_fit2, V_0_fit2, RC_fit2, offset_fit2, A_fit2, omega_fit2, phi_fit2, x_axis_mask_sin), color = "rebeccapurple", label = "Fit tensione ai capi di C con sinusoide")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Tensione (V)")
    ax.set_title ("Tensione ai capi di C in un circuito RC con sinusoide")
    ax.grid (True, alpha = 0.4)
    ax.legend ()
    plt.show ()

    # Residui normalizzati del fit con sinusoide:
    residui_normalizzati2 = (V_C - V_C_fit2_con_mask_sin (tempo_C, t_0_fit2, V_0_fit2, RC_fit2, offset_fit2, A_fit2, omega_fit2, phi_fit2)) / sigma_V_C
    fig, ax = plt.subplots ()
    ax.errorbar (tempo_C, residui_normalizzati2, yerr = np.ones_like (residui_normalizzati2), fmt = "^", color = "rebeccapurple", capsize = 4)
    ax.axhline (0, color = "red", linestyle = "--")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Residui normalizzati")
    ax.set_title ("Residui normalizzati del fit della tensione ai capi di C con sinusoide")
    ax.grid (True, alpha = 0.4)
    plt.show ()


    # CONSIDERANDO SINUSOIDE E COSINUSOIDE CON FREQUENZE DIVERSE:

    def V_C_fit3_con_mask_sin_cos (t, t_0, V_0, RC, offset, A, omega_sin, phi, B, omega_cos, eta):
        return V_C_fit3_mask_sin_cos (t, t_0, V_0, RC, offset, A, omega_sin, phi, B, omega_cos, eta, mask_sin_fit2)

    ls3 = LeastSquares (tempo_C, V_C, sigma_V_C, V_C_fit3_con_mask_sin_cos)
    m3 = Minuit (
        ls3,
        t_0 = t_0_fit2,
        V_0 = V_0_fit2,
        RC = RC_fit2,
        offset = offset_fit2,
        A = A_fit2,
        omega_sin = omega_fit2,
        phi = phi_fit2,
        B = 0.01,
        omega_cos = 30500,
        eta = 0
    )
    m3.limits["omega_sin"] = (0, None)
    m3.limits["omega_cos"] = (0, None)
    m3.migrad ()

    p_value_C3 = chi2.sf (m3.fval, m3.ndof)
    V_0_fit3 = m3.values["V_0"]
    offset_fit3 = m3.values["offset"]
    t_0_fit3 = m3.values["t_0"]
    RC_fit3 = m3.values["RC"]
    A_fit3 = m3.values["A"]
    omega_sin_fit3 = m3.values["omega_sin"]
    phi_fit3 = m3.values["phi"]
    B_fit3 = m3.values["B"]
    omega_cos_fit3 = m3.values["omega_cos"]
    eta_fit3 = m3.values["eta"]

    print (f"Chi2 fit3 C: {m3.fval}\nndof fit3 C: {m3.ndof}\nchi2/ndof fit3 C: {m3.fval / m3.ndof}\np-value fit3 C: {p_value_C3}")
    print (f"Offset temporale fit3 C: {t_0_fit3} s")
    print (f"V_0 fit3 C: {V_0_fit3} V")
    print (f"A fit3 C: {A_fit3} V")
    print (f"omega_sin fit3 C: {omega_sin_fit3} Hz")
    print (f"phi fit3 C: {phi_fit3} rad")
    print (f"B fit3 C: {B_fit3} V")
    print (f"omega_cos fit3 C: {omega_cos_fit3} Hz")
    print (f"eta fit3 C: {eta_fit3} rad")

    # Grafico della tensione ai capi di C con sinusoide e cosinusoide:
    fig, ax = plt.subplots ()
    x_axis = np.linspace (min (tempo_C), max (tempo_C), 5000)
    x_axis_mask_sin = x_axis < tempo_C[-9]
    ax.errorbar (tempo_C, V_C, yerr = sigma_V_C, fmt = "o", label = "Tensione ai capi di C", capsize = 4, color = "darkslateblue")
    ax.plot (x_axis, V_C_fit3_mask_sin_cos (x_axis, t_0_fit3, V_0_fit3, RC_fit3, offset_fit3, A_fit3, omega_sin_fit3, phi_fit3, B_fit3, omega_cos_fit3, eta_fit3, x_axis_mask_sin), color = "crimson", label = "Fit tensione ai capi di C con sinusoide e cosinusoide")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Tensione (V)")
    ax.set_title ("Tensione ai capi di C in un circuito RC con sinusoide e cosinusoide")
    ax.grid (True, alpha = 0.4)
    ax.legend ()
    plt.show ()

    # Residui normalizzati del fit con sinusoide e cosinusoide:
    residui_normalizzati3 = (V_C - V_C_fit3_con_mask_sin_cos (tempo_C, t_0_fit3, V_0_fit3, RC_fit3, offset_fit3, A_fit3, omega_sin_fit3, phi_fit3, B_fit3, omega_cos_fit3, eta_fit3)) / sigma_V_C
    fig, ax = plt.subplots ()
    ax.errorbar (tempo_C, residui_normalizzati3, yerr = np.ones_like (residui_normalizzati3), fmt = "^", color = "crimson", capsize = 4)
    ax.axhline (0, color = "red", linestyle = "--")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Residui normalizzati")
    ax.set_title ("Residui normalizzati del fit della tensione ai capi di C con sinusoide e cosinusoide")
    ax.grid (True, alpha = 0.4)
    plt.show ()

    # Grafico finale con tutti i fit sovrapposti:
    fig, ax = plt.subplots ()
    x_axis = np.linspace (min (tempo_C), max (tempo_C), 5000)
    x_axis_mask_sin = x_axis < tempo_C[-9]
    ax.errorbar (tempo_C, V_C, yerr = sigma_V_C, fmt = "o", label = "Tensione ai capi di C", capsize = 4, color = "darkslateblue")
    ax.plot (x_axis, V_C_fit (x_axis, t_0_fit, V_0_fit, RC_fit, offset_fit), color = "darkorchid", label = "Fit esponenziale")
    ax.plot (x_axis, V_C_fit2_mask_sin (x_axis, t_0_fit2, V_0_fit2, RC_fit2, offset_fit2, A_fit2, omega_fit2, phi_fit2, x_axis_mask_sin), color = "dodgerblue", label = "Fit esponenziale + sinusoide")
    ax.plot (x_axis, V_C_fit3_mask_sin_cos (x_axis, t_0_fit3, V_0_fit3, RC_fit3, offset_fit3, A_fit3, omega_sin_fit3, phi_fit3, B_fit3, omega_cos_fit3, eta_fit3, x_axis_mask_sin), color = "crimson", label = "Fit esponenziale + sinusoide + cosinusoide")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Tensione (V)")
    ax.set_title ("Confronto tra i fit della tensione ai capi di C")
    ax.grid (True, alpha = 0.4)
    ax.legend ()
    plt.show ()

    # Grafico finale con tutti i residui sovrapposti:
    fig, ax = plt.subplots ()
    ax.errorbar (tempo_C, residui_normalizzati, yerr = np.ones_like (residui_normalizzati), fmt = "o", color = "darkorchid", capsize = 4, label = "Residui fit esponenziale")
    ax.errorbar (tempo_C, residui_normalizzati2, yerr = np.ones_like (residui_normalizzati2), fmt = "^", color = "dodgerblue", capsize = 4, label = "Residui fit esponenziale + sinusoide")
    ax.errorbar (tempo_C, residui_normalizzati3, yerr = np.ones_like (residui_normalizzati3), fmt = "s", color = "crimson", capsize = 4, label = "Residui fit esponenziale + sinusoide + cosinusoide")
    ax.axhline (0, color = "red", linestyle = "--")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Residui normalizzati")
    ax.set_title ("Confronto tra i residui normalizzati dei fit della tensione ai capi di C")
    ax.grid (True, alpha = 0.4)
    ax.legend ()
    plt.show ()


    # TENSIONE AI CAPI DI R

    # Fit della tensione ai capi di R:
    ls_R = LeastSquares (tempo_R, V_R, sigma_V_R, V_R_fit)
    m_R = Minuit (ls_R, t_0 = 0, V_0 = 8, RC = RC_misurata, offset = 0)
    m_R.migrad ()
    p_value_R = chi2.sf (m_R.fval, m_R.ndof)
    V_0_fit_R = m_R.values["V_0"]
    offset_fit_R = m_R.values["offset"]
    t_0_fit_R = m_R.values["t_0"]
    RC_fit_R = m_R.values["RC"]
    print (f"Chi2 R: {m_R.fval}\nndof R: {m_R.ndof}\nchi2/ndof R: {m_R.fval / m_R.ndof}\np-value R: {p_value_R}")
    print (f"Offset temporale R: {t_0_fit_R} s")
    print (f"V_0 R: {V_0_fit_R} V")

    # Grafico della tensione ai capi di R:
    fig, ax = plt.subplots ()
    x_axis_R = np.linspace (min (tempo_R), max (tempo_R), 5000)
    ax.errorbar (tempo_R, V_R, yerr = sigma_V_R, fmt = "o", label = "Tensione ai capi di R", capsize = 4, color = "indigo")
    ax.plot (x_axis_R, V_R_fit (x_axis_R, t_0_fit_R, V_0_fit_R, RC_fit_R, offset_fit_R), color = "orangered", label = "Fit tensione ai capi di R")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Tensione (V)")
    ax.set_title ("Tensione ai capi di R in un circuito RC")
    ax.grid (True, alpha = 0.4)
    ax.legend ()
    plt.show ()

    # Residui normalizzati del fit della tensione ai capi di R:
    residui_normalizzati_R = (V_R - V_R_fit (tempo_R, t_0_fit_R, V_0_fit_R, RC_fit_R, offset_fit_R)) / sigma_V_R
    fig, ax = plt.subplots ()
    ax.errorbar (tempo_R, residui_normalizzati_R, yerr = np.ones_like (residui_normalizzati_R), fmt = "^", color = "orangered", capsize = 4)
    ax.axhline (0, color = "red", linestyle = "--")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Residui normalizzati")
    ax.set_title ("Residui normalizzati del fit della tensione ai capi di R")
    ax.grid (True, alpha = 0.4)
    plt.show ()


    '''
    # CONSIDERANDO ANCHE SINUSOIDE DELLA CORRENTE AI CAPI DI R:

    mask_sin_fit2_R = np.ones_like (tempo_R, dtype = bool)
    mask_sin_fit2_R[-9:] = False

    def V_R_fit2_con_mask_sin (t, t_0, V_0, RC, offset, A, omega, phi):
        return V_R_fit2_mask_sin (t, t_0, V_0, RC, offset, A, omega, phi, mask_sin_fit2_R)

    ls2_R = LeastSquares (tempo_R, V_R, sigma_V_R, V_R_fit2_con_mask_sin)
    m2_R = Minuit (ls2_R, t_0 = 0, V_0 = 8, RC = RC_misurata, offset = 0, A = 1, omega = 70, phi = 0)
    m2_R.migrad ()

    p_value_R2 = chi2.sf (m2_R.fval, m2_R.ndof)
    V_0_fit2_R = m2_R.values["V_0"]
    offset_fit2_R = m2_R.values["offset"]
    t_0_fit2_R = m2_R.values["t_0"]
    RC_fit2_R = m2_R.values["RC"]
    A_fit2_R = m2_R.values["A"]
    omega_fit2_R = m2_R.values["omega"]
    phi_fit2_R = m2_R.values["phi"]

    print (f"Chi2 R con sinusoide: {m2_R.fval}\nndof R con sinusoide: {m2_R.ndof}\nchi2/ndof R con sinusoide: {m2_R.fval / m2_R.ndof}\np-value R con sinusoide: {p_value_R2}")
    print (f"Offset temporale R con sinusoide: {t_0_fit2_R} s")
    print (f"V_0 R con sinusoide: {V_0_fit2_R} V")
    print (f"A R con sinusoide: {A_fit2_R} V")
    print (f"omega R con sinusoide: {omega_fit2_R} Hz")
    print (f"phi R con sinusoide: {phi_fit2_R} rad")

    # Grafico della tensione ai capi di R con la sinusoide:
    fig, ax = plt.subplots ()
    x_axis_R = np.linspace (min (tempo_R), max (tempo_R), 5000)
    x_axis_mask_sin_R = x_axis_R < tempo_R[-9]
    ax.errorbar (tempo_R, V_R, yerr = sigma_V_R, fmt = "o", label = "Tensione ai capi di R", capsize = 4, color = "seagreen")
    ax.plot (x_axis_R, V_R_fit2_mask_sin (x_axis_R, t_0_fit2_R, V_0_fit2_R, RC_fit2_R, offset_fit2_R, A_fit2_R, omega_fit2_R, phi_fit2_R, x_axis_mask_sin_R), color = "orange", label = "Fit tensione ai capi di R con sinusoide")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Tensione (V)")
    ax.set_title ("Tensione ai capi di R in un circuito RC con sinusoide")
    ax.grid (True, alpha = 0.4)
    ax.legend ()
    plt.show ()

    # Residui normalizzati del fit con sinusoide ai capi di R:
    residui_normalizzati2_R = (V_R - V_R_fit2_con_mask_sin (tempo_R, t_0_fit2_R, V_0_fit2_R, RC_fit2_R, offset_fit2_R, A_fit2_R, omega_fit2_R, phi_fit2_R)) / sigma_V_R
    fig, ax = plt.subplots ()
    ax.errorbar (tempo_R, residui_normalizzati2_R, yerr = np.ones_like (residui_normalizzati2_R), fmt = "^", color = "orange", capsize = 4)
    ax.axhline (0, color = "red", linestyle = "--")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Residui normalizzati")
    ax.set_title ("Residui normalizzati del fit della tensione ai capi di R con sinusoide")
    ax.grid (True, alpha = 0.4)
    plt.show ()


    # CONSIDERANDO SINUSOIDE E COSINUSOIDE CON FREQUENZE DIVERSE AI CAPI DI R:

    def V_R_fit3_con_mask_sin_cos (t, t_0, V_0, RC, offset, A, omega_sin, phi, B, omega_cos, eta):
        return V_R_fit3_mask_sin_cos (t, t_0, V_0, RC, offset, A, omega_sin, phi, B, omega_cos, eta, mask_sin_fit2_R)

    ls3_R = LeastSquares (tempo_R, V_R, sigma_V_R, V_R_fit3_con_mask_sin_cos)
    m3_R = Minuit (
        ls3_R,
        t_0 = t_0_fit2_R,
        V_0 = V_0_fit2_R,
        RC = RC_fit2_R,
        offset = offset_fit2_R,
        A = A_fit2_R,
        omega_sin = omega_fit2_R,
        phi = phi_fit2_R,
        B = 0.01,
        omega_cos = 30500,
        eta = 0
    )
    m3_R.limits["omega_sin"] = (0, None)
    m3_R.limits["omega_cos"] = (0, None)
    m3_R.migrad ()

    p_value_R3 = chi2.sf (m3_R.fval, m3_R.ndof)
    V_0_fit3_R = m3_R.values["V_0"]
    offset_fit3_R = m3_R.values["offset"]
    t_0_fit3_R = m3_R.values["t_0"]
    RC_fit3_R = m3_R.values["RC"]
    A_fit3_R = m3_R.values["A"]
    omega_sin_fit3_R = m3_R.values["omega_sin"]
    phi_fit3_R = m3_R.values["phi"]
    B_fit3_R = m3_R.values["B"]
    omega_cos_fit3_R = m3_R.values["omega_cos"]
    eta_fit3_R = m3_R.values["eta"]

    print (f"Chi2 fit3 R: {m3_R.fval}\nndof fit3 R: {m3_R.ndof}\nchi2/ndof fit3 R: {m3_R.fval / m3_R.ndof}\np-value fit3 R: {p_value_R3}")
    print (f"Offset temporale fit3 R: {t_0_fit3_R} s")
    print (f"V_0 fit3 R: {V_0_fit3_R} V")
    print (f"A fit3 R: {A_fit3_R} V")
    print (f"omega_sin fit3 R: {omega_sin_fit3_R} Hz")
    print (f"phi fit3 R: {phi_fit3_R} rad")
    print (f"B fit3 R: {B_fit3_R} V")
    print (f"omega_cos fit3 R: {omega_cos_fit3_R} Hz")
    print (f"eta fit3 R: {eta_fit3_R} rad")

    # Grafico della tensione ai capi di R con sinusoide e cosinusoide:
    fig, ax = plt.subplots ()
    x_axis_R = np.linspace (min (tempo_R), max (tempo_R), 5000)
    x_axis_mask_sin_R = x_axis_R < tempo_R[-9]
    ax.errorbar (tempo_R, V_R, yerr = sigma_V_R, fmt = "o", label = "Tensione ai capi di R", capsize = 4, color = "seagreen")
    ax.plot (x_axis_R, V_R_fit3_mask_sin_cos (x_axis_R, t_0_fit3_R, V_0_fit3_R, RC_fit3_R, offset_fit3_R, A_fit3_R, omega_sin_fit3_R, phi_fit3_R, B_fit3_R, omega_cos_fit3_R, eta_fit3_R, x_axis_mask_sin_R), color = "crimson", label = "Fit tensione ai capi di R con sinusoide e cosinusoide")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Tensione (V)")
    ax.set_title ("Tensione ai capi di R in un circuito RC con sinusoide e cosinusoide")
    ax.grid (True, alpha = 0.4)
    ax.legend ()
    plt.show ()

    # Residui normalizzati del fit con sinusoide e cosinusoide ai capi di R:
    residui_normalizzati3_R = (V_R - V_R_fit3_con_mask_sin_cos (tempo_R, t_0_fit3_R, V_0_fit3_R, RC_fit3_R, offset_fit3_R, A_fit3_R, omega_sin_fit3_R, phi_fit3_R, B_fit3_R, omega_cos_fit3_R, eta_fit3_R)) / sigma_V_R
    fig, ax = plt.subplots ()
    ax.errorbar (tempo_R, residui_normalizzati3_R, yerr = np.ones_like (residui_normalizzati3_R), fmt = "^", color = "crimson", capsize = 4)
    ax.axhline (0, color = "red", linestyle = "--")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Residui normalizzati")
    ax.set_title ("Residui normalizzati del fit della tensione ai capi di R con sinusoide e cosinusoide")
    ax.grid (True, alpha = 0.4)
    plt.show ()

    # Grafico finale con tutti i fit sovrapposti per R:
    fig, ax = plt.subplots ()
    x_axis_R = np.linspace (min (tempo_R), max (tempo_R), 5000)
    x_axis_mask_sin_R = x_axis_R < tempo_R[-9]
    ax.errorbar (tempo_R, V_R, yerr = sigma_V_R, fmt = "o", label = "Tensione ai capi di R", capsize = 4, color = "seagreen")
    ax.plot (x_axis_R, V_R_fit (x_axis_R, t_0_fit_R, V_0_fit_R, RC_fit_R, offset_fit_R), color = "darkgreen", label = "Fit esponenziale")
    ax.plot (x_axis_R, V_R_fit2_mask_sin (x_axis_R, t_0_fit2_R, V_0_fit2_R, RC_fit2_R, offset_fit2_R, A_fit2_R, omega_fit2_R, phi_fit2_R, x_axis_mask_sin_R), color = "orange", label = "Fit esponenziale + sinusoide")
    ax.plot (x_axis_R, V_R_fit3_mask_sin_cos (x_axis_R, t_0_fit3_R, V_0_fit3_R, RC_fit3_R, offset_fit3_R, A_fit3_R, omega_sin_fit3_R, phi_fit3_R, B_fit3_R, omega_cos_fit3_R, eta_fit3_R, x_axis_mask_sin_R), color = "crimson", label = "Fit esponenziale + sinusoide + cosinusoide")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Tensione (V)")
    ax.set_title ("Confronto tra i fit della tensione ai capi di R")
    ax.grid (True, alpha = 0.4)
    ax.legend ()
    plt.show ()

    # Grafico finale con tutti i residui sovrapposti per R:
    fig, ax = plt.subplots ()
    ax.errorbar (tempo_R, residui_normalizzati_R, yerr = np.ones_like (residui_normalizzati_R), fmt = "o", color = "darkgreen", capsize = 4, label = "Residui fit esponenziale")
    ax.errorbar (tempo_R, residui_normalizzati2_R, yerr = np.ones_like (residui_normalizzati2_R), fmt = "^", color = "orange", capsize = 4, label = "Residui fit esponenziale + sinusoide")
    ax.errorbar (tempo_R, residui_normalizzati3_R, yerr = np.ones_like (residui_normalizzati3_R), fmt = "s", color = "crimson", capsize = 4, label = "Residui fit esponenziale + sinusoide + cosinusoide")
    ax.axhline (0, color = "red", linestyle = "--")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Residui normalizzati")
    ax.set_title ("Confronto tra i residui normalizzati dei fit della tensione ai capi di R")
    ax.grid (True, alpha = 0.4)
    ax.legend ()
    plt.show ()
    '''