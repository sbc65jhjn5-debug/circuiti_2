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
    return esponenziale + sinusoide + offset# Andamento tensione ai capi di R: 

def V_C_fit2_mask_sin (t, t_0, V_0, RC, offset, A, omega, phi, mask_sin):
    esponenziale = V_0 * (1 - np.exp (- (t - t_0) / RC))
    sinusoide = A * np.sin (2 * np.pi * omega * t + phi) * mask_sin
    return esponenziale + sinusoide + offset

def V_C_fit3_mask_sin_cos (t, t_0, V_0, RC, offset, A, omega_sin, phi, B, omega_cos, eta, mask_sin):
    esponenziale = V_0 * (1 - np.exp (- (t - t_0) / RC))
    sinusoide = A * np.sin (2 * np.pi * omega_sin * t + phi)
    cosinusoide = B * np.cos (2 * np.pi * omega_cos * t + eta)
    return esponenziale + (sinusoide + cosinusoide) * mask_sin + offset

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
    ax.grid (True, alpha = 0.4)
    ax.legend ()
    plt.show ()

    # residui normalizzati del fit: (CONTROLLARE NON VANNO ANCORA BENE!!!)
    residui_normalizzati = (V_C - V_C_fit (tempo_C, t_0_fit, V_0_fit, RC_fit, offset_fit)) / sigma_V_C
    fig, ax = plt.subplots ()
    ax.errorbar (tempo_C, residui_normalizzati, yerr = np.ones_like(residui_normalizzati), fmt = "^", color = "darkorchid", capsize = 4)
    ax.axhline (0, color = "red", linestyle = "--")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Residui normalizzati")
    ax.set_title ("Residui normalizzati del fit della tensione ai capi di C")
    ax.grid (True, alpha = 0.4)
    plt.show ()




    # COONSIDERANDO ANCHE SINUSOIDE DELLA CORRENTE:

    mask_sin_fit2 = np.ones_like (tempo_C, dtype = bool)
    mask_sin_fit2[-9:] = False

    def V_C_fit2_con_mask_sin (t, t_0, V_0, RC, offset, A, omega, phi):
        return V_C_fit2_mask_sin (t, t_0, V_0, RC, offset, A, omega, phi, mask_sin_fit2)

    ls2 = LeastSquares (tempo_C, V_C, sigma_V_C, V_C_fit2_con_mask_sin)
    m2 = Minuit (ls2, t_0 = 0, V_0 = 5, RC = RC_misurata, offset = -4, A = 7, omega = 70, phi = 0)
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
    x_axis_mask_sin = x_axis < tempo_C[-9]
    ax.errorbar (tempo_C, V_C, yerr = sigma_V_C, fmt = "o", label = "Tensione ai capi di C", capsize = 4, color = "darkslateblue")
    ax.plot (x_axis, V_C_fit2_mask_sin (x_axis, t_0_fit2, V_0_fit2, RC_fit2, offset_fit2, A_fit2, omega_fit2, phi_fit2, x_axis_mask_sin), color = "dodgerblue",label = "Fit tensione ai capi di C con sinusoide")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Tensione (V)")
    ax.set_title ("Tensione ai capi di R e C in un circuito RC con sinusoide")
    ax.grid (True, alpha = 0.4)
    ax.legend ()
    plt.show ()

    # residui normalizzati del fit con sinusoide:
    residui_normalizzati2 = (V_C - V_C_fit2_con_mask_sin (tempo_C, t_0_fit2, V_0_fit2, RC_fit2, offset_fit2, A_fit2, omega_fit2, phi_fit2)) / sigma_V_C
    fig, ax = plt.subplots ()
    ax.errorbar (tempo_C, residui_normalizzati2, yerr = np.ones_like(residui_normalizzati2), fmt = "^", color = "dodgerblue", capsize = 4)
    ax.axhline (0, color = "red", linestyle = "--")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Residui normalizzati")
    ax.set_title ("Residui normalizzati del fit della tensione ai capi di C con sinusoide")
    ax.grid (True, alpha = 0.4)
    plt.show()




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

    p_value_C3 = chi2.sf(m3.fval, m3.ndof)
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

    print (f"Chi2 fit3: {m3.fval}\nndof fit3: {m3.ndof}\nchi2/ndof fit3: {m3.fval / m3.ndof}\np-value fit3: {p_value_C3}")
    print (f"Offset temporale fit3: {t_0_fit3} s")
    print (f"V_0 fit3: {V_0_fit3} V")
    print (f"A fit3: {A_fit3} V")
    print (f"omega_sin fit3: {omega_sin_fit3} Hz")
    print (f"phi fit3: {phi_fit3} rad")
    print (f"B fit3: {B_fit3} V")
    print (f"omega_cos fit3: {omega_cos_fit3} Hz")
    print (f"eta fit3: {eta_fit3} rad")

    # Grafico della tensione ai capi di C con sinusoide e cosinusoide:
    fig, ax = plt.subplots ()
    x_axis = np.linspace (min (tempo_C), max (tempo_C), 5000)
    x_axis_mask_sin = x_axis < tempo_C[-9]
    ax.errorbar (tempo_C, V_C, yerr = sigma_V_C, fmt = "o", label = "Tensione ai capi di C", capsize = 4, color = "darkslateblue")
    ax.plot (x_axis, V_C_fit3_mask_sin_cos (x_axis, t_0_fit3, V_0_fit3, RC_fit3, offset_fit3, A_fit3, omega_sin_fit3, phi_fit3, B_fit3, omega_cos_fit3, eta_fit3, x_axis_mask_sin), color = "crimson", label = "Fit tensione ai capi di C con sinusoide e cosinusoide")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Tensione (V)")
    ax.set_title ("Tensione ai capi di R e C in un circuito RC con sinusoide e cosinusoide")
    ax.grid (True, alpha = 0.4)
    ax.legend ()
    plt.show ()

    # residui normalizzati del fit con sinusoide e cosinusoide:
    residui_normalizzati3 = (V_C - V_C_fit3_con_mask_sin_cos (tempo_C, t_0_fit3, V_0_fit3, RC_fit3, offset_fit3, A_fit3, omega_sin_fit3, phi_fit3, B_fit3, omega_cos_fit3, eta_fit3)) / sigma_V_C
    fig, ax = plt.subplots ()
    ax.errorbar (tempo_C, residui_normalizzati3, yerr = np.ones_like(residui_normalizzati3), fmt = "^", color = "crimson", capsize = 4)
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
    ax.plot (x_axis, [V_C_fit (t, t_0_fit, V_0_fit, RC_fit, offset_fit) for t in x_axis], color = "darkorchid", label = "Fit esponenziale")
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
    ax.set_title ("Confronto tra i residui normalizzati dei fit")
    ax.grid (True, alpha = 0.4)
    ax.legend ()
    plt.show ()








































