import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2


R = 4.673e3 # Ohm
t0_L = 0.50e-6 # s


# Andamento tensione ai capi di L:
def V_L_fit (t, V_0, L, offset):
    return offset + V_0 * np.exp (-1 * (t - t0_L) * R / L)


def V_L_fit_parassita (t, offset, A, B, tau_1, tau_2, t0):
    return offset + A * np.exp (-1 * (t - t0) / tau_1) + B * np.exp (-1 * (t - t0) / tau_2)


# Andamento tensione ai capi di R:
def V_R_fit (t, V_0, L, offset):
    return offset + V_0 * (1 - np.exp (-1 * (t - t0_L) * R / L))

'''
def V_R_fit_parassita (t, offset, A, B, tau_1, tau_2, t0):
    return offset + A * (1 - np.exp (-1 * (t - t0) / tau_1)) + B * (1 - np.exp (-1 * (t - t0) / tau_2))
'''


if __name__ == "__main__":

    # Tensione misurata ai capi di R:
    tempo_R = np.loadtxt ("RL_tempi_R.txt") * 1e-6
    V_R = np.loadtxt ("RL_V_R.txt")

    # sensibilità --> dal minimo ci spostiamo vediamo la variazione di sens e non del segnale stesso
    Delta_V_R = np.ones (len (V_R)) * 0.04
    sigma_V_R = (2 * Delta_V_R) / np.sqrt (12)

    # Tensione misurata ai capi di L:
    tempo_L = np.loadtxt ("RL_tempi_L.txt") * 1e-6
    V_L = np.loadtxt ("RL_V_L.txt") # V

    Delta_V_L = np.ones (len (V_L)) * 0.04
    sigma_V_L = (2 * Delta_V_L) / np.sqrt (12)


    # TENSIONE AI CAPI DI L

    # Fit della tensione ai capi di L:
    mask_fit_L = tempo_L >= t0_L # per tenere conto solo dei dati che scendono, come previsto dalla legge di carica

    ls = LeastSquares (tempo_L[mask_fit_L], V_L[mask_fit_L], sigma_V_L[mask_fit_L], V_L_fit)
    m = Minuit (ls, V_0 = 7.8, L = 5e-3, offset = 0)
    m.migrad ()

    p_value_L = chi2.sf (m.fval, m.ndof)
    V_0_fit = m.values["V_0"]
    L_fit = m.values["L"]
    offset_fit = m.values["offset"]

    print (f"Chi2 L: {m.fval}\nndof L: {m.ndof}\nchi2/ndof L: {m.fval / m.ndof}\np-value L: {p_value_L}")
    print (f"V_0 L: {V_0_fit} \pm {m.errors['V_0']} V")
    print (f"L: {L_fit} \pm {m.errors['L']} H")
    print (f"offset L: {offset_fit} \pm {m.errors['offset']} V")

    # Grafico della tensione ai capi di L:
    fig, ax = plt.subplots ()
    x_axis = np.linspace (min (tempo_L[mask_fit_L]), max (tempo_L), 5000)
    ax.errorbar (tempo_L, V_L, yerr = sigma_V_L, capsize = 4, color = "indigo", linestyle = "None", marker = "o", label = "Tensione ai capi di L")
    ax.plot (x_axis, V_L_fit (x_axis, V_0_fit, L_fit, offset_fit), label = "Fit tensione ai capi di L", color = "forestgreen")
    ax.vlines (t0_L, ymin = 0.0, ymax = max (V_L), color = "red", linestyle = "--", label = "t0")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Tensione (V)")
    ax.set_title ("Circuito RL: tensione ai capi di L")
    ax.legend ()
    ax.grid (True)
    plt.show ()

    # Residui normalizzati del fit della tensione ai capi di L:
    residui_L = (V_L[mask_fit_L] - V_L_fit (tempo_L[mask_fit_L], V_0_fit, L_fit, offset_fit)) / sigma_V_L[mask_fit_L]

    fig, ax = plt.subplots ()
    ax.errorbar (tempo_L[mask_fit_L], residui_L, yerr = np.ones_like (residui_L), capsize = 4, color = "forestgreen", linestyle = "None", marker = "^", label = "Residui L")
    ax.axhline (0, color = "red", linestyle = "--")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Residui normalizzati")
    ax.set_title ("Residui normalizzati del fit della tensione ai capi di L")
    ax.legend ()
    ax.grid (True)
    plt.show ()


    # FIT CIRCUITO RLC CON C PARASSITA

    mask_fit_parassita = tempo_L <= t0_L

    ls_a = LeastSquares (tempo_L[mask_fit_parassita], V_L[mask_fit_parassita], sigma_V_L[mask_fit_parassita], V_L_fit_parassita)
    m_a = Minuit (ls_a, offset = 0, A = 10, B = -20, tau_1 = 6.677951421801726e-08, tau_2 = 6.68903067815001e-08, t0 = 0)
    m_a.migrad ()

    p_value_autoinduttanza = chi2.sf (m_a.fval, m_a.ndof)
    offset_fit_a = m_a.values["offset"]
    A_fit_a = m_a.values["A"]
    B_fit_a = m_a.values["B"]
    tau_1_fit_a = m_a.values["tau_1"]
    tau_2_fit_a = m_a.values["tau_2"]
    t0_fit_a = m_a.values["t0"]

    print (f"Chi2 C parassita: {m_a.fval}\nndof C parassita: {m_a.ndof}\nchi2/ndof C parassita: {m_a.fval / m_a.ndof}\np-value C parassita: {p_value_autoinduttanza}")
    print (f"offset C parassita: {offset_fit_a} \pm {m_a.errors['offset']} V")
    print (f"A C parassita: {A_fit_a} \pm {m_a.errors['A']} V")
    print (f"B C parassita: {B_fit_a} \pm {m_a.errors['B']} V")
    print (f"tau_1 C parassita: {tau_1_fit_a} \pm {m_a.errors['tau_1']} s")
    print (f"tau_2 C parassita: {tau_2_fit_a} \pm {m_a.errors['tau_2']} s")
    print (f"t0 C parassita: {t0_fit_a} \pm {m_a.errors['t0']} s")

    # Grafico della tensione ai capi di L con C parassita:
    fig, ax = plt.subplots ()
    x_axis2 = np.linspace (min (tempo_L), max (tempo_L[mask_fit_parassita]), 5000)
    ax.errorbar (tempo_L, V_L, yerr = sigma_V_L, capsize = 4, color = "indigo", linestyle = "None", marker = "o", label = "Tensione ai capi di L")
    ax.plot (x_axis2, V_L_fit_parassita (x_axis2, offset_fit_a, A_fit_a, B_fit_a, tau_1_fit_a, tau_2_fit_a, t0_fit_a), label = "Fit con C parassita", color = "lime")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Tensione (V)")
    ax.set_title ("Fit circuito RLC con C parassita")
    ax.legend ()
    ax.grid (True)
    plt.show ()

    # Residui normalizzati del fit con C parassita:
    residui_parassita = (V_L[mask_fit_parassita] - V_L_fit_parassita (tempo_L[mask_fit_parassita], offset_fit_a, A_fit_a, B_fit_a, tau_1_fit_a, tau_2_fit_a, t0_fit_a)) / sigma_V_L[mask_fit_parassita]

    fig, ax = plt.subplots ()
    ax.errorbar (tempo_L[mask_fit_parassita], residui_parassita, yerr = np.ones_like (residui_parassita), capsize = 4, color = "lime", linestyle = "None", marker = "^", label = "Residui C parassita")
    ax.axhline (0, color = "red", linestyle = "--")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Residui normalizzati")
    ax.set_title ("Residui normalizzati del fit con C parassita")
    ax.legend ()
    ax.grid (True)
    plt.show ()


    # FIT COMPLETO SU TUTTI I DATI

    ls_full = LeastSquares (tempo_L, V_L, sigma_V_L, V_L_fit_parassita)
    m_full = Minuit (ls_full, offset = 6e-9, A = 13, B = -22, tau_1 = 3e-6, tau_2 = 1.19e-7, t0 = 0)
    m_full.migrad ()

    p_value_full = chi2.sf (m_full.fval, m_full.ndof)
    offset_fit_full = m_full.values["offset"]
    A_fit_full = m_full.values["A"]
    B_fit_full = m_full.values["B"]
    tau_1_fit_full = m_full.values["tau_1"]
    tau_2_fit_full = m_full.values["tau_2"]
    t0_fit_full = m_full.values["t0"]

    print (f"Chi2 fit completo: {m_full.fval}\nndof fit completo: {m_full.ndof}\nchi2/ndof fit completo: {m_full.fval / m_full.ndof}\np-value fit completo: {p_value_full}")
    for par, val, err in zip (m_full.parameters, m_full.values, m_full.errors):
        print (f"{par}: {val} ± {err}")

    # Grafico della tensione ai capi di L con fit completo:
    fig, ax = plt.subplots ()
    x_axis_full = np.linspace (min (tempo_L), max (tempo_L), 5000)
    ax.errorbar (tempo_L, V_L, yerr = sigma_V_L, capsize = 4, color = "indigo", linestyle = "None", marker = "o", label = "Tensione ai capi di L")
    ax.plot (x_axis_full, V_L_fit_parassita (x_axis_full, offset_fit_full, A_fit_full, B_fit_full, tau_1_fit_full, tau_2_fit_full, t0_fit_full), label = "Fit completo", color = "darkgreen")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Tensione (V)")
    ax.set_title ("Fit completo della tensione ai capi di L")
    ax.legend ()
    ax.grid (True)
    plt.show ()

    # Residui normalizzati fit completo:
    residui_full = (V_L - V_L_fit_parassita (tempo_L, offset_fit_full, A_fit_full, B_fit_full, tau_1_fit_full, tau_2_fit_full, t0_fit_full)) / sigma_V_L

    fig, ax = plt.subplots ()
    ax.errorbar (tempo_L, residui_full, yerr = np.ones_like (residui_full), capsize = 4, color = "darkgreen", linestyle = "None", marker = "^", label = "Residui fit completo")
    ax.axhline (0, color = "red", linestyle = "--")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Residui normalizzati")
    ax.set_title ("Residui normalizzati del fit completo")
    ax.legend ()
    ax.grid (True)
    plt.show ()

    '''
    # Grafico finale con tutti i fit sovrapposti:
    fig, ax = plt.subplots ()
    ax.errorbar (tempo_L, V_L, yerr = sigma_V_L, capsize = 4, color = "indigo", linestyle = "None", marker = "o", label = "Tensione ai capi di L")
    ax.plot (x_axis, V_L_fit (x_axis, V_0_fit, L_fit, offset_fit), color = "forestgreen", label = "Fit esponenziale")
    ax.plot (x_axis2, V_L_fit_parassita (x_axis2, offset_fit_a, A_fit_a, B_fit_a, tau_1_fit_a, tau_2_fit_a, t0_fit_a), color = "lime", label = "Fit con C parassita")
    ax.plot (x_axis_full, V_L_fit_parassita (x_axis_full, offset_fit_full, A_fit_full, B_fit_full, tau_1_fit_full, tau_2_fit_full, t0_fit_full), color = "darkgreen", label = "Fit completo")
    ax.vlines (t0_L, ymin = min (V_L), ymax = max (V_L), color = "red", linestyle = "--", label = "t0")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Tensione (V)")
    ax.set_title ("Confronto tra i fit della tensione ai capi di L")
    ax.legend ()
    ax.grid (True)
    plt.show ()

    # Grafico finale con tutti i residui sovrapposti:
    fig, ax = plt.subplots ()
    ax.errorbar (tempo_L[mask_fit_L], residui_L, yerr = np.ones_like (residui_L), capsize = 4, color = "forestgreen", linestyle = "None", marker = "o", label = "Residui fit esponenziale")
    ax.errorbar (tempo_L[mask_fit_parassita], residui_parassita, yerr = np.ones_like (residui_parassita), capsize = 4, color = "lime", linestyle = "None", marker = "^", label = "Residui fit con C parassita")
    ax.errorbar (tempo_L, residui_full, yerr = np.ones_like (residui_full), capsize = 4, color = "darkgreen", linestyle = "None", marker = "s", label = "Residui fit completo")
    ax.axhline (0, color = "red", linestyle = "--")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Residui normalizzati")
    ax.set_title ("Confronto tra i residui normalizzati dei fit della tensione ai capi di L")
    ax.legend ()
    ax.grid (True)
    plt.show ()
    '''


    # TENSIONE AI CAPI DI R

    # Fit della tensione ai capi di R:
    ls_R = LeastSquares (tempo_R, V_R, sigma_V_R, V_R_fit)
    m_R = Minuit (ls_R, V_0 = 4.5, L = L_fit, offset = 0)
    m_R.migrad ()

    p_value_R = chi2.sf (m_R.fval, m_R.ndof)
    V_0_fit_R = m_R.values["V_0"]
    L_fit_R = m_R.values["L"]
    offset_fit_R = m_R.values["offset"]

    print (f"Chi2 R: {m_R.fval}\nndof R: {m_R.ndof}\nchi2/ndof R: {m_R.fval / m_R.ndof}\np-value R: {p_value_R}")
    print (f"V_0 R: {V_0_fit_R} \pm {m_R.errors['V_0']} V")
    print (f"L R: {L_fit_R} \pm {m_R.errors['L']} H")
    print (f"offset R: {offset_fit_R} \pm {m_R.errors['offset']} V")

    # Grafico della tensione ai capi di R:
    fig, ax = plt.subplots ()
    x_axis_R = np.linspace (min (tempo_R), max (tempo_R), 5000)
    ax.errorbar (tempo_R, V_R, yerr = sigma_V_R, capsize = 4, color = "indigo", linestyle = "None", marker = "o", label = "Tensione ai capi di R")
    ax.plot (x_axis_R, V_R_fit (x_axis_R, V_0_fit_R, L_fit_R, offset_fit_R), label = "Fit tensione ai capi di R", color = "lightseagreen")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Tensione (V)")
    ax.set_title ("Circuito RL: tensione ai capi di R")
    ax.legend ()
    ax.grid (True)
    plt.show ()

    # Residui normalizzati del fit della tensione ai capi di R:
    residui_R = (V_R - V_R_fit (tempo_R, V_0_fit_R, L_fit_R, offset_fit_R)) / sigma_V_R

    fig, ax = plt.subplots ()
    ax.errorbar (tempo_R, residui_R, yerr = np.ones_like (residui_R), capsize = 4, color = "lightseagreen", linestyle = "None", marker = "^", label = "Residui R")
    ax.axhline (0, color = "red", linestyle = "--")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Residui normalizzati")
    ax.set_title ("Residui normalizzati del fit della tensione ai capi di R")
    ax.legend ()
    ax.grid (True)
    plt.show ()

    '''
    # FIT COMPLETO SU TUTTI I DATI DELLA TENSIONE AI CAPI DI R

    ls_full_R = LeastSquares (tempo_R, V_R, sigma_V_R, V_R_fit_parassita)
    m_full_R = Minuit (ls_full_R, offset = -3, A = 8, B = -3, tau_1 = 8e-7, tau_2 = 1e-7, t0 = 0)
    m_full_R.migrad ()

    p_value_full_R = chi2.sf (m_full_R.fval, m_full_R.ndof)
    offset_fit_full_R = m_full_R.values["offset"]
    A_fit_full_R = m_full_R.values["A"]
    B_fit_full_R = m_full_R.values["B"]
    tau_1_fit_full_R = m_full_R.values["tau_1"]
    tau_2_fit_full_R = m_full_R.values["tau_2"]
    t0_fit_full_R = m_full_R.values["t0"]

    print (f"Chi2 fit completo R: {m_full_R.fval}\nndof fit completo R: {m_full_R.ndof}\nchi2/ndof fit completo R: {m_full_R.fval / m_full_R.ndof}\np-value fit completo R: {p_value_full_R}")
    for par, val, err in zip (m_full_R.parameters, m_full_R.values, m_full_R.errors):
        print (f"{par} R: {val} ± {err}")

    # Grafico della tensione ai capi di R con fit completo:
    fig, ax = plt.subplots ()
    ax.errorbar (tempo_R, V_R, yerr = sigma_V_R, capsize = 4, color = "darkorange", linestyle = "None", marker = "o", label = "Tensione ai capi di R")
    ax.plot (x_axis_R, V_R_fit_parassita (x_axis_R, offset_fit_full_R, A_fit_full_R, B_fit_full_R, tau_1_fit_full_R, tau_2_fit_full_R, t0_fit_full_R), label = "Fit completo tensione ai capi di R", color = "darkred")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Tensione (V)")
    ax.set_title ("Fit completo della tensione ai capi di R")
    ax.legend ()
    ax.grid (True)
    plt.show ()

    # Residui normalizzati fit completo della tensione ai capi di R:
    residui_full_R = (V_R - V_R_fit_parassita (tempo_R, offset_fit_full_R, A_fit_full_R, B_fit_full_R, tau_1_fit_full_R, tau_2_fit_full_R, t0_fit_full_R)) / sigma_V_R

    fig, ax = plt.subplots ()
    ax.errorbar (tempo_R, residui_full_R, yerr = np.ones_like (residui_full_R), capsize = 4, color = "darkred", linestyle = "None", marker = "^", label = "Residui fit completo R")
    ax.axhline (0, color = "red", linestyle = "--")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Residui normalizzati")
    ax.set_title ("Residui normalizzati del fit completo della tensione ai capi di R")
    ax.legend ()
    ax.grid (True)
    plt.show ()

    # Grafico finale con tutti i fit sovrapposti per R:
    fig, ax = plt.subplots ()
    ax.errorbar (tempo_R, V_R, yerr = sigma_V_R, capsize = 4, color = "darkorange", linestyle = "None", marker = "o", label = "Tensione ai capi di R")
    ax.plot (x_axis_R, V_R_fit (x_axis_R, V_0_fit_R, L_fit_R, offset_fit_R), color = "firebrick", label = "Fit esponenziale")
    ax.plot (x_axis_R, V_R_fit_parassita (x_axis_R, offset_fit_full_R, A_fit_full_R, B_fit_full_R, tau_1_fit_full_R, tau_2_fit_full_R, t0_fit_full_R), color = "darkred", label = "Fit completo")
    ax.vlines (t0_L, ymin = min (V_R), ymax = max (V_R), color = "red", linestyle = "--", label = "t0")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Tensione (V)")
    ax.set_title ("Confronto tra i fit della tensione ai capi di R")
    ax.legend ()
    ax.grid (True)
    plt.show ()

    # Grafico finale con tutti i residui sovrapposti per R:
    fig, ax = plt.subplots ()
    ax.errorbar (tempo_R, residui_R, yerr = np.ones_like (residui_R), capsize = 4, color = "firebrick", linestyle = "None", marker = "o", label = "Residui fit esponenziale")
    ax.errorbar (tempo_R, residui_full_R, yerr = np.ones_like (residui_full_R), capsize = 4, color = "darkred", linestyle = "None", marker = "s", label = "Residui fit completo")
    ax.axhline (0, color = "red", linestyle = "--")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Residui normalizzati")
    ax.set_title ("Confronto tra i residui normalizzati dei fit della tensione ai capi di R")
    ax.legend ()
    ax.grid (True)
    plt.show ()
    '''


   # ============================================================
    # RICAVO DI R_L e C_p dai parametri del fit parassita
    #
    # Modello fisico (dal grafico):
    #   - salita  (t < t_picco): C_p si carica attraverso R + R_L
    #                             → tau_2 ≈ (R + R_L) * C_p
    #   - discesa (t > t_picco): L si scarica attraverso R + R_L
    #                             → tau_1 = L / (R + R_L)
    #
    # Quindi:
    #   R + R_L = L / tau_1
    #   R_L     = L / tau_1 - R
    #   C_p     = tau_2 / (R + R_L) = tau_1 * tau_2 / L
    # ============================================================
 
    # --- Valori noti ---
    R       = 4.673e3       # Ohm (resistenza esterna, misurata)
 
    # --- Dal fit esponenziale su V_L (discesa RL pura) ---
    L       = L_fit         # H   <-- metti qui il valore di L_fit
    dL      = m.errors['L'] # H   <-- e il suo errore da Minuit
 
    # --- Dal fit parassita (V_L_fit_parassita su t < t0) ---
    tau_1   = tau_1_fit_a           # s  (costante di discesa ~ L/(R+R_L))
    tau_2   = tau_2_fit_a           # s  (costante di salita  ~ (R+R_L)*Cp)
    d_tau_1 = m_a.errors['tau_1']   # s
    d_tau_2 = m_a.errors['tau_2']   # s
 
    # ============================================================
    # CALCOLO CENTRALE
    # ============================================================
 
    R_tot   = L / tau_1                     # R + R_L  [Ohm]
    R_L     = R_tot - R                     # R_L      [Ohm]
    C_p     = (tau_1 * tau_2) / L           # C_p      [F]
 
    # ============================================================
    # PROPAGAZIONE DEGLI ERRORI (derivate parziali)
    # ============================================================
 
    # R_tot = L / tau_1
    dR_tot_dL     =  1.0 / tau_1
    dR_tot_dtau1  = -L   / tau_1**2
 
    sigma_R_tot = np.sqrt(
        (dR_tot_dL    * dL     )**2 +
       (dR_tot_dtau1 * d_tau_1)**2
    )
 
    # R_L = R_tot - R  →  sigma_R_L = sigma_R_tot (R è esatta)
    sigma_R_L = sigma_R_tot
 
    # C_p = tau_1 * tau_2 / L
    dCp_dtau1 =  tau_2 / L
    dCp_dtau2 =  tau_1 / L
    dCp_dL    = -(tau_1 * tau_2) / L**2
 
    sigma_C_p = np.sqrt(
       (dCp_dtau1 * d_tau_1)**2 +
       (dCp_dtau2 * d_tau_2)**2 +
       (dCp_dL    * dL     )**2
    )
 
# ============================================================
# VERIFICA CONSISTENZA
# ============================================================
 
# tau_1 dovrebbe essere ≈ L / (R + R_L)  [lo è per costruzione]
    tau_1_check = L / (R + R_L)
 
# tau_2 dovrebbe essere ≈ (R + R_L) * C_p
    tau_2_check = (R + R_L) * C_p
 
# Resistenza critica del circuito RLC
    R_critica = 2 * np.sqrt(L / C_p)
 
# ============================================================
# STAMPA RISULTATI
# ============================================================
 
    print("=" * 55)
    print("  RISULTATI  R_L  e  C_p")
    print("=" * 55)
    print(f"  R + R_L  = {R_tot:.2f} ± {sigma_R_tot:.2f}  Ohm")
    print(f"  R_L      = {R_L:.2f} ± {sigma_R_L:.2f}  Ohm")
    print(f"  C_p      = {C_p*1e12:.3f} ± {sigma_C_p*1e12:.3f}  pF")
    print()
    print("  Verifica di consistenza:")
    print(f"  tau_1 (fit)   = {tau_1*1e6:.4f} us")
    print(f"  L/(R+R_L)     = {tau_1_check*1e6:.4f} us   <- deve coincidere")
    print(f"  tau_2 (fit)   = {tau_2*1e9:.2f} ns")
    print(f"  (R+R_L)*C_p   = {tau_2_check*1e9:.2f} ns   <- deve coincidere")
    print()
    print(f"  R_critica     = {R_critica:.1f} Ohm")
    print(f"  R_L / R       = {R_L/R*100:.1f} %")
    print(f"  (R+R_L) / R_c = {(R+R_L)/R_critica:.3f}  (>1 = sovrasmorzato)")
    print("=" * 55)