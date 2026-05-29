import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

R = 4.673e3 # Ohm
t0_L = 0.50e-6 # s

def V_L_fit (t, V_0, L, offset):
    return offset + V_0 * np.exp (-1 * (t - t0_L) * R / L)

def V_L_fit_autoinduttanza (t, offset, A, B, tau_1, tau_2, t0):
    return offset + A * np.exp (-1 * (t- t0) / tau_1) + B * np.exp (-1 * (t - t0) / tau_2)

if __name__ == "__main__" :

# Tensione misurata ai capi di R

    tempo_R = np.loadtxt ("RL_tempi_R.txt")
    tempo_R = tempo_R * 1e-6
    V_R = np.loadtxt ("RL_V_R.txt")

    # sensibilità --> dal minimo ci spostiamo vediamo la variazione di sens e non del segnale stesso
    sigma_V_R = np.ones (len (V_R)) * 0.04

    # Tensione misurata ai capi di L
    tempo_L = np.loadtxt ("RL_tempi_L.txt")
    tempo_L = tempo_L * np.ones (len(tempo_L)) * 1e-6 # secondi
    V_L = np.loadtxt ("RL_V_L.txt") # V
    
    delta_V_L= np.ones (len (V_L)) * 0.04
    sigma_V_L = (2 *delta_V_L) / np.sqrt (12)

    mask_fit_L = tempo_L >= t0_L

    ls = LeastSquares (tempo_L[mask_fit_L], V_L[mask_fit_L], sigma_V_L[mask_fit_L], V_L_fit)
    m = Minuit (ls, V_0 = 7.8, L = 5e-3, offset = 0)
    m.migrad ()

    p_value_L = chi2.sf(m.fval, m.ndof)
    
    print (f"valore di V_0: {m.values['V_0']}")
    print (f"valore di L: {m.values['L']}")
    print (f"valore di offset: {m.values['offset']}")
    print (f"Chi quadro / ndof: {m.fval} / {m.ndof} = {m.fval / m.ndof}")
    print(f"P value: {p_value_L}")


    fig, ax = plt.subplots ()

    ax.errorbar (tempo_L, V_L,
                 yerr = sigma_V_L,
                 capsize = 4,
                 color = "indigo",
                 linestyle = "None",
                 marker = 'o'
                 )
    
    x_axis = np.linspace (min (tempo_L[mask_fit_L]), max(tempo_L), 5000)
    
    ax.plot (x_axis,
             V_L_fit (x_axis, m.values["V_0"], m.values["L"], m.values["offset"]),
             label = "Fit tensione ai capi di L"
             )

    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Tensione (V)")
    ax.set_title ("Circuito RL: tensione ai capi di L")
    ax.legend ()
    ax.grid (True)
    
    plt.show ()

    # residui normalizzati L
    residui_L = (V_L[mask_fit_L] - V_L_fit (tempo_L[mask_fit_L], m.values["V_0"], m.values["L"], m.values["offset"])) / sigma_V_L[mask_fit_L]

    fig, ax = plt.subplots ()

    ax.errorbar (tempo_L[mask_fit_L], residui_L,
                 yerr = np.ones_like (residui_L),
                 capsize = 4,
                 color = "indigo",
                 linestyle = "None",
                 marker = '^',
                 label = "Residui L"
                 )

    ax.axhline (0, color = "red", linestyle = "--")
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Residui normalizzati")
    ax.set_title ("Residui normalizzati circuito RL")
    ax.legend ()
    ax.grid (True)

    plt.show ()

    # FIT CON AUTOINDUTTANZA

    mask_fit_autoinduttanza = tempo_L <= t0_L

    ls_a = LeastSquares (tempo_L[mask_fit_autoinduttanza], V_L[mask_fit_autoinduttanza], sigma_V_L[mask_fit_autoinduttanza], V_L_fit_autoinduttanza)
    m_a = Minuit (ls_a, offset = 0, A = 10, B = -20, tau_1 = 6.677951421801726e-08, tau_2 = 6.68903067815001e-08, t0 = 0)

    m_a.migrad ()

    print (f"valore di offset: {m_a.values['offset']}")
    print (f"valore di A: {m_a.values['A']}")
    print (f"valore di B: {m_a.values['B']}")
    print (f"valore di tau_1: {m_a.values['tau_1']}")
    print (f"valore di tau_2: {m_a.values['tau_2']}")
    print (f"valore di t0: {m_a.values['t0']}")

    chi2_autoinduttanza = m_a.fval
    ndof_autoinduttanza = m_a.ndof
    p_value_autoinduttanza = chi2.sf(chi2_autoinduttanza, ndof_autoinduttanza)
    print (f"p value autoinduttanza: {p_value_autoinduttanza}")

    fig, ax = plt.subplots ()

    ax.errorbar (tempo_L, V_L,
                 yerr = sigma_V_L,
                 capsize = 4,
                 color = "indigo",
                 linestyle = "None",
                 marker = 'o'
                 )
    
    x_axis2 = np.linspace (min (tempo_L), max(tempo_L[mask_fit_autoinduttanza]), 5000)
    
    ax.plot (x_axis2,
             V_L_fit_autoinduttanza (x_axis2, m_a.values["offset"], m_a.values["A"], m_a.values["B"], m_a.values["tau_1"], m_a.values["tau_2"], m_a.values["t0"]),
             label = "Fit con autoinduttanza",
             color = "lime"
             )
    
    plt.legend ()
    ax.set_xlabel ("Tempo (s)")
    ax.set_ylabel ("Tensione (V)")
    ax.set_title ("Fit con autoinduttanza circuito RL")
    ax.grid (True)
    plt.show ()


    # TOTALE
    # Fit completo su tutti i dati
    ls_full = LeastSquares(tempo_L, V_L, sigma_V_L, V_L_fit_autoinduttanza)
    m_full = Minuit(ls_full, offset=6e-9, A=13, B=-22, 
                tau_1=3e-6, tau_2=1.19e-7, t0=0)
    m_full.migrad()

    for par, val, err in zip(m_full.parameters, m_full.values, m_full.errors):
        print(f"{par}: {val} ± {err}")

    p_value_full = chi2.sf(m_full.fval, m_full.ndof)
    print(f"Fit completo — chi2/ndof: {m_full.fval}/{m_full.ndof}")
    print(f"P value: {p_value_full}")

    fig, ax = plt.subplots()
    ax.errorbar(tempo_L, V_L, yerr=sigma_V_L, capsize=4, color="indigo", linestyle="None", marker='o', label="Dati")
    x_axis_full = np.linspace(min(tempo_L), max(tempo_L), 5000)
    ax.plot(x_axis_full, V_L_fit_autoinduttanza(x_axis_full, m_full.values["offset"], m_full.values["A"], m_full.values["B"], m_full.values["tau_1"], m_full.values["tau_2"], m_full.values["t0"]), label="Fit completo", color="orange")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Tensione (V)")
    ax.set_title("Fit completo circuito RL")
    ax.legend()
    ax.grid(True)
    plt.show()