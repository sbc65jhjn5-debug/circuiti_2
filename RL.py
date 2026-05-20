import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

R = 4.673e3 # Ohm
t0_L = 0.50e-6 # s

def V_L_fit (t, V_0, L, offset):
    return offset + V_0 * np.exp (-1 * (t - t0_L) * R / L)

if __name__ == "__main__" :

# Tensione misurata ai capi di R

    tempo_R = np.array ([0.4, 
                       0.6,
                       0.72,
                       0.8,
                       0.92,
                       1.0,
                       1.20,
                       1.40,
                       1.60,
                       1.80,
                       2.00,
                       2.20,
                       2.40,
                       2.60,
                       2.80,
                       3.00,
                       3.20,
                       3.40,
                       3.60,
                       3.80,
                       4.52,
                       6.00,
                       7.00
                       ]) * 1e-6 # secondi
    V_R = np.array ([-2.56, 
                   -1.24,
                   -0.48, 
                   -0.04,
                   0.52,
                   0.84,
                   1.60,
                   2.20,
                   2.64,
                   3.00,
                   3.28,
                   3.52,
                   3.72,
                   3.88,
                   4.00,
                   4.08,
                   4.16,
                   4.24,
                   4.28,
                   4.32,
                   4.44,
                   4.48,
                   4.52
                   ]) # V

    # sensibilità --> dal minimo ci spostiamo vediamo la variazione di sens e non del segnale stesso
    sigma_V_R = np.ones (len (V_R)) * 0.04

    # Tensione misurata ai capi di L
    tempo_L = np.array ([0.01,
                       0.02,
                       0.04,
                       0.06,
                       0.08,
                       0.10,
                       0.12,
                       0.14,
                       0.16,
                       0.18,
                       0.20,
                       0.22,
                       0.24,
                       0.26,
                       0.28,
                       0.30,
                       0.32,
                       0.34,
                       0.38,
                       0.40,
                       0.44,
                       0.48,
                       0.50,
                       0.54,
                       0.58,
                       0.62,
                       0.66,
                       0.70,
                       0.74,
                       0.78,
                       0.82,
                       0.86,
                       0.90,
                       1.00,
                       1.25,
                       1.50,
                       1.75,
                       2.00,
                       2.75,
                       3.50,
                       4.00
                       ]) * 1e-6 # secondi
    V_L = np.array ([1.48,
                   2.16,
                   3.36,
                   4.32,
                   5.16,
                   5.84,
                   6.32,
                   6.76,
                   7.12,
                   7.32,
                   7.52,
                   7.68,
                   7.76,
                   7.76,
                   7.80,
                   7.80,
                   7.72,
                   7.72,
                   7.52,
                   7.44,
                   7.20,
                   7.00,
                   6.84,
                   6.60,
                   6.36,
                   6.08,
                   5.84,
                   5.60,
                   5.36,
                   5.12,
                   4.88,
                   4.68,
                   4.48,
                   4.00,
                   3.00,
                   2.28,
                   1.72,
                   1.28,
                   0.56,
                   0.26,
                   0.16
                   ]) # V
    
    delta_V_L= np.ones (len (V_L)) * 0.04
    sigma_V_L = delta_V_L / np.sqrt (12)

    mask_fit_L = tempo_L >= t0_L

    ls = LeastSquares (tempo_L[mask_fit_L], V_L[mask_fit_L], delta_V_L[mask_fit_L], V_L_fit)
    m = Minuit (ls, V_0 = 7.8, L = 5e-3, offset = 0)
    m.migrad ()

    p_value_L = chi2.sf(m.fval, m.ndof)
    
    print (f"valore di V_0: {m.values['V_0']}")
    print (f"valore di L: {m.values["L"]}")
    print (f"valore di offset: {m.values['offset']}")
    print (f"Chi quadro / ndof: {m.fval} / {m.ndof} = {m.fval / m.ndof}")
    print(f"P value: {p_value_L}")


    fig, ax = plt.subplots ()

    ax.errorbar (tempo_L, V_L,
                 yerr = delta_V_L,
                 capsize = 4,
                 color = "indigo",
                 linestyle = "None",
                 marker = 'o'
                 )
    
    x_axis = np.linspace (min (tempo_L[mask_fit_L]), max(tempo_L), 5000)
    
    ax.plot (x_axis,
             [V_L_fit (x, m.values["V_0"], m.values["L"], m.values["offset"]) for x in x_axis]
             )
    
    plt.show ()