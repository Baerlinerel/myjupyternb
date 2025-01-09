import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sympy as sp
import pandas as pd

# Parameterdefinition (angepasst an deine Beschreibung)
Kg = 1/131          #Getriebeübersetzung
g = 9.81 #m/s
L1 = 0.155 #m       Länge Rotor
J1 = 0.0289 #kg*m^2 Trägheitsmoment
uv1 = 0.5091 #N*m*s viskose Reibung
uh1 = 0.09358 #N*m  Haftreibung
L2 = 0.24 #m        Länge Pendel
ms = 0.0274 #kg     Stabmasse
mp = 0.0#51 #kg      Pendelmasse
m2 = ms+mp
lp = 0.1916 #mm     Position Pendelmasse
l2 = (1/2*L2*ms+lp*mp)/m2
uv2 = 0.00001332 #N*m*s 	viskose Reibung
uh2 = 0.0003018 #N*m        Haftreibung
Km = 0.01101 #V*s/rad       Motorkonstante
RM = 4.2 #Ohm               Ankerwiderstand
Ue = 12 #V                  Eingansspannung
J0 = J1 + m2*L1**2
J2 = 1/3*ms*L2**2+mp*lp**2
e=0.01

def b_theta(theta_dot, mu_H_i, mu_V_i, epsilon):
    """
    Berechnet b_theta abhängig von theta_dot(t).
    """
    def phi(theta_dot):
        return (3 * mu_H_i + 2 * mu_V_i * epsilon) / (2 * epsilon) * theta_dot - (mu_H_i / (2 * epsilon**3)) * theta_dot**3

    if theta_dot <= -epsilon:
        return mu_V_i * theta_dot - mu_H_i
    elif abs(theta_dot) < epsilon:
        return phi(theta_dot)
    else:
        return mu_V_i * theta_dot + mu_H_i

def system_dynamics(t, state, input_signal):
    """
    Dynamik des Systems.

    state: [x1, x2, x3, x4] (Zustände des Systems)
    input_signal: Funktion, die den Eingang s(t) liefert.
    """
    x1, x2, x3, x4 = state


    # Eingangssignal tau
    s_t = input_signal(t)
    tau = s_t*(Km * Ue / (Kg * RM))

    # Berechnung von b1 und b2
    b1 = b_theta(x2, uh1, uv1, e)#+(Km**2/Kg**2+RM)*x2+x2/Kg
    b2 = b_theta(x4, uh2, uv2, e)

    # Differentialgleichungen
    dx2 = ((-J2**2*x2*x4*np.sin(2.0*x3) - 0.5*J2*L1*l2*m2*x2**2*np.sin(2.0*x3)*np.cos(x3) + J2*L1*l2*m2*x4**2*np.sin(x3) - J2*b1 + J2*tau + L1*b2*l2*m2*np.cos(x3) + 0.5*L1*g*l2**2*m2**2*np.sin(2.0*x3))/(J0*J2 + J2**2*np.sin(x3)**2 - L1**2*l2**2*m2**2*np.cos(x3)**2))
    
    dx4 = ((0.5*J0*J2*x2**2*np.sin(2.0*x3) - J0*b2 - J0*g*l2*m2*np.sin(x3) + J2**2*x2**2*np.sin(x3)**3*np.cos(x3) + J2*L1*l2*m2*x2*x4*np.sin(2.0*x3)*np.cos(x3) - J2*b2*np.sin(x3)**2 - J2*g*l2*m2*np.sin(x3)**3 - 0.5*L1**2*l2**2*m2**2*x4**2*np.sin(2.0*x3) + L1*b1*l2*m2*np.cos(x3) - L1*l2*m2*tau*np.cos(x3))/(J0*J2 + J2**2*np.sin(x3)**2 - L1**2*l2**2*m2**2*np.cos(x3)**2))
    
    dx1 = x2
    dx3 = x4

    return [dx1, dx2, dx3, dx4]


import numpy as np
from scipy.interpolate import UnivariateSpline



# Funktion für den Einsatz
def s_hat(s):
    """
    Gibt den approximierten Wert von ^s für einen gegebenen s-Wert zurück.

    Parameter:
        s (float oder np.ndarray): Der Wert oder die Werte von s, für die ^s berechnet werden soll.

    Rückgabe:
        float oder np.ndarray: Der interpolierte Wert von ^s.
    """
    return spline(s)


def simulate_system(initial_state, t_span, input_signal, t_eval):
    """
    Simuliert das System mit den gegebenen Parametern.

    initial_state: Anfangszustand [x1, x2, x3, x4]
    t_span: Zeitintervall (t0, tf)
    input_signal: Funktion, die s(t) liefert.
    t_eval: Zeitpunkte, an denen der Zustand ausgewertet wird.
    """
    solution = solve_ivp(system_dynamics, t_span, initial_state, t_eval=t_eval, args=(input_signal,))
    return solution

def input_signal(t):
    # Originaldaten
   s_values = np.array([-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.09, -0.08, -0.07, -0.065, -0.06360108, 0.06222463, 0.065, 0.07, 0.08, 0.09, 0.1, 0.2])
   s_hat_values = np.array([-1, -0.9470693, -0.9119909, -0.8782726, -0.8319191, -0.7614319, -0.6578094, -0.5145467, -0.327636, -0.095566, -0.0698916, -0.0437758, -0.0172211, -0.0037801, 0, 0,  0.0074904, 0.0209181, 0.047511, 0.0737456, 0.099612, 0.336181])

        # Spline-Interpolation
   spline = UnivariateSpline(s_values, s_hat_values, s=0.0001)  # Glatte Interpolation
   if 0 < t < 1:
        s = 3/20*t
        return spline(s)
   elif 1 <= t <= 1.4:
        s = -t+23/20
        return spline(s)
   elif 1.4 < t < 2:
        s = 5/12*t-5/6
        return spline(s)
   else:
        s=0
        return spline(s)

# Simulationseinstellungen
initial_state = [0, 0, 0, 0]  # Anfangszustände [x1, x2, x3, x4]
t_span = (0, 10)  # Simulationszeitraum von 0 bis 10 Sekunden
t_eval = np.linspace(t_span[0], t_span[1], 10000)  # Zeitpunkte für Auswertung

# Simulation starten
solution = simulate_system(initial_state, t_span, input_signal, t_eval)

'''#Ergebnisse plotten
plt.figure(figsize=(10, 6))
plt.plot(solution.t, solution.y[0], label='x1 (Theta1)')
plt.plot(solution.t, solution.y[1], label='x2 (Theta1_dot)')
plt.plot(solution.t, solution.y[2], label='x3 (Theta2)')
plt.plot(solution.t, solution.y[3], label='x4 (Theta2_dot)')
plt.xlabel('Zeit (s)')
plt.ylabel('Zustände')
plt.title('Systemsimulation')
plt.legend()
plt.grid()
plt.show()
'''
validation_data = pd.read_csv("valSimEingangsnichtlin.csv")

# Validierungszeit und Zustände extrahieren
t_val = validation_data['time'].values
x1_val = validation_data['F: theta1'].values
x2_val = validation_data['F: theta1dot'].values
x3_val = validation_data['F: theta2'].values
x4_val = validation_data['F: theta2dot'].values

# Simulationsergebnisse mit Validierungsdaten vergleichen
plt.figure(figsize=(10, 8))

plt.subplot(3, 2, 1)
plt.plot(solution.t, solution.y[0], label='Simulation x1', linewidth=1)
plt.plot(t_val, x1_val, 'r', label='Validierung x1', linewidth=1)
plt.xlabel('Zeit (s)')
plt.ylabel('x1 (Theta1)')
plt.legend()
plt.grid()

plt.subplot(3, 2, 2)
plt.plot(solution.t, solution.y[1], label='Simulation x2', linewidth=1)
plt.plot(t_val, x2_val, 'r', label='Validierung x2', linewidth=1)
plt.xlabel('Zeit (s)')
plt.ylabel('x2 (Theta1_dot)')
plt.legend()
plt.grid()

plt.subplot(3, 2, 3)
plt.plot(solution.t, solution.y[2], label='Simulation x3', linewidth=1)
plt.plot(t_val, x3_val, 'r', label='Validierung x3', linewidth=1)
plt.xlabel('Zeit (s)')
plt.ylabel('x3 (Theta2)')
plt.legend()
plt.grid()

plt.subplot(3, 2, 4)
plt.plot(solution.t, solution.y[3], label='Simulation x4', linewidth=1)
plt.plot(t_val, x4_val, 'r', label='Validierung x4', linewidth=1)
plt.xlabel('Zeit (s)')
plt.ylabel('x4 (Theta2_dot)')
plt.legend()
plt.grid()

plt.subplot(3, 2, 6)
plt.plot(solution.t, solution.y[0], label='x1 (Theta1)')
plt.plot(solution.t, solution.y[1], label='x2 (Theta1_dot)')
plt.plot(solution.t, solution.y[2], label='x3 (Theta2)')
plt.plot(solution.t, solution.y[3], label='x4 (Theta2_dot)')
plt.xlabel('Zeit (s)')
plt.ylabel('Zustände')
plt.legend()
plt.grid()

plt.subplot(3, 2, 5)
plt.plot(t_val, x1_val, label='Validierung x1', linewidth=1)
plt.plot(t_val, x2_val, label='Validierung x2', linewidth=1)
plt.plot(t_val, x3_val, label='Validierung x3', linewidth=1)
plt.plot(t_val, x4_val, label='Validierung x4', linewidth=1)
plt.xlabel('Zeit (s)')
plt.ylabel('Zustände')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
