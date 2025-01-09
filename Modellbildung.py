import sympy as sp
import numpy as np

Kg = 1/131          #Getriebeübersetzung
g = 9.81 #m/s
L1 = 0.155 #m       Länge Rotor
J1 = 0.0289 #kg*m^2 Trägheitsmoment
uv1 = 0.5091 #N*m*s viskose Reibung
uh1 = 0.09358 #N*m  Haftreibung
L2 = 0.24 #m        Länge Pendel
ms = 0.0274 #kg     Stabmasse
mp = 0.051 #kg      Pendelmasse
m2 = ms+mp
lp = 0.1916 #mm     Position Pendelmasse
l2 = (1/2*L2*ms+lp*mp)/1/2*L2+lp
uv2 = 0.00001332 #N*m*s 	viskose Reibung
uh2 = 0.0003018 #N*m        Haftreibung
Km = 0.01101 #V*s/rad       Motorkonstante
RM = 4.2 #Ohm               Ankerwiderstand
Ue = 12 #V                  Eingansspannung
J0 = J1 + m2*L1**2
J2 = 1/3*ms*L2**2+mp*lp**2
e=0.01


# Zustandsvariablen und Parameter
x1, x2, x3, x4, dx4, dx3, dx2, dx1, s = sp.symbols('x1 x2 x3 x4 dx4 dx3 dx2 dx1 s')
m2, L1, l2, J0, J2, b1, b2, g, kM, Ue, Kg, RM, tau, s= sp.symbols('m2 L1 l2 J0 J2 b1 b2 g kM Ue Kg RM tau s')
uv1, uv2, uh1, uh2=sp.symbols('uv1, uv2, uh1, uh2')


# Gleichungen
d2 = sp.solve(
    m2*L1*l2*(sp.cos(x3)*((0.5*J2*x2**2*sp.sin(2.0*x3) - L1*dx2*l2*m2*sp.cos(x3) - b2 - g*l2*m2*sp.sin(x3))/J2) - sp.sin(x3)*x4**2)+J0*dx2+J2*(sp.sin(x3)**2*dx2 + sp.sin(2*x3)*x2*x4)+b1-tau,
    dx2
)
d4 = sp.solve(
    m2*L1*l2*sp.cos(x3)*((-J2*x2*x4*sp.sin(2*x3) - L1*dx4*l2*m2*sp.cos(x3) + L1*l2*m2*x4**2*sp.sin(x3) - b1 + tau)/(J0 + J2*sp.sin(x3)**2)) - J2*(0.5*sp.sin(2*x3)*x2**2 - dx4) + b2 + m2*l2*g*sp.sin(x3), 
    dx4
)
dx1 = x2
dx3 = x4
tau = (kM*Ue/Kg*RM)*s



# Ausgabe
print('dx2 = ')
print(d2)
print('-----')
print('dx4 =')
print(d4)
