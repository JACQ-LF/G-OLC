# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:18:39 2026

@author: Julien
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Paramètres (à remplacer par tes vraies valeurs) ---
M_T = 3.986e14    # GM (m^3/s^2) si M_T = G*M_terre, sinon adapte

phi_dot = np.pi/180.0 * 0.55  #°/s


m_a = 10000.0       # masse à vide (kg)
m_f = 120000.0       # masse de carburant initiale (kg)
Dm  = 350.0         # débit massique (kg/s)

V_e = 3.5e3
T_r = Dm * V_e     # magnitude de la poussée (N)

delta_V = V_e * np.log((m_a+m_f)/m_a)
print(delta_V)

t_burnout = m_f / Dm  # instant où le carburant est épuisé
print(t_burnout)

def masse(t):
    if t < t_burnout:
        return m_a + (m_f - Dm * t)
    return m_a

def poussee(t):
    # coupe la poussée une fois le carburant épuisé
    return T_r if t < t_burnout else 0.0

def phi_t(t):
    if t < t_burnout:
        return min(np.log(phi_dot * t +1), np.pi/2)
    return 0.0

def f(t, y):
    r, rdot, theta, thetadot = y
    m = masse(t)
    T = poussee(t)
    phi = phi_t(t)

    r_ddot     = (T / m) * np.cos(phi) + r * thetadot**2 - M_T / r**2
    theta_ddot = (T * np.sin(phi)) / ( r * m) - 2*(rdot * thetadot) / ( r)

    return [rdot, r_ddot, thetadot, theta_ddot]

y0 = [6.731e6, 0, 0, 0]  

t_span = (0, 5000)                
t_eval = np.linspace(*t_span, 2000)

sol = solve_ivp(f, t_span, y0, method='RK45', t_eval=t_eval,
                 rtol=1e-9, atol=1e-9)

r, rdot, theta, thetadot = sol.y





# Changement de repère : polaire -> cartésien
x = r * np.cos(theta)
y = r * np.sin(theta)

# --- Disque (cercle plein de rayon R) ---
R = 6.731e6
theta_disque = np.linspace(0, 2 * np.pi, 200)
r_disque = np.full_like(theta_disque, R)

x_disque = r_disque * np.cos(theta_disque)
y_disque = r_disque * np.sin(theta_disque)

# --- Tracé cartésien ---
fig, ax = plt.subplots()

ax.plot(x, y, label="courbe", color='blue')
ax.fill(x_disque, y_disque, alpha=0.3, color='orange', label="disque")

ax.set_aspect('equal')  # essentiel pour que le disque reste bien circulaire
ax.axhline(0, color='gray', lw=0.5)
ax.axvline(0, color='gray', lw=0.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Changement de repère polaire → cartésien")
ax.legend()

plt.show()


