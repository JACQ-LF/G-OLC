# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:18:39 2026

@author: Julien
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --- Physical Parameters ---
G = 6.674e-11
M_T = 5.972e24    # GM (m^3/s^2) si M_T = G*M_terre, sinon adapte
GM_T = G*M_T
R_T = 6.371e6

# --- Spacecraft Parameters ---

m_a = 20000.0     # masse à vide (kg)
m_f = 336000.0  - m_a      # masse de carburant initiale (kg)
Dm  = 9*300.0         # débit massique (kg/s)

V_e = 3.0e3
T_r = Dm * V_e     # magnitude de la poussée (N)

delta_V = V_e * np.log((m_a+m_f)/m_a)
print(delta_V)

t_burnout = m_f / Dm  # instant où le carburant est épuisé
print(t_burnout)

# --- Objectives ---

h = 200e3 #m
R_desired = R_T + h

v_circ = np.sqrt(GM_T/R_desired)

# --- Simulation helpers ---

def masse(t):
    if t < t_burnout:
        return m_a + (m_f - Dm * t)
    return m_a

def poussee(t):
    # coupe la poussée une fois le carburant épuisé
    return T_r if t < t_burnout else 0.0

def phi_of_t(t, U, t_breaks):
    # t_breaks : les N+1 bornes des segments, ex np.linspace(0, t_burnout, N+1)
    if t >= t_breaks[-1]:
        return U[-1]
    k = np.searchsorted(t_breaks, t, side='right') - 1
    k = np.clip(k, 0, len(U)-1)
    return U[k]

def f(t, y, U, t_breaks):
    r, rdot, theta, thetadot = y
    m = masse(t)
    T = poussee(t)
    phi = phi_of_t(t, U, t_breaks)

    r_ddot     = (T / m) * np.cos(phi) + r * thetadot**2 - GM_T / r**2
    theta_ddot = (T * np.sin(phi)) / (r * m) - 2*(rdot * thetadot) / r

    return [rdot, r_ddot, thetadot, theta_ddot]

def J(U):

    sol = solve_ivp(f, (0, t_burnout), y0, args=(U, t_breaks),
                     method='RK45', rtol=1e-9, atol=1e-9)
    
    r_T, rdot_T, theta_T, thetadot_T = sol.y[:, -1]
    
    cout = ( w1*(r_T - R_desired)**2
           + w2*(rdot_T)**2
           + w3*(r_T*thetadot_T - v_circ)**2)
    
    return cout

w1, w2, w3 = 2/R_desired**2, 10/v_circ**2 , 5/v_circ**2 

# --- Solving ---

N = 5
U0 = np.arange(0,N,1) * np.pi/180
y0 = [6.371e6, 0, 0, 0] 
t_breaks = np.linspace(0, t_burnout, N+1)

res = minimize(J, U0, method='L-BFGS-B',
                bounds=[(0, np.pi/2)] * N)

phi_opt = res.x
print(res.fun, res.success)
# --- Simulation with solver results ---

t_span = (0, 5000)                

sol = solve_ivp(f, t_span, y0, args=(phi_opt, t_breaks),
                 method='RK45', rtol=1e-9, atol=1e-9)

r, rdot, theta, thetadot = sol.y


# --- Results ---

# Changement de repère : polaire -> cartésien
x = r * np.cos(theta)
y = r * np.sin(theta)

# --- Disque (cercle plein de rayon R) ---
theta_disque = np.linspace(0, 2 * np.pi, 200)
r_disque = np.full_like(theta_disque, R_T)

x_disque = r_disque * np.cos(theta_disque)
y_disque = r_disque * np.sin(theta_disque)

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

v = np.sqrt(rdot**2 + (r*thetadot)**2)   # vitesse totale

# construire les segments consécutifs [(x0,y0)-(x1,y1)], [(x1,y1)-(x2,y2)], ...
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

fig, ax = plt.subplots()

norm = Normalize(v.min(), v.max())
lc = LineCollection(segments, cmap='viridis', norm=norm)
lc.set_array(v[:-1])     # une valeur de vitesse par segment
lc.set_linewidth(2)

line = ax.add_collection(lc)
fig.colorbar(line, ax=ax, label="vitesse (m/s)")

ax.fill(x_disque, y_disque, alpha=0.3, color='orange', label="disque")

ax.set_aspect('equal')
ax.axhline(0, color='gray', lw=0.5)
ax.axvline(0, color='gray', lw=0.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Trajectoire colorée par vitesse")
ax.autoscale()   # add_collection ne met pas à jour les limites automatiquement
ax.legend()

plt.show()

