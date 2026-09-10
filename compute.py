# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:18:39 2026
@author: Julien
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Physical Parameters ---
G = 6.674e-11
M_T = 5.972e24
GM_T = G * M_T
R_T = 6.371e6

T0 = 288.15 #Teméprature au plancher des vcahes
P0 = 101325 #Pressino au plancher des vaches
M0 = 28.975e-3 #Masse molarie de l'air
R = 8.314 #Constante de gaz parfaits

# --- Spacecraft Parameters ---
m_a = 10000.0       # masse à vide (kg)
m_f = 395000.0       # masse de carburant initiale (kg)
Dm0 = 10*350.0  # débit massique (kg/s)
Dmin = Dm0 / 100
Dmax = Dm0
V_e = 3.5e3
C_f = 0.01
S = 16 * np.pi

# --- Paramètres (à remplacer par tes vraies valeurs) ---
phi_dot = np.pi/180.0 *0.36#°/s

delta_V = V_e * np.log((m_a+m_f)/m_a)
print(delta_V)

t_burnout = m_f / Dm0  # instant où le carburant est épuisé

print(t_burnout)

def air_pressure(r):
    if r >= R_T + 100e3:
        return 0
    elif r < R_T:
        print("crash - pression atmosphérique 'infinie'.")
        return 1e9
    else :
        g = G*M_T/(r**2)
        return P0*np.exp(-M0*g/(R*air_temp(r)) * (r - R_T))


def air_temp(r):
    h = r - R_T
    if h <= 0:
        return T0
    elif h <= 11019:
        return T0 - 0.0065 * h
    elif h <= 20063:
        return air_temp(R_T + 11019)
    elif h <= 32162:
        return air_temp(R_T + 20063) + 0.001 * (h - 20063)
    elif h <= 47350:
        return air_temp(R_T + 32162) + 0.0028 * (h - 32162)
    elif h <= 51413:
        return air_temp(R_T + 47350)
    elif h <= 71802:
        return air_temp(R_T + 51413) - 0.0028 * (h - 51413)
    elif h <= 86000:
        return air_temp(R_T + 71802) - 0.002 * (h - 71802)
    else:
        return air_temp(R_T + 86000)

# --- Paramètre cible ---
a_cible = 3 * 9.81   # accélération que tu veux maintenir (m/s^2) — à ajuster

def Dm(m, a_target=a_cible):
    """Débit massique nécessaire pour maintenir l'accélération a_target,
    borné entre Dmin et Dmax."""
    Dm_needed = a_target * m / V_e
    return np.clip(Dm_needed, Dmin, Dmax)


def poussee(Dm_val):
    return Dm_val * V_e


def phi_t(t):
    return min(np.sqrt(phi_dot * t + 1), np.pi / 2)


def drag(r, rdot, thetadot, m):
    v_r = rdot
    v_theta = r * thetadot
    v = np.hypot(v_r, v_theta)
    rho = M0 * air_pressure(r) / (R * air_temp(r))
    return -C_f * S / (2 * m) * rho * v


def f(t, y):
    r, rdot, theta, thetadot, m = y          # <-- masse ajoutée comme état

    # tant qu'il reste du carburant, on pousse ; sinon Dm=0
    if m > m_a:
        Dm_t = Dm(m)
    else:
        Dm_t = 0.0

    T = poussee(Dm_t)
    phi = phi_t(t)
    d = drag(r, rdot, thetadot, m)

    r_ddot     = (T / m) * np.cos(phi) + r * thetadot**2 - GM_T / r**2 + d * rdot
    theta_ddot = (T * np.sin(phi)) / (r * m) - 2 * (rdot * thetadot) / r + d * thetadot
    m_dot      = -Dm_t                        # masse diminue avec le débit

    return [rdot, r_ddot, thetadot, theta_ddot, m_dot]


# Conditions initiales : [r0, rdot0, theta0, thetadot0, m0]
y0 = [R_T, 0, 0, 0, m_a + m_f]
t_span = (0, 10000)


def fuel_out(t, y):
    return y[4] - m_a          # s'annule quand la masse atteint m_a
fuel_out.terminal = False       # ne stoppe pas l'intégration
fuel_out.direction = -1         # ne détecte que la décroissance

def crash(t, y):
    return y[0] - R_T        # r - R_T s'annule au sol
crash.terminal = True
crash.direction = -1         # seulement quand r décroît en dessous de R_T

sol = solve_ivp(f, t_span, y0, method='RK45',
                 rtol=1e-9, atol=1e-9,
                 events=[fuel_out, crash])

if sol.t_events[1].size > 0:
    print(f"Crash détecté à t = {sol.t_events[1][0]:.2f} s")
elif sol.t_events[0].size > 0:
    print(f"Carburant épuisé à t = {sol.t_events[0][0]:.2f} s (intégration arrêtée)")
    
r, rdot, theta, thetadot, m = sol.y

# --- Changement de repère : polaire -> cartésien ---
x = r * np.cos(theta)
y = r * np.sin(theta)

# --- Disque (cercle plein de rayon R_T) ---
theta_disque = np.linspace(0, 2 * np.pi, 200)
r_disque = np.full_like(theta_disque, R_T)
x_disque = r_disque * np.cos(theta_disque)
y_disque = r_disque * np.sin(theta_disque)

# --- Recalcul de Dm(t) a posteriori ---
Dm_t_array = np.array([
    Dm(mi) if mi > m_a else 0.0
    for mi in m
])

# --- Tracé combiné avec subplots ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# Sous-plot 1 : trajectoire
ax1.plot(x, y, label="courbe", color='blue')
ax1.fill(x_disque, y_disque, alpha=0.3, color='orange', label="disque")
ax1.set_aspect('equal')
ax1.axhline(0, color='gray', lw=0.5)
ax1.axvline(0, color='gray', lw=0.5)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_title("Changement de repère polaire → cartésien")
ax1.legend()

# Sous-plot 2 : débit massique en fonction du temps
ax2.plot(sol.t, Dm_t_array)
ax2.axhline(Dmax, color='green', linestyle='--', label="Dmax")
ax2.axhline(Dmin, color='orange', linestyle='--', label="Dmin")
if sol.t_events[0].size > 0:
    ax2.axvline(sol.t_events[0][0], color='gray', linestyle=':', label="fin de combustion")
ax2.set_xlabel("Temps (s)")
ax2.set_ylabel("Débit massique Dm (kg/s)")
ax2.set_title("Débit massique en fonction du temps")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()