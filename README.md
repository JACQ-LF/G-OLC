# G-OLC
G-OLC (Ground to Orbit Launch Calculator)

cartesian refrence frame R0(O,e_x,e_y,e_z)
polar refrence frame (spacecraft)  R1(M,e_r,e_theta,e_phi)

thus : OM vector is the spacecraft position in R0
OM = re_r

r(t) : radial position of the spacecraft
theta(t) : angle of the spacecraft relative to R0

M_T = Earth mass (or whatever space body)

V0 : EDO solver and basic setup (one stage preset steering law)
V0.1 : Steering optimization (fixed thrust one stage)
    Reach a desired orbit radius for a ground launch site
    Reach the desired orbit as fuel efficiently as possible (steering optimization)
    -> r(t_burnout) = R_desired, r_dot(t_burnout) = 0, r(t_burnout) * theta_dot(t_burnout) = sqrt(M_T/R_desired)
V0.2 : Optimal thrust (and all the previous)