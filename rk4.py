import numpy as np
import matplotlib.pyplot as plt
from utils import count_E

def rk4_step(y, t, dt):
    k1 = derivatives(y, t)
    k2 = derivatives(y + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = derivatives(y + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = derivatives(y + dt * k3, t + dt)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

MODE = int(input("1. Атмосфера + нецентральность поля т.\n2. Атмосфера.\n3. Нецентральность поля т.\n4. Без возмущений:\n"))
match(MODE):
    case 1:
        MODE = "__all__"
    case 2:
        MODE = "__drag__"
    case 3:
        MODE = "__J2__"
    case 4:
        MODE = "__NO__"
    case _:
        MODE = "__all__"

# --- 1. Константы ---
mu = 398600.4418e9       # [м^3/с^2] гравитационный параметр
R_e = 6378.136e3         # [м] экваториальный радиус Земли
J2  = 1.08263e-3         # коэффициент J2
rho0 = 1e-11             # [кг/м^3] плотность на опорной высоте
h0   = 600e3             # [м] опорная высота для rho0
H    = 75e3              # [м] масштабная высота

w_e = 7.29e-5            # [рад/с] угловая скорость Земли

# Параметры спутника
i0   = np.deg2rad(74.3)  # [рад] наклонение
ha   = 1120e3            # [м] апоцентр
hp   = 640e3             # [м] перицентр
Omega0 = np.deg2rad(60)  # [рад] долгота восходящего узла
u0     = np.deg2rad(50)  # [рад] начальное значение аргумента широты
Area   = 12.0             # [м2] площадь
m      = 1660.0          # [кг] масса
Cd     = 2.0             # коэффициент лобового сопротивления

rp = R_e + hp
ra = R_e + ha
a0 = 0.5*(rp + ra)
e0 = (ra - rp)/(ra + rp)
w0  = 0.0
nu0 = u0

n0 = np.sqrt(mu/a0**3)

# Время интегрирования
t_end = 2 * np.pi * np.sqrt(a0**3 / mu) * 30  # 3 витка
dt    = 10.0            # шаг интегрирования, с
Nsteps = int(t_end/dt)

# Массивы для записи данных
t_arr = np.zeros(Nsteps)
a_arr = np.zeros(Nsteps)
e_arr = np.zeros(Nsteps)
i_arr = np.zeros(Nsteps)
Omega_arr = np.zeros(Nsteps)
w_arr = np.zeros(Nsteps)
nu_arr = np.zeros(Nsteps)

S_arr = np.zeros(Nsteps)
T_arr = np.zeros(Nsteps)
W_arr = np.zeros(Nsteps)

# начальные условия
state = np.array([a0, e0, i0, Omega0, w0, nu0])

def derivatives(y, t):
    a, e, i, Omega, w, nu = y

    # --- Расчёт r через истинную аномалию ---
    E = count_E(nu, e)
    M = E - e*np.sin(E)
    p = a*(1 - e**2)
    r = a*(1 - e*np.cos(E))

    # --- Вычисление координат и скорости в орбитальной СК ---
    r_orb = np.array([r*np.cos(nu), r*np.sin(nu), 0])
    v_orb = np.array([
        -np.sqrt(mu/p)*np.sin(nu),
         np.sqrt(mu/p)*(e + np.cos(nu)),
         0
    ])

    # --- Матрица поворота в ИСО ---
    cosO = np.cos(Omega); sinO = np.sin(Omega)
    cosi = np.cos(i); sini = np.sin(i)
    cosw = np.cos(w); sinw = np.sin(w)

    RzO = np.array([[cosO, -sinO, 0], [sinO, cosO, 0], [0, 0, 1]])
    Rxi = np.array([[1, 0, 0], [0, cosi, -sini], [0, sini, cosi]])
    Rzw = np.array([[cosw, -sinw, 0], [sinw, cosw, 0], [0, 0, 1]])

    Q = RzO @ Rxi @ Rzw
    r_xyz = Q @ r_orb
    v_xyz = Q @ v_orb

    # --- Атмосферное торможение ---
    h = np.linalg.norm(r_xyz) - R_e
    rho = rho0 * np.exp(-(h - h0)/H)
    V = np.linalg.norm(v_xyz)
    a_drag = -0.5*Cd*(Area/m)*rho*V * v_xyz

    # --- J2 ускорение ---
    x, y, z = r_xyz
    r_norm = np.linalg.norm(r_xyz)
    fac = 1.5 * J2 * mu * R_e**2 / r_norm**5
    a_J2 = np.zeros(3)
    a_J2[0] = fac * x * (5 * (z**2 / r_norm**2) - 1)
    a_J2[1] = fac * y * (5 * (z**2 / r_norm**2) - 1)
    a_J2[2] = fac * z * (5 * (z**2 / r_norm**2) - 3)

    # --- Выбор возмущения ---
    match MODE:
        case "__all__":
            a_pert_ijk = a_drag + a_J2
        case "__drag__":
            a_pert_ijk = a_drag
        case "__J2__":
            a_pert_ijk = a_J2
        case "__NO__":
            a_pert_ijk = np.zeros(3)

    # --- Проекции в орбитальной системе ---
    ur = r_xyz / r_norm
    us = np.cross(np.cross(ur, v_xyz), ur)
    us /= np.linalg.norm(us)
    uw = np.cross(ur, us)

    S = np.dot(a_pert_ijk, ur)
    T = np.dot(a_pert_ijk, us)
    W = np.dot(a_pert_ijk, uw)

    p = a*(1 - e**2)

    da_dt = 2*np.sqrt(a**3/mu)/(np.sqrt(1-e*e)) * ( e*np.sin(nu)*S + (1+e*np.cos(nu))*T )
    de_dt = np.sqrt(p/mu)*( np.sin(nu)*S + ( (np.cos(nu)+np.cos(E)) )*T )
    di_dt = np.sqrt(p/mu)/(1+e*np.cos(nu))*np.cos(nu+w)*W
    dO_dt = np.sqrt(p/mu)/( (1+e*np.cos(nu)) * np.sin(i) )*np.sin(nu+w)*W
    dw_dt = np.sqrt(p/mu)/e * ( -np.cos(nu)*S + (2+e*np.cos(nu))/(1+e*np.cos(nu))*np.sin(nu)*T ) \
           - np.sqrt(p/mu)/( (1+e*np.cos(nu)) * np.tan(i) )*np.sin(nu+w)*W
    dnu_dt = np.sqrt(mu/p**3) * (1 + e*np.cos(nu))**2 + 1/e * np.sqrt(p/mu) \
           * (S*np.cos(nu) - T*np.sin(nu) * (2 + e*np.cos(nu)/(1+e*np.cos(nu))))

    return np.array([da_dt, de_dt, di_dt, dO_dt, dw_dt, dnu_dt])

# --- Интегрирование методом Рунге-Кутты ---
for k in range(Nsteps):
    t = k * dt
    state = rk4_step(state, t, dt)
    
    a, e, i, Omega, w, nu = state
    a_arr[k] = a
    e_arr[k] = e
    i_arr[k] = np.rad2deg(i)
    Omega_arr[k] = np.rad2deg(Omega)
    w_arr[k] = np.rad2deg(w)
    nu_arr[k] = np.rad2deg(nu)
    t_arr[k] = t / 3600

# --- График зависимости элементов орбиты от аргумента широты u = nu + w ---
u_arr = nu_arr + w_arr

plt.figure(figsize=(12, 8))

ax = plt.subplot(2, 3, 1)
ax.plot(u_arr, a_arr / 1e3, '.', ms=1)
ax.set_xlabel('u, deg')
ax.set_ylabel('a, км')
ax.grid()

ax = plt.subplot(2, 3, 2)
ax.plot(u_arr, e_arr, '.', ms=1)
ax.set_xlabel('u, deg')
ax.set_ylabel('e')
ax.grid()

ax = plt.subplot(2, 3, 3)
ax.plot(u_arr, i_arr, '.', ms=1)
ax.set_xlabel('u, deg')
ax.set_ylabel('i, deg')
ax.grid()

ax = plt.subplot(2, 3, 4)
ax.plot(u_arr, Omega_arr, '.', ms=1)
ax.set_xlabel('u, deg')
ax.set_ylabel('Ω, deg')
ax.grid()

ax = plt.subplot(2, 3, 5)
ax.plot(u_arr, w_arr, '.', ms=1)
ax.set_xlabel('u, deg')
ax.set_ylabel('ω, deg')
ax.grid()

plt.tight_layout()
plt.show()