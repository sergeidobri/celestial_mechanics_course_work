import numpy as np
import matplotlib.pyplot as plt
from utils import count_E

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
alpha = 1/298.25         # полярное сжатие Земли
rho0 = 1e-11             # [кг/м^3] плотность на опорной высоте
h0   = 600e3             # [м] опорная высота для rho0
H    = 75e3              # [м] масштабная высота

w_e = 7.29e-5            # [рад/с] угловая скорость Земли
g_e = mu / R_e**2        # [м/c^2] ускорение свободного падения на экваторе

# Параметры спутника
i0   = np.deg2rad(74.3)  # [рад] наклонение
ha   = 1120e3            # [м] апоцентр
hp   = 640e3             # [м] перицентр
Omega0 = np.deg2rad(60)  # [рад] долгота восходящего узла
u0     = np.deg2rad(50)  # [рад] аргумент широты -- считаем истинной аномалией nu0
Area      = 12.0         # [м2] площадь
m      = 1660.0          # [кг] масса
Cd     = 2.0             # коэффициент лобового сопротивления

rp = R_e + hp
ra = R_e + ha
a0 = 0.5*(rp + ra)
e0 = (ra - rp)/(ra + rp)
w0  = 0.0
nu0  = u0

n0 = np.sqrt(mu/a0**3)
E0 = count_E(nu0, e0)
M0 = E0 - e0*np.sin(E0)

# Время интегрирования

t_end = 30*24*3600.0
dt    = 10.0            # шаг интегрирования, с
Nsteps = int(t_end/dt)

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
a=a0; e=e0; i=i0; Omega=Omega0; w=w0; M=M0
nu = nu0

q = w_e**2 * R_e / g_e   # величина для подсчета возмущений, вызванных нецентральностью поля тяжести Земли

for k in range(Nsteps):
    if nu >= nu0 + 2 * np.pi: break  # период
    t = k*dt

    t_arr[k] = t/3600.0  # [ч]
    a_arr[k] = a
    e_arr[k] = e
    i_arr[k] = np.rad2deg(i)
    Omega_arr[k] = np.rad2deg(Omega)
    w_arr[k] = np.rad2deg(w)
    nu_arr[k] = np.rad2deg(nu)

    # --- 2. вычислим координаты и скорость в орбитальной системе координат ---
    
    E = count_E(nu, e)
    r = a*(1 - e*np.cos(E))
    p = a*(1 - e**2)

    # орбитальные координаты в орбитальной СК
    r_orb = np.array([r*np.cos(nu), r*np.sin(nu), 0.0])
    v_orb = np.array([
        -np.sqrt(mu/p)*np.sin(nu),  # направлена в центр, против \vec{r}
         np.sqrt(mu/p)*(e + np.cos(nu)),
         0.0
    ])

    # матрица перехода в АГЭСК (XYZ)
    cosO = np.cos(Omega); sinO = np.sin(Omega)
    cosi = np.cos(i);      sini = np.sin(i)
    cosw = np.cos(w);      sinw = np.sin(w)
    # R = Rz(Omega)*Rx(i)*Rz(w)
    RzO = np.array([
        [cosO,-sinO,0],
        [sinO,cosO,0],
        [0,0,1]
    ])
    Rxi = np.array([
        [1,0,0],
        [0,cosi,-sini],
        [0,sini,cosi]
    ])
    Rzw = np.array([
        [cosw,-sinw,0],
        [sinw,cosw,0],
        [0,0,1]
    ])
    Q = RzO @ Rxi @ Rzw
    r_xyz = Q @ r_orb
    v_xyz = Q @ v_orb

    # считаем неподвижную атмосферу в ИСО
    V  = np.linalg.norm(v_xyz)

    # --- 3. атмосферное торможение ---
    h = np.linalg.norm(r_xyz) - R_e
    rho = rho0 * np.exp(-(h-h0)/H)
    a_drag = -0.5*Cd*(Area/m)*rho*V * v_xyz

    # --- 4. J2 ускорение в ИСО ---
    x,y,z = r_xyz
    r_norm = np.linalg.norm(r_xyz)

    fac = 1.5*J2*mu*R_e**2/r_norm**5  # общая часть (множитель) для уравнений возмущений
    a_J2 = np.zeros(3)

    a_J2[0] = fac * x*(5*(z**2/r_norm**2)-1)
    a_J2[1] = fac * y*(5*(z**2/r_norm**2)-1)
    a_J2[2] = fac * z*(5*(z**2/r_norm**2)-3)

    match MODE:
        case "__all__":
            a_pert_ijk = a_drag + a_J2
        case "__drag__":
            a_pert_ijk = a_drag
        case "__J2__":
            a_pert_ijk = a_J2
        case "__NO__":
            a_pert_ijk = np.zeros(3)

    # --- 5. проекция возмущения в орбитальной системе координат  ---
    ur = r_xyz / np.linalg.norm(r_xyz)       # орт r-оси
    us = np.cross(np.cross(ur, v_xyz), ur)   # орт nu-оси (трансверсальной)
    us /= np.linalg.norm(us)  # |us|=1
    uw = np.cross(ur, us)                    # орт бинормали

    S = np.dot(a_pert_ijk, ur)              # проекция на r-ось
    T = np.dot(a_pert_ijk, us)              # проекция на nu-ось
    W = np.dot(a_pert_ijk, uw)              # проекция на бинормаль

    S_arr[k] = S
    T_arr[k] = T
    W_arr[k] = W

    # --- 6. Гауссовы уравнения для эллиптической орбиты ---
    p = a*(1-e**2)
    da_dt = 2*np.sqrt(a**3/mu)/(np.sqrt(1-e*e)) * ( e*np.sin(nu)*S + (1+e*np.cos(nu))*T )
    de_dt = np.sqrt(p/mu)*( np.sin(nu)*S + ( (np.cos(nu)+np.cos(E)) )*T )
    di_dt = np.sqrt(p/mu)/(1+e*np.cos(nu))*np.cos(nu+w)*W
    dO_dt = np.sqrt(p/mu)/( (1+e*np.cos(nu)) * np.sin(i) )*np.sin(nu+w)*W
    dw_dt = np.sqrt(p/mu)/e * ( -np.cos(nu)*S + (2+e*np.cos(nu))/(1+e*np.cos(nu))*np.sin(nu)*T ) \
            - np.sqrt(p/mu)/( (1+e*np.cos(nu)) * np.tan(i) )*np.sin(nu+w)*W
    dnu_dt = np.sqrt(mu/p**3) * (1 + e*np.cos(nu))**2 + 1/e * np.sqrt(p/mu) \
            * (S*np.cos(nu) - T*np.sin(nu) * (2 + e*np.cos(nu)/(1+e*np.cos(nu))))

    # --- 7. шаг интегрирования (Эйлер) ---
    a     += da_dt*dt
    e     += de_dt*dt
    i     += di_dt*dt
    Omega += dO_dt*dt
    w     += dw_dt*dt
    nu    += dnu_dt*dt

# --- 8. Построим графики ---
# fig1 = plt.figure(figsize=(12, 6))

# ax1 = fig1.add_subplot(1, 2, 1)
# ax1.scatter(t_arr, S_arr, s=2, alpha=0.5)
# ax1.set_xlabel('t, ч')
# ax1.set_ylabel('S, км/с²')
# ax1.grid(True)

# ax2 = fig1.add_subplot(1, 2, 2)
# ax2.scatter(t_arr, T_arr, s=2, alpha=0.5)
# ax2.set_xlabel('t, ч')
# ax2.set_ylabel('T, км/с²')
# ax2.grid(True)

# plt.tight_layout()
# plt.show()   # блокирует выполнение до закрытия первого окна

# # === Вторая фигура: W и F ===
# fig2 = plt.figure(figsize=(12, 6))

# ax3 = fig2.add_subplot(1, 2, 1)
# ax3.scatter(t_arr, W_arr, s=2, alpha=0.5)
# ax3.set_xlabel('t, ч')
# ax3.set_ylabel('W, км/с²')
# ax3.grid(True)

# # Расчёт F
# F_arr = np.sqrt(S_arr**2 + T_arr**2 + W_arr**2)
# ax4 = fig2.add_subplot(1, 2, 2)
# ax4.scatter(t_arr, F_arr, s=2, alpha=0.5)
# ax4.set_xlabel('t, ч')
# ax4.set_ylabel('F, км/с²')
# ax4.grid(True)

# plt.tight_layout()
# plt.show()


# u = ν + ω (в градусах)
u_arr = nu_arr + w_arr
u_rad = np.deg2rad(u_arr)

# Первые 2 графика
plt.figure(figsize=(10, 4))

ax = plt.subplot(1, 2, 1)
ax.scatter(u_arr, a_arr / 1e3, s=2, alpha=0.5)
ax.set_xlabel('u, deg')
ax.set_ylabel('a, км')
ax.grid()

ax = plt.subplot(1, 2, 2)
ax.scatter(u_arr, e_arr, s=2, alpha=0.5)
ax.set_xlabel('u, deg')
ax.set_ylabel('e')
ax.grid()

plt.tight_layout()
plt.show()   # ждём закрытия первого окна

# Следующие 2 графика
plt.figure(figsize=(10, 4))

ax = plt.subplot(1, 2, 1)
ax.scatter(u_arr, i_arr, s=2, alpha=0.5)
ax.set_xlabel('u, deg')
ax.set_ylabel('i, deg')
ax.grid()

ax = plt.subplot(1, 2, 2)
ax.scatter(u_arr, Omega_arr, s=2, alpha=0.5)
ax.set_xlabel('u, deg')
ax.set_ylabel('Ω, deg')
ax.grid()

plt.tight_layout()
plt.show()   # ждём закрытия второго окна

# Последние 2 графика
plt.figure(figsize=(10, 4))

ax = plt.subplot(1, 2, 1)
ax.scatter(u_arr, w_arr, s=2, alpha=0.5)
ax.set_xlabel('u, deg')
ax.set_ylabel('ω, deg')
ax.grid()

ax = plt.subplot(1, 2, 2)
ax.scatter(u_arr, nu_arr, s=2, alpha=0.5)
ax.set_xlabel('u, deg')
ax.set_ylabel('ν, deg')
ax.grid()

plt.tight_layout()
plt.show()
