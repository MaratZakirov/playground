# Ideal GAS model
#%matplotlib inline
#from IPython.display import HTML
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Macro parameters
x0 = 0; x1 = 16
y0 = 0; y1 = 16
# TODO set to 8000 for evaluation
N = 4000
E = 3700
L = 10

# State
X = np.random.uniform((x0, y0), (x0 + 1, y0 + 1), (N, 2))
# TODO understand velocity distribution
V = 6 * np.random.randn(N, 2)
#V = 30 * (np.random.rand(N, 2) - 0.5)

g = np.array([0, -9.8])
dt = 0.01

lines = np.array([[x0, y0, x1, y0],
                  [x0, y1, x1, y1],
                  [x0, y0, x0, y1],
                  [x1, y0, x1, y1]])

# For statistics storing
x_data = []
v_data = []
Ek_data = []
Ep_data = []
Hkh_data = []
Hk2h_data = []
Hph_data = []
Hfh_data = []

# Every particle in ideal gas model has same amount of energy
Ef_const = X[:, 1] * (-g[1]) + (V * V / 2).sum(1)
dE = (Ef_const.mean() + 4 * Ef_const.std()) / L
dE2 = (np.sqrt(Ef_const).mean() + 4 * np.sqrt(Ef_const).std()) / L
dh = (y1 - y0) / L

for epoch in range(E):
    X_n = X + V * dt + 0.5 * g * dt ** 2
    V = V + g * dt

    # Fix velocities to return particles in volume
    V[(X_n[:, 0] < x0) + (X_n[:, 0] > x1), 0] *= -1
    V[(X_n[:, 1] < y0) + (X_n[:, 1] > y1), 1] *= -1

    X = X_n

    # Store statistics
    x_data.append(X)
    v_data.append(V)
    Ek_data.append((V * V / 2).sum(1))
    Ep_data.append(X[:, 1] * (-g[1]))

    # Calculating entropy
    Ek_lev = np.clip(Ek_data[-1] / dE, 0, L - 1).astype(int)
    Ek2_lev = np.clip(np.sqrt(Ek_data[-1]) / dE2, 0, L - 1).astype(int)
    Ep_lev = np.clip(Ep_data[-1] / dE, 0, L - 1).astype(int)
    Ef_lev = np.clip((Ep_data[-1] + Ek_data[-1]) / dE, 0, L - 1).astype(int)
    h_lev = np.clip((X[:, 1] - y0) / dh, 0, L - 1).astype(int)
    p_kh = np.bincount(h_lev * L + Ek_lev, minlength = L * L) / N
    p_k2h = np.bincount(h_lev * L + Ek2_lev, minlength = L * L) / N
    p_ph = np.bincount(h_lev * L + Ep_lev, minlength=L * L) / N
    p_pf = np.bincount(h_lev * L + Ef_lev, minlength=L * L) / N
    Hkh_data.append(-np.sum(p_kh * np.log(p_kh + 0.00001)))
    Hk2h_data.append(-np.sum(p_k2h * np.log(p_k2h + 0.00001)))
    Hph_data.append(-np.sum(p_ph * np.log(p_ph + 0.00001)))
    Hfh_data.append(-np.sum(p_pf * np.log(p_pf + 0.00001)))

x_data = np.stack(x_data)
v_data = np.stack(v_data)
Ek_data = np.array(Ek_data)
Ep_data = np.array(Ep_data)
Hkh_data = np.array(Hkh_data)
Hk2h_data = np.array(Hk2h_data)
Hph_data = np.array(Hph_data)
Hfh_data = np.array(Hfh_data)

# Print dT/dh = const * dE(Ek)/dh of stationary state
Levs = 6
stEp = 1000
Ek0 = Ek_data[-stEp:]
h0 = x_data[-stEp:, :, 1]

T_whole = []
N_whole = []
for e in range(stEp):
    N_cur = np.zeros(Levs)
    T_cur = np.zeros(Levs)

    Ek0_cur = Ek0[e]
    h0_cur = (Levs * (h0[e] + 0.2) / (y1 - y0 + 1)).astype(int)

    for l in range(Levs):
        if (h0_cur == l).sum() > 0:
            T_cur[l] = Ek0_cur[h0_cur == l].mean()
            N_cur[l] = (h0_cur == l).sum()

    N_whole.append(N_cur)
    T_whole.append(T_cur)

N_whole = np.stack(N_whole).mean(0)
T_whole = np.stack(T_whole).mean(0)

print('T_whole', T_whole)
print('N_whole', N_whole)

fig, ax1 = plt.subplots()

ax1.plot(N_whole, color='tab:blue')
ax1.set_xlabel('h level')
ax1.set_ylabel('Num of particles')
ax2 = ax1.twinx()
ax2.plot(T_whole, color='tab:red')
ax2.set_ylabel('T or average kinetic energy of particles')
fig.tight_layout()
plt.show()

# Print correlation
Ek = Ek_data
Ep = Ep_data
Ef = Ek_data + Ep_data
h = x_data[..., 1]
rk = ((Ek - Ek.mean(1, keepdims=True)) * (h - h.mean(1, keepdims=True))).mean(1) / Ek.std(1) / h.std(1)
rp = ((Ep - Ep.mean(1, keepdims=True)) * (h - h.mean(1, keepdims=True))).mean(1) / Ep.std(1) / h.std(1)
rf = ((Ef - Ef.mean(1, keepdims=True)) * (h - h.mean(1, keepdims=True))).mean(1) / Ef.std(1) / h.std(1)
plt.plot(rk)
plt.plot(rp)
plt.plot(rf)
plt.legend(['Ek with h', 'Ep with h', 'Ef with h'])
plt.show()

plt.ylim(0, Ef_const.sum() * 2)
plt.plot(Ep_data.sum(1) + Ek_data.sum(1))
plt.plot(Ef_const.sum().repeat(len(Ep_data)))
plt.legend(['Current full energy', 'Old full energy'])
plt.show()

plt.plot(Hkh_data)
plt.plot(Hk2h_data)
plt.plot(Hph_data)
plt.plot(Hfh_data)
plt.ylabel('H (Entropy)')
plt.legend(['H(Ek, h)', 'H(v, h)', 'H(Ep, h)', 'H(Ef, h)'])
plt.xlabel('time')
plt.show()

fig, _ = plt.subplots()
plt.xlim(x0, x1)
plt.ylim(y0, y1)

scatter = plt.scatter(x_data[0, :, 0], x_data[0, :, 1], s=4)
def animate(i):
    scatter.set_offsets(x_data[i])
    scatter.set_array(Ek_data[i] + Ep_data[i])
    return scatter,

a = FuncAnimation(fig, animate, frames=len(x_data), interval=20, blit=True, repeat=True)
plt.show()