# Ideal GAS model
#%matplotlib inline
#from IPython.display import HTML
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.linalg import norm
np.random.seed(0)

# Macro parameters
x0 = 0; x1 = 16
y0 = 0; y1 = 16
# TODO set to 8000 for evaluation
N = 800
E = 30000
L = 10
period = 10
SAVEFIG = False
MAKEANIM = True
MAXVELL = True
ENANLEMIX = 'maxwell'
ENABLETURN = False
ENABLECOL = True

if SAVEFIG:
    plt.ioff()

# State
m = np.random.choice([0.8, 1.2], size=(N, 1))
r = 0.01
X = np.random.uniform((x0, y0), (x0 + 1, y0 + 1), (N, 2))
# TODO understand velocity distribution
if 1:#MAXVELL:
    V = 8 * np.random.randn(N, 2)
else:
    V = 30 * (np.random.rand(N, 2) - 0.5)

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
Ef_const = m[:, 0] * X[:, 1] * (-g[1]) + (m * V * V / 2).sum(1)
dE = (Ef_const.mean() + 4 * Ef_const.std()) / L
dE2 = (np.sqrt(Ef_const).mean() + 4 * np.sqrt(Ef_const).std()) / L
dh = (y1 - y0) / L

for epoch in range(E):
    X_n = X + V * dt + 0.5 * g * dt ** 2
    V = V + g * dt

    # Fix velocities to return particles in volume
    Ix = (X_n[:, 0] < x0) + (X_n[:, 0] > x1)
    Iy = (X_n[:, 1] < y0) + (X_n[:, 1] > y1)
    Iy0 = X_n[:, 1] < y0
    Iy1 = X_n[:, 1] > y1

    V[Ix, 0] *= -1
    V[Iy0, 1] = np.abs(V[Iy0, 1])
    V[Iy1, 1] = -np.abs(V[Iy1, 1])

    if 0:#ENANLEMIX == 'uniform' and Iy0.sum() >= 2:
        alpha = 0.4
        n = Iy0.sum()

        # Storing kinetic energy of affected particles
        E = np.linalg.norm(V[Iy0], axis=1, keepdims=True) ** 2

        # Creating random mixing matrix
        M = np.random.uniform(low=0, high=1, size=(n, n))
        M = M / M.sum(axis=1, keepdims=True)
        M = np.eye(n) * (1 - alpha) + alpha * M

        E_n = np.dot(M, E)
        E_n = E_n * (E.sum() / E_n.sum())

        V[Iy0] = (V[Iy0] / np.sqrt(E)) * np.sqrt(E_n)

    if 0:#ENANLEMIX == 'maxwell' and Iy0.sum() >= 2:
        alpha = 0.4
        n = Iy0.sum()

        # Storing kinetic energy of affected particles
        E = m[Iy0] * np.linalg.norm(V[Iy0], axis=1, keepdims=True) ** 2

        E_n = np.clip(E.std() * np.random.randn(n, 1) + E.mean(), a_min=0, a_max=1000000)
        E_n = E_n * (E.sum() / E_n.sum())
        E_n = E * (1 - alpha) + E_n * alpha
        E_n = E_n * (E.sum() / E_n.sum())

        V[Iy0] = (V[Iy0] / np.sqrt(E)) * np.sqrt(E_n)

        if (V != V).any():
            assert 0

    if 0:#ENABLETURN:
        eps = 1.0
        # Find situation when particles are just in inner volume
        I_n_pos_in = (X_n[:, 0] > (x0 + eps)) * (X_n[:, 0] < (x1 - eps)) * (X_n[:, 1] > (y0 + eps)) * (X_n[:, 1] < (y1 - eps))
        I_o_pos_not = ~((X[:, 0] > (x0 + eps)) * (X[:, 0] < (x1 - eps)) * (X[:, 1] > (y0 + eps)) * (X[:, 1] < (y1 - eps)))
        I_rr = I_n_pos_in * I_o_pos_not

        ralpha = np.random.uniform(low=-0.1, high=0.1, size=(I_rr).sum())
        Rrm = np.array([[np.cos(ralpha), -np.sin(ralpha)],
                        [np.sin(ralpha), np.cos(ralpha)]]).transpose([2, 0, 1])

        # Calculate potential new velocities
        V[I_rr] = np.stack([(V[I_rr] * Rrm[..., 0]).sum(1),
                            (V[I_rr] * Rrm[..., 1]).sum(1)], axis=1)

        if (V != V).any():
            assert 0

    if 1:#ENABLECOL:
        D = np.linalg.norm(X_n[:, np.newaxis, :] - X_n[np.newaxis, ...], axis=2, keepdims=False)
        num_of_col = ((D < 2 * r).sum() - N) // 2
        # Here get nearest particle to collision with
        D = D + np.eye(N) * 1000
        N_dist = D.min(axis=1)
        N_part = D.argmin(axis=1)
        I = (N_dist < 2 * r)

        if I.sum() != 0:
            P_a = np.arange(N)[I]
            P_b = N_part[I]

            # Ensure uniqueness
            P_ab = np.sort(np.stack([P_a, P_b], axis=1), axis=1)
            P_ab, C = np.unique(P_ab, axis=0, return_counts=True)
            P_ab = P_ab[C == 2]

            P_a = np.concatenate([P_ab[:, 0], P_ab[:, 1]])
            P_b = np.concatenate([P_ab[:, 1], P_ab[:, 0]])

            # No duplicates!
            assert len(np.unique(P_b)) == len(P_b)

            m_a = m[P_a]
            m_b = m[P_b]
            v_a = V[P_a]
            v_b = V[P_b]

            # Calculate energy before
            E_k_ab = m_a[:, 0] * (np.linalg.norm(V[P_a], axis=1) ** 2) + m_b[:, 0] * (np.linalg.norm(V[P_b], axis=1) ** 2)

            V[P_a] = ((m_a - m_b) * v_a + 2 * m_b * v_b)/(m_a + m_b)
            V[P_b] = ((m_b - m_a) * v_b + 2 * m_a * v_a)/(m_a + m_b)

            # Calculate energy after
            E_k_ab_new = m_a[:, 0] * (np.linalg.norm(V[P_a], axis=1) ** 2) + m_b[:, 0] * (np.linalg.norm(V[P_b], axis=1) ** 2)

            # Energy levels must mach
            assert ((E_k_ab / E_k_ab_new).round(14) != 1.0).sum() == 0

            if epoch % period == 0:
                print('Number collisions:', num_of_col, '/', len(P_a)//2, 'mean:', X_n[P_a, 1].mean(), 'max:', X_n[P_a, 1].max())

    X = X_n

    if epoch % period == 0:
        if (V != V).any():
            assert 0

        print(epoch)
        # Store statistics
        x_data.append(X)
        v_data.append(V)
        Ek_data.append((m * V * V / 2).sum(1))
        Ep_data.append(m[:, 0] * X[:, 1] * (-g[1]))

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

print('Stalled particles:', (x_data[-234:, :, 1].max(axis=0) < 0).sum())

# Print dT/dh = const * dE(Ek)/dh of stationary state
Levs = 4
stEp = len(Ek_data) // 6
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
print('N_whole', N_whole, N_whole.sum())

fig, ax1 = plt.subplots()

ax1.plot(N_whole, color='tab:blue')
ax1.set_xlabel('h level')
ax1.set_ylabel('Num of particles')
ax2 = ax1.twinx()
ax2.plot(T_whole, color='tab:red')
ax2.set_ylabel('T or average kinetic energy of particles')
fig.tight_layout()

if SAVEFIG:
    plt.savefig('T_grad.png')
    plt.close()
else:
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

if SAVEFIG:
    plt.savefig('Ek_Ep_Ef_cor_height.png')
    plt.close()
else:
    plt.show()

plt.ylim(0, Ef_const.sum() * 2)
plt.plot(Ep_data.sum(1) + Ek_data.sum(1))
plt.plot(Ef_const.sum().repeat(len(Ep_data)))
plt.legend(['Current full energy', 'Old full energy'])

if SAVEFIG:
    plt.savefig('ConservationOfEnergy_check.png')
    plt.close()
else:
    plt.show()

plt.plot(Hkh_data)
plt.plot(Hk2h_data)
plt.plot(Hph_data)
plt.plot(Hfh_data)
plt.ylabel('H (Entropy)')
plt.legend(['H(Ek, h)', 'H(v, h)', 'H(Ep, h)', 'H(Ef, h)'])
plt.xlabel('time')

if SAVEFIG:
    plt.savefig('SoCalledEntropyCheck.png')
    plt.close()
else:
    plt.show()

fig, _ = plt.subplots()
plt.xlim(x0-1, x1+1)
plt.ylim(y0-1, y1+1)

if not MAKEANIM:
    if not SAVEFIG:
        plt.show()
    exit()

scatter = plt.scatter(x_data[0, :, 0], x_data[0, :, 1], s=4)
def animate(i):
    scatter.set_offsets(x_data[i])
    scatter.set_array(Ek_data[i] + Ep_data[i])
    return scatter,

a = FuncAnimation(fig, animate, frames=len(x_data), interval=20, blit=True, repeat=True)
plt.show()