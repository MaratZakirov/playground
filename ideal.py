import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
np.random.seed(0)

"""
a = np.array([3, 3, 3, 0, 0, 0, 1, 2, 12])
x,y,z = np.unique(a, return_counts=True, return_index=True)
print(x, y, z)
print(np.repeat(x, z))
"""

from ctypes import cdll
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

cpputils = cdll.LoadLibrary("./cpputils.so")

# Call this function in C++
# int * GetCuts(int node_num, int edge_num, int * nodes_from, int * nodes_to, float * weigh)
def getColPairs(Xint):
    N = len(Xint)
    x = np.copy(Xint[:, 0])
    y = np.copy(Xint[:, 1])
    Pa = np.zeros(N, dtype=np.int32) - 1
    Pb = np.zeros(N, dtype=np.int32) - 1
    nums = np.arange(N, dtype=np.int32)
    np.random.shuffle(nums)
    cpputils.getColPairs.argtypes = [ctypes.c_int,
                                     ndpointer(dtype=ctypes.c_int, shape=(N,)),
                                     ndpointer(dtype=ctypes.c_int, shape=(N,)),
                                     ndpointer(dtype=ctypes.c_int, shape=(N,)),
                                     ndpointer(dtype=ctypes.c_int, shape=(N,)),
                                     ndpointer(dtype=ctypes.c_int, shape=(N,))]

    cpputils.getColPairs(N, nums, x, y, Pa, Pb)

    Pa = Pa[Pa >= 0]
    Pb = Pb[Pb >= 0]

    if len(Pa) > 0:
        assert np.abs(Xint[Pa] - Xint[Pb]).max() == 0

    return Pa, Pb

# Macro parameters
x0 = 0; x1 = 2
y0 = 0; y1 = 24
# TODO set to 8000 for evaluation
N = 9000
EPO = 10000
L = 10
Levs = 3
period = 10

# State
#m = np.random.choice([0.7, 1.3, 1.0, 0.8, 1.2, 0.6, 1.4], size=(N, 1))
m = np.random.uniform(low=0.8, high=1.2, size=(N, 1))
r = 0.01
X = np.random.uniform((x0, y0), (x0 + 1, y0 + 1), (N, 2))
V = 13 * np.random.randn(N, 2)
#V = 40 * (np.random.rand(N, 2) - 0.5)

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
Ek_levs = []
N_levs = []
CH = []

# Every particle in ideal gas model has same amount of energy
Ef_const = m[:, 0] * X[:, 1] * (-g[1]) + (m * V * V / 2).sum(1)
I_prev = np.zeros(N).astype(bool)

def CollisionSimplifiedFast(X, V, m, r=0.002, alpha=0.0):
    Xint = (X / r).astype(np.int32)
    P_a, P_b = getColPairs(Xint)

    if len(P_a) > 0:
        assert len(P_a) == len(P_b)

        m_a = m[P_a]
        m_b = m[P_b]
        v_a = V[P_a]
        v_b = V[P_b]

        """
        if 0:
            E_beg = m[P_a] * norm(V[P_a], axis=1, keepdims=True) ** 2 + m[P_b] * norm(V[P_b], axis=1, keepdims=True) ** 2
            V[P_a] *= np.random.randn(len(P_a), 2)
            V[P_b] *= np.random.randn(len(P_b), 2)
            E_end = m[P_a] * norm(V[P_a], axis=1, keepdims=True) ** 2 + m[P_b] * norm(V[P_b], axis=1, keepdims=True) ** 2
            coef = np.sqrt(E_beg/E_end)
            V[P_a] = V[P_a] * coef
            V[P_b] = V[P_b] * coef
            return len(P_a)
        """

        V[P_a] = ((m_a - m_b) * v_a + 2 * m_b * v_b) / (m_a + m_b)
        V[P_b] = ((m_b - m_a) * v_b + 2 * m_a * v_a) / (m_a + m_b)

        # Mixing energy on alpha amount
        if 0:#alpha > 0:
            E_beg = m[P_a] * norm(V[P_a], axis=1, keepdims=True) ** 2 + m[P_b] * norm(V[P_b], axis=1, keepdims=True) ** 2
            V[P_a] = V[P_a] * np.random.uniform(low=1-alpha, high=1+alpha, size=(len(P_a), 1))
            V[P_b] = V[P_b] * np.random.uniform(low=1-alpha, high=1+alpha, size=(len(P_a), 1))
            E_end = m[P_a] * norm(V[P_a], axis=1, keepdims=True) ** 2 + m[P_b] * norm(V[P_b], axis=1, keepdims=True) ** 2
            V[P_a] = V[P_a] * np.sqrt(E_beg/E_end)
            V[P_b] = V[P_b] * np.sqrt(E_beg/E_end)

    return len(P_a)

"""
def CollisionFastAndRough(X, V, m, h=0.4, alpha=0.1):
    # First select randomly region
    xy_b = np.random.uniform(low=(x0, y0), high=(x1-h, y1-h))
    Isel = (X[:, 0] > xy_b[0]) & (X[:, 1] > xy_b[1]) & (X[:, 0] < (xy_b[0] + h)) & (X[:, 1] < (xy_b[1] + h))

    if Isel.sum() >= 2:
        P_ab = np.where(Isel)[0][:(Isel.sum()//2)*2]
        np.random.shuffle(P_ab)

        P_a = P_ab[:len(P_ab)//2]
        P_b = P_ab[-len(P_ab)//2:]

        assert len(P_a) == len(P_b)

        # TODO This scheme doesnt help intersting... total randomnes is bad...
        if 0:
            E_beg = m[P_a] * norm(V[P_a], axis=1, keepdims=True) ** 2 + m[P_b] * norm(V[P_b], axis=1, keepdims=True) ** 2
            V[P_a] = np.random.randn(len(P_a), 2)
            V[P_b] = np.random.randn(len(P_b), 2)
            E_end = m[P_a] * norm(V[P_a], axis=1, keepdims=True) ** 2 + m[P_b] * norm(V[P_b], axis=1, keepdims=True) ** 2
            coef = np.sqrt(E_beg.sum()/E_end.sum())
            V[P_a] = V[P_a] * coef
            V[P_b] = V[P_b] * coef

            return Isel.sum() // 2

        m_a = m[P_a]
        m_b = m[P_b]
        v_a = V[P_a]
        v_b = V[P_b]

        V[P_a] = ((m_a - m_b) * v_a + 2 * m_b * v_b) / (m_a + m_b)
        V[P_b] = ((m_b - m_a) * v_b + 2 * m_a * v_a) / (m_a + m_b)

        if 1:
            E_beg = m[P_a] * norm(V[P_a], axis=1, keepdims=True) ** 2 + m[P_b] * norm(V[P_b], axis=1, keepdims=True) ** 2
            V[P_a] *= (1 + 0.2 * np.random.randn(len(P_a), 2))
            V[P_b] *= (1 + 0.2 * np.random.randn(len(P_b), 2))
            E_end = m[P_a] * norm(V[P_a], axis=1, keepdims=True) ** 2 + m[P_b] * norm(V[P_b], axis=1, keepdims=True) ** 2
            coef = np.sqrt(E_beg/E_end)
            V[P_a] = V[P_a] * coef
            V[P_b] = V[P_b] * coef

            return Isel.sum() // 2

        # Mixing energy on alpha amount
        if 0:#alpha > 0.0:
            E = m[P_ab] * np.linalg.norm(V[P_ab], axis=1, keepdims=True) ** 2
            E_n = np.clip(E.std() * np.random.randn(len(E), 1) + E.mean(), a_min=0.1, a_max=1000000)
            E_n = E_n * (E.sum() / E_n.sum())
            E_n = E * (1 - alpha) + E_n * alpha
            E_n = E_n * (E.sum() / E_n.sum())
            V[P_ab] = (V[P_ab] / np.sqrt(E)) * np.sqrt(E_n)

    return Isel.sum() // 2

def energyMixer(m, V, Iy0):
    alpha = 0.4
    n = Iy0.sum()

    # Storing kinetic energy of affected particles
    E = m[Iy0] * np.linalg.norm(V[Iy0], axis=1, keepdims=True) ** 2

    # E_n = np.clip(E.std() * np.random.randn(n, 1) + E.mean(), a_min=0, a_max=1000000)
    E_n = 0 * E + E.mean(keepdims=True)
    E_n = E_n * (E.sum() / E_n.sum())
    E_n = E * (1 - alpha) + E_n * alpha
    E_n = E_n * (E.sum() / E_n.sum())

    V[Iy0] = (V[Iy0] / np.sqrt(E)) * np.sqrt(E_n)

    if (V != V).any():
        assert 0
"""

for epoch in range(EPO):
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

    col_r = None

    if True:
        col_r = CollisionSimplifiedFast(X_n, V, m)

    #if True:
    #    col_r = CollisionFastAndRough(X_n, V, m)

    #if False and Iy0.sum() >= 2:
    #    energyMixer(m, V, Iy0)

    X = X_n

    if epoch % period == 0:
        assert not (V != V).any()
        print(epoch)
        # Store statistics
        x_data.append(X)
        v_data.append(V)
        Ek_data.append((m * V * V / 2).sum(1))
        Ep_data.append(m[:, 0] * X[:, 1] * (-g[1]))

        h_lev = np.clip((Levs * X[:, 1] / (y1 - y0)).astype(int), a_min=0, a_max=Levs - 1)

        Ek_lev = [0] * Levs
        N_lev = [0] * Levs
        for l in range(Levs):
            Ek_lev[l] = Ek_data[-1][h_lev == l].mean().round(4)
            N_lev[l] = (h_lev == l).sum()

        print('Number levels:   ', N_lev)
        print('E kinetic levels:', Ek_lev, Ek_data[-1].mean())
        print('Collison rate:', col_r)

        Ek_levs.append(Ek_lev)
        N_levs.append(N_lev)

# After loop
x_data = np.stack(x_data)
v_data = np.stack(v_data)
Ek_data = np.array(Ek_data)
Ep_data = np.array(Ep_data)
Ek_levs = np.array(Ek_levs)
N_levs = np.array(N_levs)
if len(CH) > 0:
    CH = np.concatenate(CH)
    print(np.histogram(CH, bins=20)[0])
    plt.hist(CH, bins=100)
    plt.show()

print('Stalled particles:', (x_data[-234:, :, 1].max(axis=0) < 0).sum())

NN = len(Ek_data) // 4
print('T_whole', Ek_levs[-NN:].mean(axis=0))
print('N_whole', N_levs[-NN:].mean(axis=0), N_levs[-NN:].mean(axis=0).sum())

fig, ax1 = plt.subplots()

ax1.plot(N_levs[-NN:].mean(axis=0), color='tab:blue')
ax1.set_xlabel('h level')
ax1.set_ylabel('Num of particles')
ax2 = ax1.twinx()
ax2.plot(Ek_levs[-NN:].mean(axis=0), color='tab:red')
ax2.set_ylabel('T or average kinetic energy of particles')
fig.tight_layout()
plt.show()

Ef = Ep_data.sum(1) + Ek_data.sum(1)
plt.ylim(Ef.min() - 0.02 * Ef.mean(), Ef.max() + 0.02 * Ef.mean())
plt.plot(Ep_data.sum(1) + Ek_data.sum(1))
plt.legend(['Full energy'])
plt.show()

plt.plot(Ek_levs[:, 2])
plt.plot(Ek_levs[:, 1])
plt.plot(Ek_levs[:, 0])
plt.plot(Ek_data.mean(1))
plt.legend(['Lev 2', 'Lev 1', 'Lev 0', 'Lev com'])
plt.show()

Efull = Ek_data + Ep_data
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
ax1.hist(Ep_data[-1], bins=32, alpha=0.3, color='red', label="E potential")
ax2.hist(Ek_data[-1], bins=32, alpha=0.3, color='blue', label="E kinetic")
ax3.hist(Efull[-1],   bins=32, alpha=0.3, color='green', label="E full")
ax1.legend(['E potential'])
ax2.legend(['E kintectic'])
ax3.legend(['E full'])
plt.show()