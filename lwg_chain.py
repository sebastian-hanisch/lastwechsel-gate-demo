"""Die Kette der gesehenen Zahl am Gate (exakt): Zahl der Lkw im Gate, die ein ANGENOMMENER Lkw bei der Ankunft sieht.

Gate: c Spuren, K Plätze insgesamt (in Bedienung und wartend), Poisson-Ankünfte mit Rate lam, exponentielle Abfertigung mit Rate mu je Spur. Zeiteinheit = mittlere
Abfertigungsdauer 1/mu. Zwischen zwei Ankünften sterben nur Lkw (Todesprozess, Rate min(j, c)·mu von j nach j - 1), die Lücke ist Exp(lam):
M[j, m] = P(nächste Ankunft sieht m | j im System unmittelbar nach einer Ankunft) = lam · (lam·I − G)^-1 [j, m]  (G: Generator des Todesprozesses).
Abgewiesene Lkw sehen K und werden übersprungen: R[n, m] = M[n+1, m] + M[n+1, K] · M[K, m] / (1 − M[K, K]) ist die Übergangsmatrix der von angenommenen Lkw
gesehenen Zahl n (0 bis K − 1). Alles nur numpy; zweiter Rechenweg (Integral der Matrixexponentialfunktion) und die Geburts-Sterbe-Formel stehen in den Tests."""

import numpy as np

import lwg_constants as C


def death_gen(c: int = C.C_SERV, K: int = C.K_CAP, mu: float = C.MU) -> np.ndarray:
    """Generator des reinen Todesprozesses (Zustände 0 bis K): Rate min(j, c)·mu von j nach j - 1."""
    G = np.zeros((K + 1, K + 1))
    for j in range(1, K + 1):
        d = min(j, c) * mu
        G[j, j - 1] = d
        G[j, j] = -d
    return G


def m_matrix(lam: float, c: int = C.C_SERV, K: int = C.K_CAP, mu: float = C.MU) -> np.ndarray:
    """M[j, m] über die Resolvente des Todesprozesses (siehe Modul-Kopf)."""
    G = death_gen(c, K, mu)
    return lam * np.linalg.inv(lam * np.eye(K + 1) - G)


def acc_chain_from_m(M: np.ndarray) -> np.ndarray:
    """Übergangsmatrix R der von angenommenen Lkw gesehenen Zahl (K × K) aus M ((K + 1) × (K + 1))."""
    K = M.shape[0] - 1
    R = np.zeros((K, K))
    for n in range(K):
        j = n + 1
        for m in range(K):
            R[n, m] = M[j, m] + M[j, K] * M[K, m] / (1.0 - M[K, K])
    return R


def acc_chain(lam: float, c: int = C.C_SERV, K: int = C.K_CAP, mu: float = C.MU) -> np.ndarray:
    return acc_chain_from_m(m_matrix(lam, c, K, mu))


def stat_dist(R: np.ndarray) -> np.ndarray:
    """Stationäre Verteilung von R: pi R = pi, Summe 1 (quadratisches System, letzte Gleichung durch die Normierung ersetzt)."""
    K = R.shape[0]
    A = R.T - np.eye(K)
    A[-1, :] = 1.0
    b = np.zeros(K)
    b[-1] = 1.0
    return np.linalg.solve(A, b)


def wait_mean_given_n(c: int = C.C_SERV, K: int = C.K_CAP, mu: float = C.MU) -> np.ndarray:
    """Mittlere Wartezeit eines Lkw, der n im Gate sieht: max(0, n − c + 1) / (c·mu) (gegeben n ist W Erlang(n − c + 1, c·mu), 0 für n < c)."""
    return np.array([max(0, n - c + 1) / (c * mu) for n in range(K)])


def seen_moments(pi: np.ndarray) -> tuple[float, float]:
    """Mittel und Standardabweichung der gesehenen Zahl unter der Verteilung pi."""
    n = np.arange(len(pi))
    mean = float(pi @ n)
    return mean, float(np.sqrt(pi @ n ** 2 - mean ** 2))


def wait_moments(pi: np.ndarray, c: int = C.C_SERV, mu: float = C.MU) -> tuple[float, float, float]:
    """Mittel, Standardabweichung der Wartezeit eines angenommenen Lkw und P(W = 0) aus der gesehenen Zahl (Mischung aus Atom und Erlang-Anteilen)."""
    rate = c * mu
    m1 = m2 = 0.0
    p0 = 0.0
    for n, w in enumerate(pi):
        a = n - c + 1
        if a <= 0:
            p0 += w
        else:
            m1 += w * a / rate
            m2 += w * a * (a + 1) / rate ** 2
    return m1, float(np.sqrt(m2 - m1 ** 2)), float(p0)


class Gate:
    """Die Kette des Gates bei der Ankunftsrate lam: R, pi, Momente (kleine Hülle um die Funktionen oben)."""

    def __init__(self, lam: float, c: int = C.C_SERV, K: int = C.K_CAP, mu: float = C.MU):
        self.lam, self.c, self.K, self.mu = lam, c, K, mu
        self.R = acc_chain(lam, c, K, mu)
        self.pi = stat_dist(self.R)
        self.mean_n, self.sd_n = seen_moments(self.pi)
        self.mean_w, self.sd_w, self.p0 = wait_moments(self.pi, c, mu)
