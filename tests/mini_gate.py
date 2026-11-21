# ruff: noqa: E741, E731, F841
"""Handrechnung des Mini-Gates (EINE Spur, DREI Plätze insgesamt) mit exakten Brüchen und eine unabhängige ereignisdiskrete Simulation: Orakel und Handbeispiele.

Wörtlich aus dem Handbeispiel-Skript der Vorarbeit übernommen (nur `fractions`, `math`, `random`; kein Code der Demo): Gauß-Jordan mit Brüchen, die Matrizen M und R,
stationäre Verteilung, ARL der Karte (n, C) als Bruch, Lauflängenverteilung und die ereignisdiskrete Simulation des Gates Lkw für Lkw."""
from fractions import Fraction as Fr
import math

C_SERV, K_CAP = 1, 3


# ---------------------------------------------------------------- exakte Brüche: Matrix invertieren
def frac_solve(A, B):
    """A X = B, Gauss-Jordan mit Brüchen (A quadratisch)."""
    n = len(A)
    M = [list(map(Fr, A[i])) + list(map(Fr, B[i])) for i in range(n)]
    m = len(B[0])
    for c in range(n):
        p = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        M[c] = [x / piv for x in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [row[n:] for row in M]


def M_matrix(lam, c=C_SERV, K=K_CAP, mu=1):
    """M[j][m] = P(die nächste Ankunft sieht m im System | j im System unmittelbar nach einer Ankunft/Abweisung); nur Abgänge dazwischen."""
    lam = Fr(lam)
    # lam * (lam I - G)^-1, G: Sterberate min(j, c) * mu von j nach j-1
    A = [[Fr(0)] * (K + 1) for _ in range(K + 1)]
    for j in range(K + 1):
        d = min(j, c) * mu
        A[j][j] = lam + d
        if j >= 1:
            A[j][j - 1] = -Fr(d)
    I = [[Fr(int(i == j)) for j in range(K + 1)] for i in range(K + 1)]
    X = frac_solve(A, I)
    return [[lam * X[i][j] for j in range(K + 1)] for i in range(K + 1)]


def R_matrix(lam, c=C_SERV, K=K_CAP):
    M = M_matrix(lam, c, K)
    R = [[Fr(0)] * K for _ in range(K)]
    for n in range(K):
        j = n + 1
        for m in range(K):
            R[n][m] = M[j][m] + M[j][K] * M[K][m] / (1 - M[K][K])
    return M, R


def stationary(R):
    K = len(R)
    A = [[R[j][i] - (1 if i == j else 0) for j in range(K)] for i in range(K)]
    A[-1] = [Fr(1)] * K
    B = [[Fr(0)] for _ in range(K)]
    B[-1] = [Fr(1)]
    return [x[0] for x in frac_solve(A, B)]


# ---------------------------------------------------------------- Karte (n, C)
def chain_arl(R, score, H, init):
    """Zustände (n, C), C = 0..H-1; Score score[m] zum neuen n = m. Löse L = 1 + Q L, Start-Verteilung init über n bei C = 0."""
    K = len(R)
    idx = lambda n, C: n * H + C
    S = K * H
    A = [[Fr(0)] * S for _ in range(S)]
    for n in range(K):
        for C in range(H):
            i = idx(n, C)
            A[i][i] += 1
            for m in range(K):
                C2 = max(0, C + score[m])
                if C2 < H:
                    A[i][idx(m, C2)] -= R[n][m]
    L = [x[0] for x in frac_solve(A, [[Fr(1)] for _ in range(S)])]
    arl = sum(init[n] * L[idx(n, 0)] for n in range(K))
    return arl, L


def rl_distribution(R, score, H, init, tmax):
    """P(RL = t), t = 1..tmax (Fließkomma, Vektor-Iteration)."""
    K = len(R)
    v = {(n, 0): float(init[n]) for n in range(K)}
    out = []
    for t in range(1, tmax + 1):
        w = {}
        dead = 0.0
        for (n, C), p in v.items():
            for m in range(K):
                q = p * float(R[n][m])
                C2 = max(0, C + score[m])
                if C2 >= H:
                    dead += q
                else:
                    w[(m, C2)] = w.get((m, C2), 0.0) + q
        out.append(dead)
        v = w
    return out




def simulate(rng, lam_pre, lam_post, burn, runs, H=2):
    """Gate Lkw fuer Lkw per ereignisdiskreter Simulation (Exp-Ankuenfte, Exp-Abfertigung, eine Spur, 3 Plaetze insgesamt).
    Der burn-te angenommene Lkw ist nu (letzter Lkw vor der Aenderung): ab der naechsten Ankunftsluecke gilt lam_post.
    Karte startet danach bei 0; Rueckgabe: Zahl angenommener Lkw nach nu bis zum Alarm (Alarm-Lkw zaehlt mit)."""
    res = []
    for _ in range(runs):
        n_sys, lam, accepted, C, steps = 0, lam_pre, 0, 0, 0
        t_arr = rng.expovariate(lam)
        t_dep = math.inf
        while True:
            if t_arr < t_dep:
                t = t_arr
                seen = n_sys
                if seen < K_CAP:
                    accepted += 1
                    n_sys += 1
                    if n_sys == 1:
                        t_dep = t + rng.expovariate(1.0)
                    if accepted == burn:
                        lam = lam_post
                    elif accepted > burn:
                        steps += 1
                        C = max(0, C + seen - 1)
                        if C >= H:
                            res.append(steps)
                            break
                t_arr = t + rng.expovariate(lam)
            else:
                t = t_dep
                n_sys -= 1
                t_dep = t + rng.expovariate(1.0) if n_sys > 0 else math.inf
    return res

