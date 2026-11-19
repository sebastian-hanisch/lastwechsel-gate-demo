"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import lwg_constants as C
import lwg_reference as REF


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_surv(curves, vlines=(), title_y="Anteil der Läufe mit Alarm bis zum Lkw t"):
    """Verteilungsfunktion der Lauflänge: curves = [(Name, P(RL > t)-Feld, Farbe, Strichart)]; vlines = [(x, Text)]."""
    fig = go.Figure()
    for name, surv, color, dash in curves:
        t = np.arange(len(surv))
        fig.add_trace(go.Scatter(x=t, y=1 - np.asarray(surv), mode="lines", name=name, line=dict(color=color, width=3, dash=dash),
                                 hovertemplate=name + ", t = %{x}: %{y:.3f}<extra></extra>"))
    for x, text in vlines:
        fig.add_vline(x=x, line=dict(color="rgba(120,120,120,0.7)", dash="dot"))
        fig.add_annotation(x=x, y=1.0, yref="paper", text=text, showarrow=False, bgcolor="rgba(255,255,255,0.88)", font=dict(color="#333"), xanchor="left", yanchor="top")
    fig.update_xaxes(title_text="Lkw seit dem Start der Karte (t)")
    fig.update_yaxes(title_text=title_y, range=[0, 1.02])
    return _base(fig)


def build_kennlinie(kappa: int, lam1: float, target: int):
    """Verzögerung nach dem Wechsel (Zero-state) gegen ARL₀ (log-Achse) für unabhängige Lkw und für die Kette; senkrechte Linie beim Ziel."""
    fig = go.Figure()
    for var, name, color in (("iid", "unabhängige Lkw (Lehrbuchrechnung)", C.COLOR_IID), ("kette", "tatsächliche Kette", C.COLOR_KETTE)):
        cv = REF.curve(kappa, var)
        a0 = np.asarray(cv["arl0"], float)
        a1 = np.asarray(cv["arl1"][str(lam1)], float)
        ok = (a0 >= 5) & (a0 <= 5e3)
        fig.add_trace(go.Scatter(x=a0[ok], y=a1[ok], mode="lines", name=name, line=dict(color=color, width=3),
                                 hovertemplate=name + "<br>ARL₀ %{x:.0f}, nach dem Wechsel %{y:.1f} Lkw<extra></extra>"))
    fig.add_vline(x=target, line=dict(color="rgba(120,120,120,0.8)", dash="dot"))
    fig.update_xaxes(title_text="ARL₀ (Lkw ohne Änderung bis zum Alarm)", type="log", range=[np.log10(5), np.log10(5e3)])
    fig.update_yaxes(title_text="Lkw nach dem Wechsel bis zum Alarm (Zero-state)", rangemode="tozero")
    return _base(fig)


def build_bars(items, xtitle="Lkw nach dem Wechsel bis zum Alarm (ARL₁, Mittel ± Standardfehler)", selected=None, height=None):
    """Waagerechte Balken: items = [(Name, Wert, Standardfehler oder None, Farbe)]; der ausgewählte Balken ist kräftig, die übrigen blasser."""
    names = [i[0] for i in items]
    fig = go.Figure()
    for name, val, se, color in items:
        op = 1.0 if (selected is None or name == selected) else 0.45
        fig.add_trace(go.Bar(y=[name], x=[val], orientation="h", marker=dict(color=color, opacity=op), showlegend=False,
                             error_x=dict(type="data", array=[se], visible=True) if se else None,
                             hovertemplate=name + ": %{x:.1f}<extra></extra>", selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=op))))
    fig.update_yaxes(categoryorder="array", categoryarray=names[::-1], automargin=True)
    fig.update_xaxes(title_text=xtitle, rangemode="tozero")
    fig.update_layout(barmode="overlay")
    return _base(fig, height or max(260, 38 * len(items) + 80))


def build_block_lag(stats: dict):
    """Lag-1-Autokorrelation der Blockmittel je Blocklänge (b = 1: die Wartezeit selbst)."""
    bs = ["1"] + list(stats)
    lag = [REF.satz(0)["ref"]["acf"]["1"]] + [stats[b]["lag1"] for b in stats]
    fig = go.Figure(go.Bar(x=[f"b = {b}" for b in bs], y=lag, marker_color=[C.COLOR_ROH] + [C.COLOR_BLOCK] * len(stats), text=[f"{v:.2f}".replace(".", ",") for v in lag], textposition="outside",
                           hovertemplate="%{x}: Lag-1 %{y:.3f}<extra></extra>"))
    fig.update_yaxes(title_text="Lag-1-Autokorrelation der Beobachtungen der Karte", range=[0, 0.95])
    return _base(fig, 300)


def build_exp4_arl1(rows, lam_sel):
    """ARL₁ der Rohkarte, der Modellkarte (entworfen für das gewählte λ₁) und der für das wahre λ₁ entworfenen Modellkarte je wahrem λ₁ (log-Achse)."""
    xs = [f"{str(r['lam']).replace('.', ',')}" for r in rows]
    fig = go.Figure()
    for key, name, color in (("roh", "Rohkarte auf n", C.COLOR_N), ("modell", "Modellkarte (Entwurf wie gewählt)", C.COLOR_MODELL), ("eigen", "Modellkarte, für das wahre λ₁ entworfen", "#8fd3df")):
        fig.add_trace(go.Bar(x=xs, y=[r[key] for r in rows], name=name, marker_color=color, hovertemplate=name + "<br>λ₁ = %{x}: %{y:.1f} Lkw<extra></extra>"))
    fig.update_xaxes(title_text="wahres λ₁ (alt: 1,6)", type="category")
    fig.update_yaxes(title_text="ARL₁ in Lkw (ARL₀ = 370)", type="log")
    fig.update_layout(barmode="group")
    return _base(fig)


def build_null(rows, lam0_sel):
    """Faktor ARL₀(wahr λ₀′) / ARL₀(λ₀ = 1,6) der beiden festen Karten gegen das wahre λ₀′."""
    xs = [r["lam0"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["faktor_roh"] for r in rows], mode="lines+markers", name="Rohkarte auf n", line=dict(color=C.COLOR_N, width=3)))
    fig.add_trace(go.Scatter(x=xs, y=[r["faktor_modell"] for r in rows], mode="lines+markers", name="Modellkarte", line=dict(color=C.COLOR_MODELL, width=3, dash="dash")))
    fig.add_hline(y=1.0, line=dict(color="rgba(120,120,120,0.7)", dash="dot"))
    fig.add_vline(x=lam0_sel, line=dict(color="rgba(120,120,120,0.7)", dash="dot"))
    fig.update_xaxes(title_text="wahre Rate λ₀′ ohne Änderung (die Karten sind für 1,6 geeicht)")
    fig.update_yaxes(title_text="ARL₀ gegen nominelle ARL₀ (Faktor)", type="log")
    return _base(fig)
