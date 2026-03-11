import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DQQ Model · TRC 2025",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Color palette (light / clean academic) ────────────────────────────────────
C = {
    "bg":         "#ffffff",
    "card":       "#f8fafd",
    "border":     "#e2e8f0",
    "blue":       "#2563eb",
    "teal":       "#0891b2",
    "amber":      "#d97706",
    "rose":       "#e11d48",
    "green":      "#16a34a",
    "violet":     "#7c3aed",
    "slate":      "#475569",
    "slate_lt":   "#94a3b8",
    "text":       "#1e293b",
    "text_dim":   "#64748b",
    "blue_bg":    "#eff6ff",
    "amber_bg":   "#fffbeb",
    "rose_bg":    "#fff1f2",
    "green_bg":   "#f0fdf4",
    "teal_bg":    "#ecfeff",
}

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [data-testid="stApp"] {{
      background-color: {C['bg']};
      font-family: 'Inter', sans-serif;
      color: {C['text']};
  }}
  [data-testid="stAppViewContainer"] > .main {{
      background-color: {C['bg']};
  }}
  section[data-testid="stSidebar"] {{ display: none; }}

  /* Header strip */
  .header-strip {{
      background: linear-gradient(135deg, {C['blue']} 0%, {C['teal']} 100%);
      border-radius: 14px;
      padding: 28px 36px 24px;
      margin-bottom: 24px;
      color: white;
  }}
  .header-strip h1 {{
      font-size: 22px; font-weight: 700; margin: 0 0 6px 0; line-height: 1.35;
      font-family: 'Inter', sans-serif;
  }}
  .header-strip .authors {{
      font-size: 13px; opacity: 0.85; margin: 0 0 14px 0;
  }}
  .header-strip .badge {{
      display: inline-block;
      background: rgba(255,255,255,0.2);
      border: 1px solid rgba(255,255,255,0.35);
      border-radius: 20px; padding: 3px 12px;
      font-size: 11px; font-weight: 600;
      letter-spacing: 0.05em; margin-right: 8px;
      font-family: 'JetBrains Mono', monospace;
  }}

  /* Stat cards */
  .stat-row {{ display: flex; gap: 14px; margin-bottom: 20px; flex-wrap: wrap; }}
  .stat-card {{
      background: {C['card']}; border: 1px solid {C['border']};
      border-radius: 12px; padding: 16px 20px; flex: 1; min-width: 130px;
      border-top: 3px solid var(--accent);
  }}
  .stat-card .val {{
      font-size: 26px; font-weight: 700; color: var(--accent);
      font-family: 'JetBrains Mono', monospace; line-height: 1;
  }}
  .stat-card .lbl {{ font-size: 11px; color: {C['text_dim']}; margin-top: 5px; }}
  .stat-card .sub {{ font-size: 10px; color: {C['slate_lt']}; margin-top: 2px; }}

  /* Section cards */
  .sec-card {{
      background: {C['card']}; border: 1px solid {C['border']};
      border-radius: 12px; padding: 22px 24px; margin-bottom: 18px;
  }}
  .sec-title {{
      font-size: 10.5px; font-weight: 700; letter-spacing: 0.10em;
      text-transform: uppercase; color: {C['slate']}; margin-bottom: 14px;
      display: flex; align-items: center; gap: 8px;
  }}
  .sec-title::before {{
      content: ''; width: 3px; height: 14px; background: {C['blue']};
      border-radius: 2px; display: inline-block;
  }}

  /* Hypothesis cards */
  .hyp-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  .hyp-card {{
      border-radius: 10px; padding: 14px 16px;
      border-left: 4px solid var(--hc);
      background: var(--hbg);
  }}
  .hyp-label {{
      display: inline-block; background: var(--hc); color: white;
      border-radius: 5px; padding: 1px 9px; font-size: 10px; font-weight: 700;
      font-family: 'JetBrains Mono', monospace; margin-bottom: 6px;
  }}
  .hyp-verdict {{ color: var(--hc); font-size: 11px; font-weight: 700; margin-left: 8px; }}
  .hyp-title {{ font-size: 12.5px; font-weight: 600; color: {C['text']}; margin-bottom: 4px; }}
  .hyp-desc {{ font-size: 11px; color: {C['text_dim']}; line-height: 1.55; }}

  /* Info boxes */
  .info-box {{
      border-radius: 8px; padding: 10px 14px;
      border-left: 3px solid var(--ic); background: var(--ibg);
      font-size: 11.5px; color: {C['text_dim']}; margin-top: 12px; line-height: 1.55;
  }}

  /* Formula cards */
  .formula-card {{
      display: flex; justify-content: space-between; align-items: center;
      padding: 9px 13px; border-radius: 8px; margin-bottom: 7px;
      background: var(--fc-bg);
  }}
  .formula-label {{ font-size: 11px; color: {C['text_dim']}; }}
  .formula-eq {{
      font-family: 'JetBrains Mono', monospace; font-size: 12px;
      font-weight: 600; color: var(--fc);
  }}

  /* Model compare table */
  .model-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  .model-table th {{
      color: {C['text_dim']}; font-size: 10px; font-weight: 600;
      letter-spacing: 0.04em; padding: 8px 12px; text-align: left;
      border-bottom: 1px solid {C['border']};
  }}
  .model-table td {{ padding: 9px 12px; border-bottom: 1px solid #f1f5f9; }}

  /* Phase legend */
  .phase-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 7px; }}
  .phase-dot {{ width: 10px; height: 10px; border-radius: 3px; flex-shrink: 0; }}

  /* Divider */
  hr.sec-divider {{
      border: none; border-top: 1px solid {C['border']}; margin: 6px 0 20px 0;
  }}

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] {{
      background: {C['card']}; border-radius: 12px;
      padding: 6px 8px; border: 1px solid {C['border']};
      margin-bottom: 20px; gap: 4px;
  }}
  .stTabs [data-baseweb="tab"] {{
      border-radius: 8px; padding: 8px 20px;
      font-size: 12px; font-weight: 500;
      color: {C['text_dim']}; border: none; background: transparent;
  }}
  .stTabs [aria-selected="true"] {{
      background: {C['blue']} !important; color: white !important;
  }}
  .stTabs [data-baseweb="tab-panel"] {{ padding: 0; }}

  /* Streamlit overrides */
  .stSelectbox label, .stRadio label {{ font-size: 12px; color: {C['text_dim']}; }}
  div[data-testid="stVerticalBlock"] > div {{ gap: 0px; }}
  .block-container {{ padding-top: 24px !important; padding-bottom: 40px !important; max-width: 1120px !important; }}
  .element-container {{ margin-bottom: 0px !important; }}
  iframe {{ border: none; }}
</style>
""", unsafe_allow_html=True)


# ── Helper: Plotly layout defaults ───────────────────────────────────────────
def base_layout(title="", h=300):
    return dict(
        title=dict(text=title, font=dict(size=12, color=C['slate']), x=0),
        height=h,
        margin=dict(l=46, r=16, t=30 if title else 16, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafd",
        font=dict(family="Inter", size=11, color=C['text_dim']),
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)",
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor="#e2e8f0", linecolor="#e2e8f0", showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="#e2e8f0", linecolor="#e2e8f0", showgrid=True, zeroline=False),
    )


# ── DQQ curve generator (smooth double quadratic) ───────────────────────────
# π(t) = dQ/dt: piecewise quadratic — π_D = α(t−t₀)(t−t₂), π_R = β(t−t₂)(t−t₄)
# Q(t) = ∫π dτ: piecewise cubic, C¹ at t₂; minimum at t₂, recovers t₂→t₄, then flat at Δ
def _dqq_params(t2, t4, scenario):
    """Set α,β so Q(t₂)=Q_min (lowest) and Q(t₄)=Δ (scenario). No clipping needed."""
    R = t4 - t2
    delta_map = {"loss": -15, "normal": 0, "gain": 12}
    delta = delta_map.get(scenario, -15)
    Q_min = -55.0  # desired minimum at t2
    # Q(t₂) = -α t₂³/6  =>  α = 6·|Q_min|/t₂³
    alpha = 6.0 * abs(Q_min) / (t2 ** 3)
    # Q(t₄) = Q(t₂) - β R³/6 = Δ  =>  β = 6(Q(t₂)−Δ)/R³
    beta = 6.0 * (Q_min - delta) / (R ** 3)
    return alpha, beta, delta


def dqq_curve(alpha=None, beta=None, t2=45, t4=140, scenario="loss", n_points=400):
    R = t4 - t2
    if alpha is None or beta is None:
        alpha, beta, delta = _dqq_params(t2, t4, scenario)
    else:
        delta_map = {"loss": -15, "normal": 0, "gain": 12}
        delta = delta_map.get(scenario, -15)
    t_vals = np.linspace(0, 181, n_points)
    Q = np.zeros_like(t_vals)
    for i, t in enumerate(t_vals):
        if t <= t2:
            # Q(t) = (α/6) t² (2t − 3t₂)  →  min at t₂
            Q[i] = (alpha / 6) * t**2 * (2*t - 3*t2)
        elif t <= t4:
            dt = t - t2
            # Q(t) = Q(t₂) + ∫ π_R dτ  →  gradual recovery to Q(t₄)=Δ
            Q[i] = -(alpha/6)*t2**3 - beta*dt**2*(R/2 - dt/3)
        else:
            Q[i] = delta  # stop recovering at t₄
    return t_vals, Q


def dqq_pi(alpha=None, beta=None, t2=45, t4=140, scenario="loss", n_points=400):
    """π(t) = dQ/dt: piecewise quadratic (double quadratic), zero at t₂ and t₄."""
    if alpha is None or beta is None:
        alpha, beta, _ = _dqq_params(t2, t4, scenario)
    t_vals = np.linspace(0, 181, n_points)
    pi_vals = np.zeros_like(t_vals)
    for i, t in enumerate(t_vals):
        if t <= t2:
            pi_vals[i] = alpha * t * (t - t2)  # π_D = α(t−t₀)(t−t₂)
        elif t <= t4:
            pi_vals[i] = beta * (t - t2) * (t - t4)  # π_R = β(t−t₂)(t−t₄)
        else:
            pi_vals[i] = 0.0
    return t_vals, pi_vals


def vshaped_curve(t2=45, t4=140, Qmax=-50, n_points=400):
    t_vals = np.linspace(0, 181, n_points)
    Q = np.where(t_vals <= t2,
                 (Qmax / t2) * t_vals,
                 Qmax + (-Qmax / (t4 - t2)) * (t_vals - t2))
    return t_vals, np.clip(Q, -70, 30)


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="header-strip">
  <div>
    <span class="badge">Transportation Research Part C: Emerging Technologies 2025</span>
    <span class="badge">DOI: 10.1016/j.trc.2025.105122</span>
  </div>
  <h1 style="margin-top:10px;">Mobility Resilience and Recovery Dynamics:<br>Parsimonious Framework Beyond V-shapes</h1>
  <p class="authors">Fang Tang · Xiangyong Luo · Xuesong (Simon) Zhou &nbsp;·&nbsp; Arizona State University</p>
</div>
""", unsafe_allow_html=True)

# Stat row
st.markdown("""
<div class="stat-row">
  <div class="stat-card" style="--accent:#2563eb">
    <div class="val">0.966</div>
    <div class="lbl">Best R² — DQQ on Q(t)</div>
    <div class="sub">nationwide, U.S. transit</div>
  </div>
  <div class="stat-card" style="--accent:#0891b2">
    <div class="val">51</div>
    <div class="lbl">Regions Analyzed</div>
    <div class="sub">50 states + D.C.</div>
  </div>
  <div class="stat-card" style="--accent:#7c3aed">
    <div class="val">DQQ</div>
    <div class="lbl">Novel ODE Framework</div>
    <div class="sub">only 2 parameters (α, β)</div>
  </div>
  <div class="stat-card" style="--accent:#d97706">
    <div class="val">−4.05</div>
    <div class="lbl">Mean t-stat, H Line impact</div>
    <div class="sub">all routes, 95% CI</div>
  </div>
  <div class="stat-card" style="--accent:#16a34a">
    <div class="val">81.8%</div>
    <div class="lbl">Variance Explained</div>
    <div class="sub">transit Δ regression R²</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["📋  Overview", "📐  DQQ Model", "📊  Performance Metrics", "🔍  Findings"])


# ────────────────────────────────────────────────────────────────────────────
# TAB 1 · OVERVIEW
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns([1.05, 1], gap="medium")

    with col_a:
        st.markdown(f"""
        <div class="sec-card">
          <div class="sec-title">Research Motivation</div>
          <p style="font-size:13px; color:{C['text_dim']}; line-height:1.75; margin:0">
            Existing resilience models rely on <strong style="color:{C['amber']}">V-shaped / U-shaped / trapezoid</strong>
            approximations that assume <em>linear</em> disruption and recovery — incapable of capturing
            non-linear dynamics, identifying steady-state points, or quantifying partial recovery.
            <br><br>
            This paper adapts Newell's (1982) fluid-based <strong>Polynomial Arrival Queue (PAQ)</strong> model
            into the <strong style="color:{C['blue']}">Double Quadratic Queue (DQQ)</strong> ODE framework,
            providing closed-form solutions for resilience metrics and operating across scales from
            national to route level.
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="sec-card">
          <div class="sec-title">Four Mobility Phases</div>
          <div class="phase-row"><div class="phase-dot" style="background:{C['slate_lt']}"></div>
            <span style="font-size:12px"><strong style="color:{C['slate']}">Pre-disruption</strong> — baseline mobility patterns</span></div>
          <div class="phase-row"><div class="phase-dot" style="background:{C['rose']}"></div>
            <span style="font-size:12px"><strong style="color:{C['rose']}">Disruption</strong> — sharp decline, t₀ → t₂</span></div>
          <div class="phase-row"><div class="phase-dot" style="background:{C['amber']}"></div>
            <span style="font-size:12px"><strong style="color:{C['amber']}">Recovery</strong> — gradual restoration, t₂ → t₄</span></div>
          <div class="phase-row"><div class="phase-dot" style="background:{C['green']}"></div>
            <span style="font-size:12px"><strong style="color:{C['green']}">Stabilization</strong> — new normal at t₄ (loss / normal / gain)</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="sec-card">
          <div class="sec-title">Key Resilience Metrics</div>
          {"".join([f'<div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid {C["border"]};font-size:12px"><span style="color:{C["text_dim"]}">{n}</span><span style="font-family:JetBrains Mono,monospace;font-size:11px;color:{C["blue"]}">{v}</span></div>'
            for n, v in [
              ("D — Disruption duration", "t₂ − t₀"),
              ("R — Recovery duration", "t₄ − t₂"),
              ("r — ROD ratio", "R / D ∈ [0, ∞)"),
              ("Q_max — Maximum loss", "min Q(t) at t₂"),
              ("Δ — Transition loss/gain", "Q(t₄) − baseline"),
              ("RTA — Resilience Trajectory Area", "∫ Q_R(τ)dτ + Δ·R"),
            ]])}
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div class="sec-card">
          <div class="sec-title">Four Research Hypotheses</div>
          <div class="hyp-grid">
            <div class="hyp-card" style="--hc:{C['blue']};--hbg:{C['blue_bg']}">
              <div><span class="hyp-label">H1</span></div>
              <div class="hyp-title">Model Fit Quality</div>
              <div class="hyp-desc">Does DQQ significantly outperform V-shaped and Exponential models by R², RMSE, and AIC?</div>
            </div>
            <div class="hyp-card" style="--hc:{C['teal']};--hbg:{C['teal_bg']}">
              <div><span class="hyp-label">H2</span></div>
              <div class="hyp-title">Metrics Correlation</div>
              <div class="hyp-desc">Is there a statistically significant relationship between R, Δ, and Q_max?</div>
            </div>
            <div class="hyp-card" style="--hc:{C['amber']};--hbg:{C['amber_bg']}">
              <div><span class="hyp-label">H3</span></div>
              <div class="hyp-title">Socioeconomic Impact</div>
              <div class="hyp-desc">Does household income significantly influence recovery duration and transition loss?</div>
            </div>
            <div class="hyp-card" style="--hc:{C['violet']};--hbg:#f5f3ff">
              <div><span class="hyp-label">H4</span></div>
              <div class="hyp-title">Intervention Effect</div>
              <div class="hyp-desc">Does the new H Line opening lead to statistically significant changes in ridership?</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="sec-card">
          <div class="sec-title">Datasets</div>
          <div style="padding:12px 14px;background:{C['blue_bg']};border-radius:8px;margin-bottom:10px;border-left:3px solid {C['blue']}">
            <div style="font-size:12px;font-weight:600;color:{C['blue']};margin-bottom:4px">Google COVID-19 Community Mobility Reports</div>
            <div style="font-size:11px;color:{C['text_dim']};line-height:1.55">
              Nationwide · Statewide (50 + D.C.) · Countywide (SF, Maricopa, King County)<br>
              6 mobility categories · Jan 2020 – Oct 2022
            </div>
          </div>
          <div style="padding:12px 14px;background:#f5f3ff;border-radius:8px;border-left:3px solid {C['violet']}">
            <div style="font-size:12px;font-weight:600;color:{C['violet']};margin-bottom:4px">SoundTransit RapidRide — King County</div>
            <div style="font-size:11px;color:{C['text_dim']};line-height:1.55">
              Lines A, B, C, D, E, F, H · bidirectional monthly boardings<br>
              2019 – 2024 · AM peak (05:00 – 09:00)
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 2 · DQQ MODEL
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    col1, col2 = st.columns([1.4, 1], gap="medium")

    with col1:
        st.markdown(f'<div class="sec-title" style="margin-bottom:10px">Interactive DQQ Trajectory Q(t)</div>', unsafe_allow_html=True)

        ctrl_c1, ctrl_c2 = st.columns([1, 1])
        with ctrl_c1:
            scenario = st.radio("Stabilization scenario", ["Transition Loss", "Normal Recovery", "Transition Gain"], horizontal=True, label_visibility="collapsed")
        with ctrl_c2:
            show_v = st.checkbox("Overlay V-shaped model", value=True)

        scen_key = {"Transition Loss": "loss", "Normal Recovery": "normal", "Transition Gain": "gain"}[scenario]
        t, Q = dqq_curve(scenario=scen_key)
        tv, Qv = vshaped_curve()

        fig = go.Figure()
        # Shaded area under DQQ (smooth piecewise cubic)
        fig.add_trace(go.Scatter(
            x=t, y=Q, mode="lines", name="DQQ Q(t)",
            line=dict(color=C['blue'], width=2.5),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.08)",
        ))
        # Phase markers
        for tx, label in [(0, "t₀"), (45, "t₂"), (140, "t₄")]:
            fig.add_vline(x=tx, line_dash="dot", line_color=C['slate_lt'], line_width=1.2)
            fig.add_annotation(x=tx, y=5, text=label, showarrow=False,
                               font=dict(size=11, color=C['slate'], family="JetBrains Mono"),
                               bgcolor="white", bordercolor=C['border'], borderwidth=1)
        if show_v:
            fig.add_trace(go.Scatter(
                x=tv, y=Qv, mode="lines", name="V-shaped Q(t)",
                line=dict(color=C['amber'], width=1.8, dash="dash"),
            ))
        fig.add_hline(y=0, line_color=C['slate_lt'], line_width=1)
        lay = base_layout(h=280)
        lay["xaxis"]["title"] = dict(text="Time (days)", font=dict(size=11))
        lay["yaxis"]["title"] = dict(text="Q(t) — Cumulative Change (%)", font=dict(size=11))
        fig.update_layout(**lay)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # π(t) — analytical piecewise quadratic (smooth double quadratic)
        t_pi, pi_vals = dqq_pi(t2=45, t4=140, scenario=scen_key)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=t_pi, y=pi_vals, mode="lines", name="π(t)",
                                   line=dict(color=C['teal'], width=2),
                                   fill="tozeroy", fillcolor="rgba(8,145,178,0.07)"))
        fig2.add_hline(y=0, line_color=C['slate_lt'], line_width=1)
        for tx in [0, 45, 140]:
            fig2.add_vline(x=tx, line_dash="dot", line_color=C['slate_lt'], line_width=1)
        lay2 = base_layout(h=180)
        lay2["xaxis"]["title"] = dict(text="Time (days)", font=dict(size=11))
        lay2["yaxis"]["title"] = dict(text="π(t) — Rate of Change", font=dict(size=11))
        fig2.update_layout(**lay2)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    with col2:
        st.markdown(f"""
        <div class="sec-card">
          <div class="sec-title">Mathematical Formulation</div>
          <div style="font-size:11px;color:{C['text_dim']};margin-bottom:10px">Rate of change π(t) — piecewise quadratic:</div>

          <div class="formula-card" style="--fc:{C['rose']};--fc-bg:{C['rose_bg']}">
            <span class="formula-label">Disruption phase</span>
            <span class="formula-eq">π_D = α(t−t₀)(t−t₂)</span>
          </div>
          <div class="formula-card" style="--fc:{C['green']};--fc-bg:{C['green_bg']}">
            <span class="formula-label">Recovery phase</span>
            <span class="formula-eq">π_R = β(t−t₂)(t−t₄)</span>
          </div>
          <div class="formula-card" style="--fc:{C['blue']};--fc-bg:{C['blue_bg']}">
            <span class="formula-label">Cumulative change</span>
            <span class="formula-eq">Q(t) = ∫ π(τ) dτ</span>
          </div>

          <div style="font-size:11px;color:{C['text_dim']};margin:14px 0 10px">Closed-form resilience metrics:</div>
          <div class="formula-card" style="--fc:{C['amber']};--fc-bg:{C['amber_bg']}">
            <span class="formula-label">Max loss Q_max</span>
            <span class="formula-eq">−α/6 · t₂³</span>
          </div>
          <div class="formula-card" style="--fc:{C['violet']};--fc-bg:#f5f3ff">
            <span class="formula-label">Transition Δ</span>
            <span class="formula-eq">−α/6·t₂³ − β/6·R³</span>
          </div>
          <div class="formula-card" style="--fc:{C['green']};--fc-bg:{C['green_bg']}">
            <span class="formula-label">Full recovery condition</span>
            <span class="formula-eq">α = −β · r³</span>
          </div>
          <div class="formula-card" style="--fc:{C['teal']};--fc-bg:{C['teal_bg']}">
            <span class="formula-label">ROD ratio r</span>
            <span class="formula-eq">R/D ∈ [0, ∞)</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="sec-card" style="margin-top:0">
          <div class="sec-title">Model Comparison</div>
          <table class="model-table">
            <thead><tr>
              <th>Model</th><th>ROD r</th><th>Limitation</th>
            </tr></thead>
            <tbody>
              <tr>
                <td style="font-weight:600;color:{C['amber']};font-family:JetBrains Mono,monospace;font-size:11px">V-shaped</td>
                <td style="color:{C['text_dim']}">—</td>
                <td style="color:{C['text_dim']}">Linear only; no steady-state</td>
              </tr>
              <tr>
                <td style="font-weight:600;color:{C['slate']};font-family:JetBrains Mono,monospace;font-size:11px">Single Quad.</td>
                <td style="color:{C['text_dim']}">[0, ½]</td>
                <td style="color:{C['text_dim']}">Cannot identify t₄</td>
              </tr>
              <tr>
                <td style="font-weight:600;color:{C['violet']};font-family:JetBrains Mono,monospace;font-size:11px">Cubic</td>
                <td style="color:{C['text_dim']}">[0, 1]</td>
                <td style="color:{C['text_dim']}">Fails short disruptions</td>
              </tr>
              <tr style="background:{C['blue_bg']}">
                <td style="font-weight:700;color:{C['blue']};font-family:JetBrains Mono,monospace;font-size:11px">DQQ ✦</td>
                <td style="color:{C['green']};font-weight:600">[0, ∞)</td>
                <td style="color:{C['green']};font-weight:600">All scenarios covered</td>
              </tr>
            </tbody>
          </table>
        </div>
        """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 3 · PERFORMANCE METRICS
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    col_p1, col_p2 = st.columns(2, gap="medium")

    with col_p1:
        # R² grouped bar
        df_r2 = pd.DataFrame({
            "Metric": ["R² Q(t)", "R² π(t)"],
            "DQQ":         [0.966, 0.766],
            "V-shaped":    [0.748, 0.039],
            "Exponential": [0.941, 0.693],
        })
        fig_r2 = go.Figure()
        bar_colors = [C['blue'], C['amber'], C['violet']]
        for model, clr in zip(["DQQ", "V-shaped", "Exponential"], bar_colors):
            fig_r2.add_trace(go.Bar(name=model, x=df_r2["Metric"], y=df_r2[model],
                                    marker_color=clr, marker_line_width=0))
        lay = base_layout("Nationwide Model Fit — R²", h=270)
        lay["yaxis"]["range"] = [0, 1.05]
        lay["barmode"] = "group"
        fig_r2.update_layout(**lay)
        st.plotly_chart(fig_r2, use_container_width=True, config={"displayModeBar": False})

        # RMSE
        df_rmse = pd.DataFrame({
            "Metric": ["RMSE Q(t)", "RMSE π(t)"],
            "DQQ":         [1.852, 0.460],
            "V-shaped":    [5.018, 0.931],
            "Exponential": [2.422, 0.526],
        })
        fig_rmse = go.Figure()
        for model, clr in zip(["DQQ", "V-shaped", "Exponential"], bar_colors):
            fig_rmse.add_trace(go.Bar(name=model, x=df_rmse["Metric"], y=df_rmse[model],
                                      marker_color=clr, marker_line_width=0))
        lay2 = base_layout("RMSE (lower = better)", h=250)
        lay2["barmode"] = "group"
        fig_rmse.update_layout(**lay2)
        st.plotly_chart(fig_rmse, use_container_width=True, config={"displayModeBar": False})

    with col_p2:
        # Resolution R²
        df_res = pd.DataFrame({
            "Resolution": ["Nationwide", "Statewide", "Countywide"],
            "DQQ":         [0.866, 0.700, 0.864],
            "V-shaped":    [0.393, 0.515, 0.515],
            "Exponential": [0.817, 0.594, 0.450],
        })
        fig_res = go.Figure()
        for model, clr in zip(["DQQ", "V-shaped", "Exponential"], bar_colors):
            fig_res.add_trace(go.Bar(name=model, x=df_res["Resolution"], y=df_res[model],
                                     marker_color=clr, marker_line_width=0))
        lay3 = base_layout("Average R² Across Spatial Resolutions", h=270)
        lay3["yaxis"]["range"] = [0, 1.05]
        lay3["barmode"] = "group"
        fig_res.update_layout(**lay3)
        st.plotly_chart(fig_res, use_container_width=True, config={"displayModeBar": False})

        # Disruption & Recovery duration
        mobility_types = ["Grocery & Pharmacy", "Parks", "Residential",
                          "Retail & Recreation", "Transit Stations", "Workplaces"]
        D_vals = [27, 35, 35, 36, 36, 40]
        R_vals = [57, 73, 36, 74, 70, 68]
        pal = [C['amber'], C['green'], C['violet'], C['rose'], C['blue'], C['teal']]

        view = st.radio("Resilience metric", ["Disruption Duration (D)", "Recovery Duration (R)"],
                        horizontal=True, label_visibility="collapsed")
        vals = D_vals if "Disruption" in view else R_vals
        fig_mob = go.Figure(go.Bar(
            y=mobility_types, x=vals, orientation="h",
            marker_color=pal, marker_line_width=0,
            text=[f"{v} d" for v in vals], textposition="outside",
            textfont=dict(size=11, color=C['text_dim'])
        ))
        lay4 = base_layout(view + " by Mobility Type (median, days)", h=260)
        lay4["xaxis"]["title"] = dict(text="Days", font=dict(size=11))
        lay4["yaxis"]["autorange"] = "reversed"
        lay4["showlegend"] = False
        fig_mob.update_layout(**lay4)
        st.plotly_chart(fig_mob, use_container_width=True, config={"displayModeBar": False})

    # Transition distribution
    st.markdown(f"""
    <div class="info-box" style="--ic:{C['blue']};--ibg:{C['blue_bg']}">
      <strong style="color:{C['blue']}">Key Takeaway (H1 ✓):</strong>&nbsp; DQQ achieves R²=0.966/0.766 on Q(t)/π(t) nationally — vs V-shaped (0.748/0.039) and Exponential (0.941/0.693) — using only 2 parameters (α, β). The advantage is most pronounced in capturing the <em>rate of change</em> π(t) and correctly identifying the stabilization point t₄.
    </div>
    """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 4 · FINDINGS
# ────────────────────────────────────────────────────────────────────────────
with tab4:
    col_f1, col_f2 = st.columns(2, gap="medium")

    with col_f1:
        # Income vs Rapidity scatter (H3)
        np.random.seed(42)
        income = np.array([58,61,65,68,70,73,75,78,80,83,87,90,93,97])
        rapidity = np.array([0.58,0.52,0.48,0.47,0.44,0.42,0.40,0.38,0.35,0.30,0.28,0.22,0.20,0.18])
        n = len(income)
        # Linear trend and 99% CI
        z = np.polyfit(income, rapidity, 1)
        p = np.poly1d(z)
        x_line = np.linspace(income.min() - 2, income.max() + 2, 100)
        y_fit = p(x_line)
        # Standard error for 99% CI (t ~ 2.98 for df=12)
        res = rapidity - p(income)
        se = np.sqrt(np.sum(res**2) / (n - 2)) * np.sqrt(1/n + (x_line - income.mean())**2 / np.sum((income - income.mean())**2))
        t99 = 2.681  # approx for df=12, 99%
        y_upper = y_fit + t99 * se
        y_lower = y_fit - t99 * se

        fig_inc = go.Figure()
        fig_inc.add_trace(go.Scatter(
            x=x_line, y=y_upper, mode="lines", line=dict(width=0), showlegend=False,
        ))
        fig_inc.add_trace(go.Scatter(
            x=x_line, y=y_lower, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(225,29,72,0.12)", showlegend=False,
        ))
        fig_inc.add_trace(go.Scatter(
            x=income, y=rapidity, mode="markers", name="States",
            marker=dict(color=C['amber'], size=10, line=dict(color="white", width=1.5)),
        ))
        fig_inc.add_trace(go.Scatter(
            x=x_line, y=y_fit, mode="lines", name="Trend (99% CI)",
            line=dict(color=C['rose'], width=2.5),
        ))
        lay5 = base_layout("Income vs. Recovery Rapidity (H3)", h=320)
        lay5["xaxis"].update(title=dict(text="Avg. Household Income (k USD)", font=dict(size=11)), range=[54, 102])
        lay5["yaxis"].update(title=dict(text="Rapidity", font=dict(size=11)), range=[0.12, 0.65])
        lay5["legend"]["x"] = 0.02
        lay5["legend"]["xanchor"] = "left"
        fig_inc.update_layout(**lay5)
        st.plotly_chart(fig_inc, use_container_width=True, config={"displayModeBar": False})

        st.markdown(f"""
        <div class="info-box" style="--ic:{C['rose']};--ibg:{C['rose_bg']}">
          <strong style="color:{C['rose']}">H3 ✓ Supported:</strong>&nbsp; Higher-income regions recover more slowly. Wealthier populations shift to personal cars, reducing transit urgency and delaying system-level recovery. They also exhibit larger transition losses (more negative Δ).
        </div>
        """, unsafe_allow_html=True)

    with col_f2:
        # H Line impact (H4)
        months_short = list("JFMAMJJASOND")
        c_line = [0, 0, 0, 80, 120, -50, -180, -220, 150, -200, 100, -240]
        de_line = [0, 0, 0, 40, 60, 30, -80, -100, 50, -90, 30, -70]
        abf_line = [0, 0, 0, 15, 10, 8, -5, -10, 5, -8, 3, -12]

        fig_h = go.Figure()
        fig_h.add_hline(y=0, line_color=C['slate_lt'], line_width=1)
        fig_h.add_vline(x=2, line_color=C['violet'], line_width=2, line_dash="dash",  # March
                        annotation_text="H Line opens", annotation_font_size=11,
                        annotation_font_color=C['violet'], annotation_position="top",
                        annotation_bgcolor="rgba(255,255,255,0.9)")
        for vals, name, clr, dash in [
            (c_line,   "Cluster 3 — C Line (5 overlaps)",  C['rose'],   "solid"),
            (de_line,  "Cluster 2 — D, E Lines (3 overlaps)", C['amber'], "solid"),
            (abf_line, "Cluster 1 — A, B, F (baseline)",  C['slate'], "dot"),
        ]:
            fig_h.add_trace(go.Scatter(
                x=list(range(12)), y=vals, mode="lines+markers", name=name,
                line=dict(color=clr, width=2.5 if dash == "solid" else 2, dash=dash),
                marker=dict(size=6, line=dict(width=0)),
            ))
        lay6 = base_layout("H Line Opening Impact — Month-by-Month Δπ(t) (H4)", h=320)
        lay6["xaxis"] = dict(**lay6["xaxis"], tickvals=list(range(12)),
                              ticktext=months_short, title=dict(text="Month (2023)", font=dict(size=11)))
        lay6["yaxis"].update(title=dict(text="Δπ(t) month-over-month", font=dict(size=11)), range=[-260, 140])
        lay6["legend"]["x"] = 0.02
        lay6["legend"]["xanchor"] = "left"
        lay6["legend"]["orientation"] = "v"
        lay6["legend"]["yanchor"] = "top"
        lay6["legend"]["y"] = 0.98
        fig_h.update_layout(**lay6)
        st.plotly_chart(fig_h, use_container_width=True, config={"displayModeBar": False})

        st.markdown(f"""
        <div class="info-box" style="--ic:{C['violet']};--ibg:#f5f3ff">
          <strong style="color:{C['violet']}">H4 ✓ Supported:</strong>&nbsp; Mean |t| = 4.05 > 1.69 critical. C Line (5 overlaps) t = −5.63; E Line (3 overlaps) t = −2.09. Geometric alignment and service overlap are key determinants of impact magnitude.
        </div>
        """, unsafe_allow_html=True)

    # Hypothesis summary
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sec-card">
      <div class="sec-title">All Four Hypotheses — Verdict</div>
      <div class="hyp-grid">
        <div class="hyp-card" style="--hc:{C['green']};--hbg:{C['green_bg']}">
          <div><span class="hyp-label">H1</span><span class="hyp-verdict">✓ Supported</span></div>
          <div class="hyp-title">DQQ superior model fit</div>
          <div class="hyp-desc">Highest R², lowest RMSE and AIC across all resolutions. V-shaped π(t) R²=0.039 vs DQQ 0.766 — a 20× improvement.</div>
        </div>
        <div class="hyp-card" style="--hc:{C['green']};--hbg:{C['green_bg']}">
          <div><span class="hyp-label">H2</span><span class="hyp-verdict">✓ Supported</span></div>
          <div class="hyp-title">Q_max correlates with R and Δ</div>
          <div class="hyp-desc">Larger initial disruption → longer recovery (positive) and greater net transition loss (negative correlation with Δ).</div>
        </div>
        <div class="hyp-card" style="--hc:{C['green']};--hbg:{C['green_bg']}">
          <div><span class="hyp-label">H3</span><span class="hyp-verdict">✓ Supported</span></div>
          <div class="hyp-title">Income negatively affects recovery</div>
          <div class="hyp-desc">Higher income → slower rapidity, longer R, larger Δ. Car substitution reduces transit urgency in wealthier regions.</div>
        </div>
        <div class="hyp-card" style="--hc:{C['green']};--hbg:{C['green_bg']}">
          <div><span class="hyp-label">H4</span><span class="hyp-verdict">✓ Supported</span></div>
          <div class="hyp-title">H Line opening significantly impacts ridership</div>
          <div class="hyp-desc">Mean t-stat = −4.05. C Line most affected (t = −5.63) with 5 overlapping segments. Spatial alignment is the key driver.</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Regression table
    st.markdown(f"""
    <div class="sec-card">
      <div class="sec-title">Regression — Transit Station Transition Loss &nbsp; R²=0.818, F=23.71, p&lt;0.0001</div>
      <table class="model-table">
        <thead><tr>
          <th>Variable</th><th>Avg. Value</th><th>Coefficient</th><th>p-value</th><th>Significance</th>
        </tr></thead>
        <tbody>
          {"".join([
            f'<tr><td style="font-size:12px;color:{row[5]}">{row[0]}</td>'
            f'<td style="font-family:JetBrains Mono,monospace;font-size:11px;color:{C["text_dim"]}">{row[1]}</td>'
            f'<td style="font-family:JetBrains Mono,monospace;font-size:11px">{row[2]}</td>'
            f'<td style="font-family:JetBrains Mono,monospace;font-size:11px;color:{C["text_dim"]}">{row[3]}</td>'
            f'<td><span style="background:{"#dcfce7" if row[4]!="ns" else "#f1f5f9"};color:{"#16a34a" if row[4]!="ns" else C["text_dim"]};padding:2px 9px;border-radius:5px;font-size:10px;font-weight:700;font-family:JetBrains Mono,monospace">{row[4]}</span></td></tr>'
            for row in [
              ("Retail & Recreation Δ", "−10.75%", "0.878", "0.0103", "**", C['blue']),
              ("Grocery & Pharmacy Δ", "−2.91%",  "0.488", "0.0502", "*",  C['teal']),
              ("Workplaces Δ",          "−32.90%", "0.544", "0.2424", "ns", C['text']),
              ("Residential Δ",         "+11.45%", "−1.662","0.3296", "ns", C['text']),
              ("Avg. Household Income", "$76,552", "≈0",    "0.6809", "ns", C['text']),
            ]])}
        </tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<hr class="sec-divider">
<div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;padding:4px 0 16px">
  <span style="color:{C['slate_lt']};font-size:10px;font-family:JetBrains Mono,monospace">Transportation Research Part C · Vol. 175 · 2025 · Article 105122</span>
  <span style="color:{C['slate_lt']};font-size:10px">Keywords: Mobility · Resilience · DQQ · ODE · COVID-19 · Transit</span>
</div>
""", unsafe_allow_html=True)
