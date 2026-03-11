## DQQ Mobility Resilience Dashboard

This repository contains a **Streamlit** dashboard that visualizes the
_Double Quadratic Queue (DQQ)_ model for mobility resilience and recovery,
together with key empirical findings (H1–H4) from U.S. transit data.

The app is designed as an interactive, paper-style companion for:

> Fang Tang, Xiangyong Luo, Xuesong (Simon) Zhou  
> *Mobility Resilience and Recovery Dynamics: Parsimonious Framework Beyond V-shapes*  
> Transportation Research Part C, 2025, Article 105122.

---

### 1. Features

- **Interactive DQQ model tab**
  - Smooth double-quadratic rate function \( \pi(t) \) and its integral \( Q(t) \).
  - Scenario toggle: **Transition Loss / Normal Recovery / Transition Gain**.
  - Optional overlay of classical V-shaped recovery for comparison.
- **Overview tab**
  - Research motivation, four mobility phases, and key resilience metrics (D, R, r, \(Q_{\max}\), Δ, RTA).
- **Performance metrics tab**
  - Bar charts for nationwide model fit (R², RMSE) and spatial-resolution comparisons.
  - Disruption / recovery durations by mobility category.
- **Findings tab**
  - H3: Income vs. recovery rapidity scatter with 99% CI band.
  - H4: H Line opening impact on RapidRide clusters, month-by-month \( \Delta \pi(t) \).
  - Hypothesis verdicts and regression summary table.

---

### 2. Installation

#### 2.1 Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

#### 2.2 Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

#### 2.3 Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` includes core packages such as:

- `streamlit`
- `plotly`
- `pandas`
- `numpy`

---

### 3. Run the dashboard

From the project root:

```bash
streamlit run dqq_dashboard.py
```

By default Streamlit will open the app in your browser at
`http://localhost:8501`.

---

### 4. File structure

- `dqq_dashboard.py` – main Streamlit app:
  - global styling (CSS, color palette, layout)
  - DQQ and V-shaped trajectory generators
  - four tabs (`Overview`, `DQQ Model`, `Performance Metrics`, `Findings`)
- `requirements.txt` – Python dependencies needed to run the dashboard.

You can add any figures (e.g., screenshots from the paper) under an `assets/`
folder and reference them from the app or GitHub README as needed.

---

### 5. Citation

If you use this dashboard or the underlying model in academic work, please cite:

> Tang, F., Luo, X., Zhou, X. (2025).  
> Mobility Resilience and Recovery Dynamics: Parsimonious Framework Beyond V-shapes.  
> *Transportation Research Part C: Emerging Technologies*, 175, 105122.  
> DOI: 10.1016/j.trc.2025.105122

