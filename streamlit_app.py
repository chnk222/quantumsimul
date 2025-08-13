# app.py
import numpy as np
import streamlit as st
from scipy.special import sph_harm, assoc_laguerre
import plotly.graph_objects as go

# -----------------------------
# Page and Styling
# -----------------------------
st.set_page_config(page_title="Quantum Atom Simulator", page_icon="🧪", layout="wide")

st.markdown(
    """
    <style>
    .main {background: radial-gradient(1200px 800px at 80% 10%, #0b1220 0%, #070b14 40%, #05080f 100%);}
    .stApp {color: #e6edf7; font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica, Arial;}
    h1, h2, h3 {letter-spacing: 0.5px;}
    .info-card {background: linear-gradient(135deg, rgba(20,40,70,.65), rgba(8,16,30,.65)); border: 1px solid rgba(140,180,255,.18); padding: 14px 18px; border-radius: 14px;}
    .stSlider > div[data-baseweb='slider'] div[role='slider']{ box-shadow: 0 0 0 4px rgba(120,160,255,.2);}
    .stSlider label, .stSelectbox label, .stRadio label {font-weight: 500; color: #b9c7e6}
    .stTabs [data-baseweb='tab']{ color: #b9c7e6}
    .tight-row > div {padding-right: 0.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Physics helpers
# -----------------------------
def radial_wavefunction(r, n, l, Z=1, a0=5.29177210903e-11, normalize=True):
    """
    Hydrogen-like radial wavefunction R_{n,l}(r), vectorized.
    Normalization integrates over r^2 dr on 1D r to avoid ambiguity.
    """
    if n < 1 or l < 0 or l >= n:
        return np.zeros_like(r)

    r_arr = np.asarray(r)
    r_1d = r_arr.reshape(-1)  # flatten for stable integration

    rho = 2.0 * Z * r_1d / (n * a0)
    k = n - l - 1
    alpha = 2*l + 1

    from math import factorial
    prefac = (2.0 * Z / (n * a0))**1.5
    Cnl = np.sqrt(factorial(n - l - 1) / (2.0*n * factorial(n + l)))

    L_vals = assoc_laguerre(rho, k, alpha)
    R_1d = (prefac * Cnl) * np.exp(-rho/2.0) * (rho**l) * L_vals
    R_1d = np.nan_to_num(R_1d, nan=0.0, posinf=0.0, neginf=0.0)

    if normalize and r_1d.size > 1:
        dr = r_1d[1] - r_1d[0]
        norm = np.sqrt(np.trapz((np.abs(R_1d)**2) * (r_1d**2), dx=dr))
        if np.isfinite(norm) and norm > 0:
            R_1d = R_1d / norm

    return R_1d.reshape(r_arr.shape)

def hydrogen_orbital(n, l, m, r_grid, theta_grid, phi_grid, Z=1, a0=5.29177210903e-11):
    """
    ψ_{n,l,m}(r,θ,φ) = R_{n,l}(r) * Y_l^m(θ,φ).
    SciPy's sph_harm signature: sph_harm(m, l, phi, theta).
    """
    R_1d = radial_wavefunction(r_grid, n, l, Z=Z, a0=a0, normalize=True)  # (Nr,)
    R = R_1d[:, None, None]
    Y = sph_harm(m, l, phi_grid[None, None, :], theta_grid[None, :, None])  # (1, Nθ, Nφ)
    psi = R * Y
    return psi

# Shell/element label for n
N_SHELL_LABEL = {
    1: ("K-shell (1s)", "Hydrogen"),
    2: ("L-shell (2s/2p)", "Oxygen"),
    3: ("M-shell (3s/3p/3d)", "Sodium"),
    4: ("N-shell (4s/4p/4d)", "Potassium"),
    5: ("O-shell (5s/5p/5d)", "Rubidium"),
    6: ("P-shell (6s/6p/6d)", "Cesium"),
    7: ("Q-shell (7s/7p/7d)", "Francium"),
    8: ("R-shell (8s/8p)", "Hypothetical"),
}

# -----------------------------
# UI controls
# -----------------------------
st.title("Quantum Atom Simulator")
st.caption("Hydrogen-like orbitals with interactive quantum numbers, atoms, and visualization modes")

with st.sidebar:
    st.subheader("Quantum Numbers")
    c1, c2 = st.columns([2, 3])
    with c1:
        n = st.slider("Principal quantum number n", min_value=1, max_value=8, value=2)
    shell_text, elem_text = N_SHELL_LABEL.get(n, ("Shell", "Element"))
    with c2:
        st.text_input("Shell / example element", f"{shell_text} — e.g., {elem_text}", disabled=True)

    l = st.slider("Azimuthal quantum number l", min_value=0, max_value=max(0, n-1), value=min(1, n-1))
    m = st.slider("Magnetic quantum number m", min_value=-l, max_value=l, value=0)

    st.subheader("Atom & Scale")
    Z = st.slider("Atomic number Z (hydrogen-like)", 1, 10, 1)
    a0 = 5.29177210903e-11  # meters

    st.subheader("Visualization")
    view_mode = st.selectbox(
        "Viewing mode",
        [
            "3D volume + point cloud",
            "Radial probability P(r)",
            "Angular density |Y_l^m(θ,φ)|^2 map",
        ],
    )
    grid_detail = st.select_slider("Detail level", options=["Low", "Medium", "High"], value="Medium")
    cmap = st.selectbox("Colormap", ["viridis", "plasma", "magma", "inferno", "cividis", "turbo", "icefire"])

# Grid resolution
if grid_detail == "Low":
    Nr, Nth, Nph = 80, 60, 60
elif grid_detail == "High":
    Nr, Nth, Nph = 160, 120, 120
else:
    Nr, Nth, Nph = 120, 90, 90

# Spatial grid (cutoff capturing most probability ~ n^2 a0/Z)
Rmax = 12 * (n**2) * (a0 / max(Z, 1))
r = np.linspace(0, Rmax, Nr)
theta = np.linspace(1e-6, np.pi - 1e-6, Nth)  # avoid singular ends
phi = np.linspace(0, 2*np.pi, Nph, endpoint=False)

# Compute orbital and density
psi = hydrogen_orbital(n, l, m, r, theta, phi, Z=Z, a0=a0)
psi2 = np.abs(psi)**2

# -----------------------------
# Visualizations
# -----------------------------
if view_mode == "Radial probability P(r)":
    # P(r) = 4π r^2 |R_{n,l}(r)|^2
    Rnl = radial_wavefunction(r, n, l, Z=Z, a0=a0, normalize=True)
    P_r = 4*np.pi * (r**2) * (np.abs(Rnl)**2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r*1e10, y=P_r, mode='lines', line=dict(color='#84a9ff')))
    fig.update_layout(
        template='plotly_dark',
        xaxis_title='r (Å)',
        yaxis_title='P(r)',
        title=f'Radial Probability Distribution P(r) for n={n}, l={l}, Z={Z}',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

elif view_mode == "Angular density |Y_l^m(θ,φ)|^2 map":
    # Angular map independent of r
    Y = sph_harm(m, l, phi[None, :], theta[:, None])
    Y2 = np.abs(Y)**2

    fig = go.Figure(data=[go.Heatmap(
        z=Y2,
        x=phi,
        y=theta,
        colorscale=cmap,
        colorbar=dict(title='|Y|^2')
    )])
    fig.update_layout(
        template='plotly_dark',
        xaxis_title='φ',
        yaxis_title='θ',
        title=f'Angular Density |Y_l^m(θ,φ)|^2 for n={n}, l={l}, m={m}',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    # 3D volume + point cloud
    Rg, Tg, Pg = np.meshgrid(r, theta, phi, indexing='ij')
    X = Rg * np.sin(Tg) * np.cos(Pg)
    Yc = Rg * np.sin(Tg) * np.sin(Pg)
    Zc = Rg * np.cos(Tg)

    dens = psi2
    maxd = dens.max() if np.isfinite(psi2).any() else 1.0
    dens_norm = dens / (maxd + 1e-18)

    # Downsample for speed
    step = 2 if grid_detail != "High" else 1
    Xs = X[::step, ::step, ::step]
    Ys = Yc[::step, ::step, ::step]
    Zs = Zc[::step, ::step, ::step]
    Cs = dens_norm[::step, ::step, ::step]

    # Volume trace
    fig = go.Figure()
    fig.add_trace(go.Volume(
        x=Xs.flatten()*1e10,
        y=Ys.flatten()*1e10,
        z=Zs.flatten()*1e10,
        value=Cs.flatten(),
        opacity=0.06,          # slightly reduced to reveal points
        isomin=0.30,           # show higher probability regions
        isomax=1.0,
        surface_count=5,       # modest count to avoid heavy fog
        colorscale=cmap,
        caps=dict(x_show=False, y_show=False, z_show=False),
        name="Probability volume"
    ))

    # Sample point cloud from density and add AFTER the volume so it renders on top
    rng = np.random.default_rng(0)
    prob = Cs / (Cs.sum() + 1e-18)
    n_points = min(20000, Cs.size)  # cap for performance
    idx = rng.choice(Cs.size, size=n_points, replace=False, p=prob.ravel())
    px = Xs.ravel()[idx] * 1e10
    py = Ys.ravel()[idx] * 1e10
    pz = Zs.ravel()[idx] * 1e10

    fig.add_trace(go.Scatter3d(
        x=px, y=py, z=pz,
        mode='markers',
        marker=dict(
            size=2,
            color='rgba(255,255,255,0.95)',  # bright, nearly opaque points
            line=dict(width=0),
        ),
        name='Point cloud'
    ))

    fig.update_layout(
        template='plotly_dark',
        scene=dict(
            xaxis_title='x (Å)', yaxis_title='y (Å)', zaxis_title='z (Å)',
            aspectmode='data',
            bgcolor='rgba(0,0,0,0)'
        ),
        title=f'3D Electron Probability Density for n={n}, l={l}, m={m}, Z={Z}',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Info section
# -----------------------------
with st.expander("Wavefunction details"):
    st.markdown(
        """
        The hydrogen-like orbital factorizes as ψₙₗₘ(r,θ,φ) = Rₙₗ(r) · Yₗᵐ(θ,φ).

        - Radial part Rₙₗ(r): associated Laguerre polynomials L^{(2l+1)}_{n−l−1}(ρ) with ρ = 2Zr/(na₀) and exponential decay e^{-ρ/2}.
        - Angular part Yₗᵐ(θ,φ): spherical harmonics determine angular nodes and orientation set by m.
        - Radial probability: P(r) = 4π r² |Rₙₗ(r)|²; most-likely radii scale roughly with n² a₀/Z.
        - Implemented with SciPy’s assoc_laguerre and sph_harm for stable, vectorized evaluation.
        """
    )

with st.expander("Tips"):
    st.markdown(
        """
        - Increase Z for hydrogen-like ions (He⁺, Li²⁺, …); orbitals shrink ∝1/Z.
        - l ranges 0..n−1; m ranges from −l..+l.
        - Use Radial probability to locate peaks (shell radii); peaks move outward with n.
        - In 3D, points are intentionally drawn after the volume and with bright color so they remain visible.
        """
    )

st.markdown(
    "<div class='info-card'>Built with Streamlit, NumPy, SciPy, and Plotly · Hydrogen-like (single-electron) model</div>",
    unsafe_allow_html=True
)
