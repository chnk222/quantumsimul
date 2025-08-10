# app.py
import os
import io
import numpy as np
import streamlit as st
from PIL import Image

# Math for hydrogenic orbitals
from sympy import symbols, sqrt, exp, factorial, assoc_laguerre, S
from sympy import N as symN
from scipy.special import sph_harm

# 3D rendering
import pyvista as pv

# ----------------------------
# UI THEME AND PAGE SETTINGS
# ----------------------------
st.set_page_config(page_title="Quantum Atom Orbital Simulator", page_icon="🔬", layout="wide")

# Minimal futuristic styling
st.markdown("""
<style>
/* Futuristic minimal styling */
:root {
  --bg: #0e1117;
  --panel: #151a23;
  --accent: #6ee7ff;
  --text: #e6edf3;
  --muted: #9aa4b2;
}
html, body, [class*="css"] { background-color: var(--bg) !important; color: var(--text) !important; }
section.main > div { padding-top: 1rem; }
.block-container { padding-top: 1rem; }
h1, h2, h3 { color: var(--text); letter-spacing: 0.02em; }
small, p, label { color: var(--muted); }
div[data-baseweb="select"] { color: black; }
.stSlider > div > div > div { background: linear-gradient(90deg, #3a3f4b, #232833); }
.stButton>button {
  background: linear-gradient(90deg, #1c2430, #0f1722);
  color: var(--text);
  border: 1px solid #2b3444; border-radius: 10px;
}
.panel-card {
  background: var(--panel);
  border: 1px solid #263142;
  border-radius: 14px;
  padding: 14px 16px;
}
hr { border: none; border-top: 1px solid #243042; margin: 0.6rem 0 1rem; }
.footer-note { font-size: 0.85rem; color: var(--muted); }
.code-tag {
  display: inline-block; padding: 2px 6px; border-radius: 6px;
  background: #0f1622; border: 1px solid #263142; color: #a8b3c4;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# HELPER: HYDROGENIC ORBITAL
# ----------------------------
# Atomic units: a0=1. For hydrogen-like ion with nuclear charge Z, scale radius by a0/Z.
def radial_wavefunction(n, l, r, Z=1.0, a0=1.0):
    """
    R_{n,l}(r) for hydrogen-like atom in atomic units (a0=1 unless changed).
    Returns real-valued radial part.
    """
    # Ensure valid quantum numbers
    if n <= 0 or l < 0 or l >= n:
        return np.zeros_like(r)

    rho = 2.0 * Z * r / a0 / n
    # Normalization constant (standard hydrogenic form)
    # R_{n,l}(r) = (2Z/na0)^{3/2} * sqrt( (n-l-1)! / [2n (n+l)!] ) * e^{-rho/2} * rho^l * L_{n-l-1}^{2l+1}(rho)
    # Use sympy for associated Laguerre with numeric evaluation.
    n_sym, l_sym, rho_sym = symbols('n l rho')
    # Precompute constants numerically
    prefac = (2.0 * Z / (n * a0))**1.5
    num = symN(factorial(n - l - 1))
    den = symN(2*n * factorial(n + l))
    Cnl = symN(sqrt(num / den))
    # Evaluate polynomial for vector rho using sympy then convert to numpy
    # To keep performance reasonable, compute the polynomial coefficients once
    # With sympy's assoc_laguerre(k, alpha, x) where k=n-l-1, alpha=2l+1
    k = n - l - 1
    alpha = 2*l + 1
    # Create a fast lambda for polynomial by sampling via direct sympy eval on array
    # For speed, we evaluate per-array using list comprehension (works fine up to moderate grid sizes)
    L_vals = np.array([float(symN(assoc_laguerre(k, alpha, float(rv)))) for rv in rho], dtype=float)

    R = float(prefac * Cnl) * np.exp(-rho/2.0) * (rho**l) * L_vals
    return R

def real_spherical_harmonic(l, m, theta, phi):
    """
    Real form of spherical harmonics:
    - m > 0: sqrt(2) * Re(Y_l^m)
    - m = 0: Y_l^0
    - m < 0: sqrt(2) * Im(Y_l^{|m|})
    Uses scipy.special.sph_harm which returns complex Y_l^m(phi, theta) with convention (m, l, phi, theta).
    """
    # scipy.sph_harm signature: sph_harm(m, l, phi, theta)
    if m > 0:
        y = sph_harm(m, l, phi, theta)
        return np.sqrt(2.0) * y.real
    elif m == 0:
        y = sph_harm(0, l, phi, theta)
        return y.real
    else:  # m < 0
        y = sph_harm(-m, l, phi, theta)
        return np.sqrt(2.0) * y.imag

def hydrogenic_orbital(n, l, m, Z=1.0, a0=1.0, grid_N=120, r_max_bohr=25.0):
    """
    Compute ψ_nlm(r,θ,φ) on a 3D Cartesian grid, returning:
      - grid (pyvista.UniformGrid) with scalars 'psi_real' and 'psi_prob' (|ψ|^2)
    We build ψ as real combination using real spherical harmonics.
    """
    # Build spherical grid from Cartesian grid for evaluation
    # Choose a cubic volume with extent based on n and Z to show relevant density.
    # Typical radius scale ~ a0 * n^2 / Z
    scale = (n**2) / max(Z, 1e-6)
    lim = min(r_max_bohr, 8.0 * scale)  # cap for big n
    N = grid_N

    xs = np.linspace(-lim, lim, N)
    ys = np.linspace(-lim, lim, N)
    zs = np.linspace(-lim, lim, N)
    X, Y, Zc = np.meshgrid(xs, ys, zs, indexing='ij')

    r = np.sqrt(X**2 + Y**2 + Zc**2)
    # Avoid division by zero in angles
    theta = np.arccos(np.clip(np.where(r>0, Zc/r, 1.0), -1.0, 1.0))  # polar angle [0, π]
    phi = np.mod(np.arctan2(Y, X), 2*np.pi)                           # azimuth [0, 2π)

    # Radial part
    R = radial_wavefunction(n, l, r, Z=Z, a0=a0)

    # Angular part (real Y_lm)
    Ylm = real_spherical_harmonic(l, m, theta, phi)

    psi = R * Ylm  # real-valued orbital representation
    prob = psi**2  # |ψ|^2 since psi is real here

    # Create PyVista UniformGrid
    spacing = (xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0])
    origin = (xs[0], ys[0], zs[0])
    grid = pv.UniformGrid()
    grid.dimensions = np.array(psi.shape) + 1  # dims are points, add 1
    grid.origin = origin
    grid.spacing = spacing

    # Add scalars
    grid.cell_data["psi_real"] = psi.ravel(order='F')
    grid.cell_data["psi_prob"] = prob.ravel(order='F')

    return grid, lim

# ----------------------------
# SIDEBAR CONTROLS
# ----------------------------
st.title("Quantum Hydrogen-Like Atom Orbital Simulator")

with st.sidebar:
    st.markdown("### Controls")
    # Atom selection (hydrogen-like ions via Z)
    Z = st.slider("Nuclear charge Z (Hydrogen=1)", min_value=1, max_value=10, value=1, help="Hydrogen-like ions: Z=1(H), 2(He⁺), 3(Li²⁺), ...")

    # Quantum numbers
    n = st.slider("Principal quantum number n", min_value=1, max_value=8, value=3)
    l = st.slider("Azimuthal quantum number l", min_value=0, max_value=n-1, value=min(1, n-1))
    m = st.slider("Magnetic quantum number m", min_value=-l, max_value=l, value=0)

    st.caption("Quantum number constraints: 0 ≤ l ≤ n-1, and -l ≤ m ≤ l.")

    # Visualization settings
    st.markdown("### Visualization")
    view_mode = st.selectbox(
        "Viewing mode",
        ["Volume (probability cloud)", "Isosurface (phase colors)", "Point density (sampled)"],
        index=0
    )
    colormap = st.selectbox("Colormap", ["magma", "plasma", "viridis", "coolwarm", "RdBu"])
    grid_N = st.slider("Grid resolution (N³)", 60, 160, 100, step=10)
    opacity_scale = st.slider("Opacity strength", 1, 100, 40, help="Higher → denser appearance for volume/points.")
    iso_level = st.slider("Isosurface level (relative)", 1, 99, 50, help="Percentile for |ψ| used for isosurface.")
    clip_axis = st.selectbox("Clip plane", ["None", "+X", "-X", "+Y", "-Y", "+Z", "-Z"], index=0)

    st.markdown("### Camera and Aesthetics")
    bg_color = st.color_picker("Background", "#000000")
    show_axes = st.checkbox("Show axes", value=False)
    zoom = st.slider("Zoom", 1.0, 3.0, 1.6, step=0.1)

# ----------------------------
# COMPUTE ORBITAL
# ----------------------------
with st.spinner("Computing orbital..."):
    grid, lim = hydrogenic_orbital(n, l, m, Z=float(Z), a0=1.0, grid_N=grid_N)

# ----------------------------
# RENDER WITH PYVISTA (OFF-SCREEN), CAPTURE IMAGE
# ----------------------------
def normal_from_choice(choice):
    if choice == "None":
        return None
    return {
        "+X": (1,0,0), "-X": (-1,0,0),
        "+Y": (0,1,0), "-Y": (0,-1,0),
        "+Z": (0,0,1), "-Z": (0,0,-1)
    }[choice]

pv.set_plot_theme("document")
plotter = pv.Plotter(off_screen=True, window_size=(1100, 800))
plotter.set_background(bg_color)

added_actor = None
if view_mode == "Volume (probability cloud)":
    # Volume rendering using probability density as opacity
    # Build RGBA mapping: color by sign(psi_real) to show phase change; opacity ~ |ψ|
    vals = grid.cell_data["psi_real"]
    prob = grid.cell_data["psi_prob"]
    neg_mask = vals < 0
    rgba = np.zeros((vals.size, 4), dtype=np.uint8)
    rgba[neg_mask, 0] = 255  # red for negative
    rgba[~neg_mask, 1] = 255 # green for positive
    # Normalize opacity
    opac = prob / (prob.max() + 1e-12)
    opac = np.clip(opac**(opacity_scale/40.0), 0, 1)  # adjust density
    rgba[:, 3] = (opac * 255).astype(np.uint8)

    grid_copy = grid.copy()
    grid_copy.cell_data["rgba"] = rgba
    added_actor = plotter.add_volume(grid_copy, scalars="rgba")
elif view_mode == "Isosurface (phase colors)":
    # Isosurface of |psi_real| at chosen percentile, color by sign
    psi = grid.cell_data["psi_real"]
    level = np.percentile(np.abs(psi), iso_level)
    # Marching cubes works on point data; convert cell to point data
    gpt = grid.cell_data_to_point_data()
    abs_psi = np.abs(gpt.point_data["psi_real"])
    iso = gpt.contour([level], scalars=abs_psi)

    # Phase color by sign of psi
    sign = (gpt.point_data["psi_real"] >= 0).astype(int)
    # Map to two colors via a simple bit: 0 -> red, 1 -> green
    iso.point_data["phase"] = sign
    added_actor = plotter.add_mesh(iso, scalars="phase", cmap=["red", "green"], smooth_shading=True, show_scalar_bar=False, ambient=0.25, specular=0.4)
else:
    # Point density sampling according to probability
    # Convert to point data for sampling
    gpt = grid.cell_data_to_point_data()
    prob = gpt.point_data["psi_prob"].copy()
    prob = prob / (prob.sum() + 1e-12)
    n_samples = int(60000 * (opacity_scale/40.0))  # sample count scaled
    rng = np.random.default_rng(0)
    idx = rng.choice(gpt.n_points, size=min(n_samples, gpt.n_points), replace=False, p=prob)
    pts = gpt.points[idx]
    # Add small jitter
    pts = pts + (rng.random(pts.shape) - 0.5) * (gpt.spacing[0] if hasattr(gpt, "spacing") else (2*lim/grid_N))*0.8

    cloud = pv.PolyData(pts)
    phase = (gpt.point_data["psi_real"][idx] >= 0).astype(int)
    cloud["phase"] = phase
    # Render as spheres glyphs for nicer look
    d = cloud.glyph(geom=pv.Sphere(theta_resolution=8, phi_resolution=8, radius=0.02*lim/grid_N), scale=False)
    added_actor = plotter.add_mesh(d, scalars="phase", cmap=["#ff4d4d", "#5cff5c"], show_scalar_bar=False, smooth_shading=True, ambient=0.3)

# Optional clip plane
normal = normal_from_choice(clip_axis)
if normal is not None and added_actor is not None:
    try:
        plotter.add_volume_clip_plane(added_actor, normal=normal, normal_rotation=False)
    except Exception:
        # For meshes, use a mesh clip
        if isinstance(added_actor, pv.Actor):
            try:
                plotter.remove_actor(added_actor)
            except Exception:
                pass

# Camera and axes
plotter.camera.zoom(zoom)
if show_axes:
    plotter.show_axes()

# Take screenshot
img = plotter.screenshot(return_img=True)
plotter.close()

# ----------------------------
# LAYOUT: VIEW + INFO
# ----------------------------
left, right = st.columns([2, 1])

with left:
    st.markdown("## Orbital View")
    st.image(img, use_column_width=True)
    st.caption(f"Z={Z}, n={n}, l={l}, m={m} — Mode: {view_mode} — Colormap: {colormap}")

with right:
    st.markdown("## Wavefunction Info")
    st.markdown(f"""
- Principal number n: {n}  
- Azimuthal number l: {l}  
- Magnetic number m: {m}  
- Nuclear charge Z: {Z}  
- Extent: ±{lim:.2f} a₀  
- Rendering: {view_mode}  
    """)

    st.markdown("### What is being shown?")
    st.markdown("""
- The electron orbital is visualized from the hydrogen-like solution ψₙₗₘ(r,θ,φ) = Rₙₗ(r) Yₗᵐ(θ,φ).  
- The radial part Rₙₗ(r) is the analytic hydrogenic function with Laguerre polynomials.  
- The angular part uses real spherical harmonics, producing familiar s, p, d, f shapes.  
- The electron probability density is |ψ|², shown here as:
  - Volume: probability mapped to opacity; red/green indicates phase sign of ψ.  
  - Isosurface: a surface at a chosen |ψ| level; colored by phase sign.  
  - Point density: points sampled with probability ∝|ψ|²; colored by phase sign.  
- Changing Z simulates hydrogen-like ions (He⁺, Li²⁺, …) via radial scaling ~a₀/Z.  
""")

    st.markdown("### Valid quantum numbers")
    st.markdown("""
- Constraints: 0 ≤ l ≤ n-1 and -l ≤ m ≤ l.  
- Larger n expands the cloud roughly with scale ∼n²/Z.  
""")

    st.markdown("### Notes")
    st.markdown("""
- Rendering uses a uniform 3D grid and off‑screen PyVista; performance depends on resolution and system.  
- For subtle nodal features, increase resolution and adjust opacity or isosurface level.  
""")

st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("""
<div class="footer-note">
This visualization approach follows well-known PyVista hydrogen orbital examples (analytic hydrogenic wavefunctions and RGBA/isosurface/point-density rendering) and standard spherical harmonic visualization practices.[5][7][15]
</div>
""", unsafe_allow_html=True)
