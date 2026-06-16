"""
Generate MCEq atmospheric muon flux tables for TamboAtmMuonFlux.jl.

Output: ../data/mceq_flux_tables.h5
  axes:
    E_GeV      [N_E]              MCEq's native energy grid
    cos_theta  [N_theta]          linear, 0.0–1.0
    h_km       [N_h]              altitude above sea level in km

  dataset:
    flux [N_E, N_theta, N_h]     GeV⁻¹ cm⁻² s⁻¹ sr⁻¹  (mu+ + mu- summed)

Usage:
    conda run -n py310 python generate_tables.py
"""

import numpy as np
import h5py
import os

from MCEq.core import MCEqRun
import crflux.models as pm

# ── Grid ──────────────────────────────────────────────────────────────────────
N_theta = 40
N_h     = 30

cos_theta = np.linspace(0.0, 1.0, N_theta)   # 0=horizontal, 1=vertical
h_km      = np.linspace(0.0, 50.0, N_h)      # 0–50 km, ~1.7 km spacing

# ── MCEq setup ────────────────────────────────────────────────────────────────
mceq = MCEqRun(
    interaction_model="SIBYLL23D",
    primary_model=(pm.HillasGaisser2012, "H3a"),
    theta_deg=0.0,
)

# Use MCEq's native energy grid — no interpolation, no resampling
E_GeV = mceq.e_grid  # set after first init
N_E   = len(E_GeV)
print(f"MCEq native energy grid: {N_E} points, {E_GeV[0]:.1f}–{E_GeV[-1]:.2e} GeV")

flux = np.zeros((N_E, N_theta, N_h))

for i_t, ct in enumerate(cos_theta):
    theta_deg = float(np.degrees(np.arccos(ct)))
    print(f"cos_theta={ct:.2f}  theta={theta_deg:.1f}°", flush=True)

    mceq.set_theta_deg(theta_deg)

    # Convert altitudes (km) to slant depth X (g/cm²). h2X takes height in cm.
    # h2X(h_cm) = total slant depth above altitude h: X=0 at TOA, X=max_X at sea level.
    dm = mceq.density_model
    X_obs = np.array([dm.h2X(h * 1e5) for h in h_km])  # km → cm → X
    X_obs = np.clip(X_obs, 0.01, None)  # MCEq requires > X_start

    sort_idx = np.argsort(X_obs)
    X_sorted = X_obs[sort_idx]

    mceq.solve(int_grid=list(X_sorted), grid_var="X")

    # Retrieve flux at each altitude, mapping back to original h_km order
    for i_h, orig_i in enumerate(np.argsort(sort_idx)):
        mu_p = mceq.get_solution("mu+", mag=0, grid_idx=int(orig_i))
        mu_m = mceq.get_solution("mu-", mag=0, grid_idx=int(orig_i))
        flux[:, i_t, i_h] = mu_p + mu_m  # directly on mceq.e_grid — no interpolation

    print(f"  peak at ground: {flux[:, i_t, 0].max():.3e} GeV⁻¹cm⁻²s⁻¹sr⁻¹")

# ── Write HDF5 ────────────────────────────────────────────────────────────────
out_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "mceq_flux_tables.h5")
)
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with h5py.File(out_path, "w") as f:
    f.create_dataset("E_GeV",     data=E_GeV)
    f.create_dataset("cos_theta", data=cos_theta)
    f.create_dataset("h_km",      data=h_km)
    f.create_dataset("flux",      data=flux, compression="gzip", compression_opts=6)
    f.attrs["units"]             = "GeV^-1 cm^-2 s^-1 sr^-1"
    f.attrs["interaction_model"] = "SIBYLL23D"
    f.attrs["primary_model"]     = "HillasGaisser2012 H3a"
    f.attrs["particle"]          = "mu+ + mu-"

print(f"\nSaved: {out_path}")
print(f"Shape: flux{flux.shape}")
print(f"Peak flux: {flux.max():.3e} GeV^-1 cm^-2 s^-1 sr^-1")
