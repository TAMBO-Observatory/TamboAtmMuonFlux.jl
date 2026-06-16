"""
Diagnostic plots of mceq_flux_tables.h5.
2x3 grid: rows = (fixed h=3km, fixed cos θ=1), columns = (vs cos θ, vs E, vs altitude)
Energy plots show E³·Φ; altitude/zenith plots show E³·Φ at fixed E.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

table_path = "../data/mceq_flux_tables.h5"

with h5py.File(table_path, "r") as f:
    E_GeV     = f["E_GeV"][:]
    cos_theta = f["cos_theta"][:]
    h_km      = f["h_km"][:]
    flux      = f["flux"][:]   # (N_E, N_theta, N_h)

log10E = np.log10(E_GeV)
interp = RegularGridInterpolator(
    (log10E, cos_theta, h_km), flux,
    method="cubic", bounds_error=False, fill_value=0.0,
)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

E_vals  = [1e2, 1e4, 1e6]
ct_vals = [0.05, 0.3, 0.6, 1.0]
h_vals  = [0.5, 3.0, 10.0, 30.0]
E_fine  = np.logspace(2, 8, 300)
ct_fine = np.linspace(0.01, 1.0, 300)
h_fine  = np.linspace(0.5, 50, 300)
colors  = ['C0', 'C1', 'C2', 'C3']

def e3phi(E, phi):
    """E³ · Φ weighting."""
    return E**3 * phi

# ── Row 0: fixed h = 3 km ────────────────────────────────────────────────────
h_fix = 3.0

# vs cos θ: show E³·Φ at fixed E values
ax = axes[0, 0]
for E, col in zip(E_vals, colors):
    pts = np.column_stack([np.full_like(ct_fine, np.log10(E)), ct_fine, np.full_like(ct_fine, h_fix)])
    ax.semilogy(ct_fine, e3phi(E, interp(pts)), color=col, label=f"E={E:.0e} GeV")
ax.set_xlabel("cos θ")
ax.set_ylabel("E³ · Φ_μ  [GeV² cm⁻² s⁻¹ sr⁻¹]")
ax.set_title(f"vs cos θ  (h={h_fix} km)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# vs Energy: E³·Φ for several zenith angles
ax = axes[0, 1]
for ct, col in zip(ct_vals, colors):
    pts = np.column_stack([np.log10(E_fine), np.full_like(E_fine, ct), np.full_like(E_fine, h_fix)])
    ax.loglog(E_fine, e3phi(E_fine, interp(pts)), color=col, label=f"θ={np.degrees(np.arccos(ct)):.0f}°")
ax.set_xlabel("E (GeV)")
ax.set_ylabel("E³ · Φ_μ  [GeV² cm⁻² s⁻¹ sr⁻¹]")
ax.set_title(f"vs Energy  (h={h_fix} km)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# vs Altitude: E³·Φ at fixed E, several zenith angles
ax = axes[0, 2]
E_alt = 1e3
for ct, col in zip(ct_vals, colors):
    pts = np.column_stack([np.full_like(h_fine, np.log10(E_alt)), np.full_like(h_fine, ct), h_fine])
    ax.semilogy(h_fine, e3phi(E_alt, interp(pts)), color=col, label=f"θ={np.degrees(np.arccos(ct)):.0f}°")
ax.set_xlabel("h (km)")
ax.set_ylabel("E³ · Φ_μ  [GeV² cm⁻² s⁻¹ sr⁻¹]")
ax.set_title(f"vs Altitude  (E={E_alt:.0e} GeV)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── Row 1: fixed cos θ = 1 (vertical) ────────────────────────────────────────
ct_fix = 1.0

# vs Altitude: E³·Φ for several energies
ax = axes[1, 0]
for E, col in zip(E_vals, colors):
    pts = np.column_stack([np.full_like(h_fine, np.log10(E)), np.full_like(h_fine, ct_fix), h_fine])
    ax.semilogy(h_fine, e3phi(E, interp(pts)), color=col, label=f"E={E:.0e} GeV")
ax.set_xlabel("h (km)")
ax.set_ylabel("E³ · Φ_μ  [GeV² cm⁻² s⁻¹ sr⁻¹]")
ax.set_title(f"vs Altitude  (cos θ={ct_fix}, vertical)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# vs Energy: E³·Φ for several altitudes
ax = axes[1, 1]
for h, col in zip(h_vals, colors):
    pts = np.column_stack([np.log10(E_fine), np.full_like(E_fine, ct_fix), np.full_like(E_fine, h)])
    ax.loglog(E_fine, e3phi(E_fine, interp(pts)), color=col, label=f"h={h} km")
ax.set_xlabel("E (GeV)")
ax.set_ylabel("E³ · Φ_μ  [GeV² cm⁻² s⁻¹ sr⁻¹]")
ax.set_title(f"vs Energy  (cos θ={ct_fix}, vertical)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# vs cos θ: E³·Φ for several altitudes at E=1 TeV
ax = axes[1, 2]
E_zen = 1e3
for h, col in zip(h_vals, colors):
    pts = np.column_stack([np.full_like(ct_fine, np.log10(E_zen)), ct_fine, np.full_like(ct_fine, h)])
    ax.semilogy(ct_fine, e3phi(E_zen, interp(pts)), color=col, label=f"h={h} km")
ax.set_xlabel("cos θ")
ax.set_ylabel("E³ · Φ_μ  [GeV² cm⁻² s⁻¹ sr⁻¹]")
ax.set_title(f"vs cos θ  (E={E_zen:.0e} GeV)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── Match y-axis scales within each column ────────────────────────────────────
for col in range(3):
    ylims = [ax.get_ylim() for ax in axes[:, col]]
    ymin = min(y[0] for y in ylims)
    ymax = max(y[1] for y in ylims)
    for ax in axes[:, col]:
        ax.set_ylim(ymin, ymax)

plt.suptitle("TamboAtmMuonFlux diagnostic — E³·Φ_μ  [GeV² cm⁻² s⁻¹ sr⁻¹]", fontsize=12)
plt.tight_layout()
out = "../data/flux_diagnostic.png"
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
