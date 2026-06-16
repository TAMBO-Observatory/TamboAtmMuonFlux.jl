using HDF5
using Interpolations
using Unitful

# Path to the pre-generated MCEq table (relative to this source file)
const _TABLE_PATH = joinpath(@__DIR__, "..", "data", "mceq_flux_tables.h5")

# Interpolant built once at module load time
const _INTERP = Ref{Any}(nothing)

function _load_interp()
    h5open(_TABLE_PATH, "r") do f
        E_GeV     = read(f["E_GeV"])      # [N_E]
        cos_theta = read(f["cos_theta"])  # [N_theta]
        h_km      = read(f["h_km"])       # [N_h]
        flux      = read(f["flux"])       # [N_E, N_theta, N_h]

        log10E = log10.(E_GeV)

        # HDF5 reads Python (N_E, N_theta, N_h) arrays as Julia (N_h, N_theta, N_E)
        # due to row-major vs column-major layout, so permute back to (N_E, N_theta, N_h)
        flux_jl = permutedims(flux, (3, 2, 1))

        # Build a 3D linear interpolant on (log10(E), cos_theta, h_km)
        itp = interpolate(
            (log10E, cos_theta, h_km),
            flux_jl,
            Gridded(Linear()),
        )
        # Extrapolate to zero outside the grid
        _INTERP[] = extrapolate(itp, 0.0)
    end
end

function __init__()
    _load_interp()
end

"""
    Φ_μ(E_GeV, costheta, h_km) -> Float64

Atmospheric muon flux (μ⁺ + μ⁻) in GeV⁻¹ cm⁻² s⁻¹ sr⁻¹.

- `E_GeV`    : muon kinetic energy in GeV (valid range: 100–1e8)
- `costheta` : cosine of local zenith angle (0 = horizontal, 1 = vertical)
- `h_km`     : observation altitude above sea level in km (valid range: 0–20)

Returns 0.0 outside the interpolation grid.
"""
function Φ_μ(E_GeV::Float64, costheta::Float64, h_km::Float64)::Float64
    _INTERP[](log10(E_GeV), costheta, h_km)
end

# Convenience: accept any Real arguments
Φ_μ(E, ct, h) = Φ_μ(Float64(E), Float64(ct), Float64(h))
