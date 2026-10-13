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

const _FLUX_UNIT = u"GeV^-1*cm^-2*s^-1*sr^-1"

"""
    Φ_μ(E, costheta, h) -> Quantity

Atmospheric muon flux (μ⁺ + μ⁻), returned in GeV⁻¹ cm⁻² s⁻¹ sr⁻¹.

- `E`        : muon kinetic energy, a Unitful energy (valid range: 100 GeV–1e8 GeV)
- `costheta` : cosine of local zenith angle (0 = horizontal, 1 = vertical)
- `h`        : observation altitude above sea level, a Unitful length (valid range: 0–50 km)

Plain numbers are also accepted for `E` and `h`, interpreted as GeV and km.

Returns zero flux outside the interpolation grid.
"""
function Φ_μ(E_GeV::Float64, costheta::Float64, h_km::Float64)
    _INTERP[](log10(E_GeV), costheta, h_km) * _FLUX_UNIT
end

Φ_μ(E::Unitful.Energy, ct::Real, h::Unitful.Length) =
    Φ_μ(Float64(ustrip(u"GeV", E)), Float64(ct), Float64(ustrip(u"km", h)))

Φ_μ(E::Real, ct::Real, h::Real) = Φ_μ(Float64(E), Float64(ct), Float64(h))
