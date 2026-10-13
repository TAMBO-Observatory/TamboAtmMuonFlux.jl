module TamboAtmMuonFluxTamboSimExt

using TamboAtmMuonFlux
using TamboSim
using Unitful
import LinearAlgebra: norm, dot

"""
    Φ_μ(q::TamboSim.Frame) -> Quantity

Atmospheric muon flux for a TamboSim Frame event (GeV⁻¹ cm⁻² s⁻¹ sr⁻¹).

Reads energy and direction from `q["injection_initial_state"]` and altitude
from `q["injection"]["altitude"]` (km), which is always snapshotted onto the
M frame by CosmicRayInjection.
"""
function TamboAtmMuonFlux.Φ_μ(q::TamboSim.Frame)
    p     = q["injection_initial_state"]
    cosθ  = _local_zenith(p)
    h     = Float64(q["injection"]["altitude"]) * u"km"
    Φ_μ(p.energy, cosθ, h)
end

# cos of local zenith: dot(-direction, r̂) where r̂ points radially outward.
# The particle is stored in the detector (site-local ENU) frame, so convert to
# Earth-centered coordinates before taking the radial unit vector.
function _local_zenith(p)
    pos = ustrip.(u"m", convert(ecefcoordinates, p.position).point)
    dir = convert(ecefcoordinates, p.direction).point
    r̂ = pos / norm(pos)
    clamp(-dot(dir, r̂), 0.0, 1.0)
end

end
