module TamboAtmMuonFluxTamboSimExt

using TamboAtmMuonFlux
using TamboSim
using Unitful
import LinearAlgebra: norm, dot

# TAMBO terrain elevation hardcoded until cr-injection-altitude-snapshot merged
const TAMBO_H_KM = 3.0

"""
    Φ_μ(q::TamboSim.Frame) -> Float64

Atmospheric muon flux for a TamboSim Frame event (GeV⁻¹ cm⁻² s⁻¹ sr⁻¹).

Reads energy and direction from `q["injection_initial_state"]`.
Altitude is fixed at 3.0 km (TAMBO terrain elevation).
"""
function TamboAtmMuonFlux.Φ_μ(q::TamboSim.Frame)
    p = q["injection_initial_state"]
    E_GeV   = ustrip(u"GeV", p.energy)
    cosθ    = _local_zenith(p)
    Φ_μ(E_GeV, cosθ, TAMBO_H_KM)
end

# cos of local zenith: dot(-direction, r̂) where r̂ points radially outward
function _local_zenith(p)
    pos = p.position.point
    dir = p.direction.point
    r̂ = pos / norm(pos)
    clamp(-dot(dir, r̂), 0.0, 1.0)
end

end
