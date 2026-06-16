using Test
using TamboAtmMuonFlux

@testset "TamboAtmMuonFlux" begin

    @testset "basic sanity" begin
        φ = Φ_μ(1e4, 1.0, 3.0)
        @test isfinite(φ)
        @test φ > 0.0
        # rough order-of-magnitude check: ~O(1e-14 – 1e-4)
        @test 1e-15 < φ < 1e-4
    end

    @testset "zenith dependence" begin
        # At low energy (< ~1 TeV), vertical > horizontal (less absorption).
        # At high energy the crossover favors horizontal — test at 100 GeV where vertical wins.
        φ_vert  = Φ_μ(1e2, 1.0, 3.0)
        φ_horiz = Φ_μ(1e2, 0.1, 3.0)
        @test φ_vert > φ_horiz
    end

    @testset "altitude dependence" begin
        # Muons produced high up and propagate down — flux higher near sea level
        φ_high = Φ_μ(1e4, 1.0, 10.0)
        φ_low  = Φ_μ(1e4, 1.0, 0.0)
        @test φ_low > φ_high
    end

    @testset "energy dependence" begin
        # Flux falls steeply with energy
        @test Φ_μ(1e3, 1.0, 3.0) > Φ_μ(1e6, 1.0, 3.0)
    end

    @testset "out-of-range returns zero" begin
        @test Φ_μ(1e-3, 1.0, 3.0) == 0.0   # below MCEq grid minimum (~0.1 GeV)
        @test Φ_μ(1e4, 1.0, 100.0) == 0.0  # above 20 km
    end

    @testset "Real argument promotion" begin
        @test Φ_μ(10000, 1, 3) == Φ_μ(1e4, 1.0, 3.0)
    end

end
