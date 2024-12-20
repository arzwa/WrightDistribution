using WrightDistribution
using Test

@testset "Neutral" begin
    Ne  = 100
    s   = 0.0
    h   = 0.5
    u10 = 0.01   # q -> p
    u01 = 0.005  # p -> q
    d   = Wright(Ne*s, 2Ne*u10, 2Ne*u01, h)
    @test mean(d) ≈ u10/(u01 + u10)
    Epq = expectedpq(d)
    Ep  = mean(d)
    Eq  = 1-Ep
    Vp  = Ep*Eq - Epq
    vp  = (2Ne)^2*u01*u10/((2Ne)^2*(u01 + u10)^2*(2Ne*(u01 + u10) + 1))
    @test vp ≈ Vp
    # mean(d) = E[p]
end

@testset "Selection" begin
    Ne  = 1000
    s   = 0.06
    u   = 0.01
    h   = 0.5
    # q -> p mutation should be approximately zero (u/s result holds for
    # one-way mutation)
    # Note that for haploids we should double Ns and use h=0.5
    d   = Wright(-Ne*(2s), Ne*u/1000, Ne*u, h)
    q   = 1 - mean(d)
    ϵ   = 0.01
    @test u/s < q < u/s + ϵ  # drift causes E[q] > deterministic prediction
end



