module WrightDistribution
# REFACTOR -> scaled parameters only

using QuadGK, Parameters, Reexport, LogExpFunctions
@reexport using Distributions
export Wright, sfs, expectedpq, Epq

"""
    Wright(Nₑs, Nₑα, Nₑβ, h)

Wright's distribution with unnormalized pdf:
```
f(p) ∝ p^(2Ne*α - 1) * q^(2Nₑ*β - 1) exp(-Nₑs p (2h + (1-2h)*(2-p)))
```
Here `Nₑs` is the scaled selection coefficient for the allele measured by the
frequency `q = 1-p` (e.g. `Nₑs = -10.0` is a moderately strong selective
disadvantage of the allele measured by the frequency `q = 1-p`).
That is, we assume that at HWE we have the following genotype frequencies and
associated fitnesses:
```
p^2 : 2pq    : q^2
1   : 1 + sh : 1 + s
```
**Note** that `mean(d::Wright)` yields 𝔼[p].

**Note**: `s` and `h` should be effective selection and dominance coefficients.
For nstance, to model **haploid selection** with relative fitness of `1+s` for
the allele with frequency `q`, population size `N`, migration rate `m` and
mainland allele frequency `p̄` use
```
Wright(2N*s, N*(m*p̄ + u), N*(m*(1-p̄) + u), 0.5)
```

To model **diploid selection**, note that one should specify `Nₑ = 2N`. For
instance with dominance `h`:
```
Wright(2N*s, 2N*(m*p̄ + u), 2N*(m*(1-p̄) + u), h)
```
"""
struct Wright{T} <: ContinuousUnivariateDistribution
    Ns :: T
    Nα :: T 
    Nβ :: T
    h  :: T
    Z  :: T
end

function Wright(Ns, Nα, Nβ, h; kwargs...)
    @assert Nα >= zero(Nα) && Nβ >= zero(Nβ)
    Wright(promote(Ns, Nα, Nβ, h, Zfun(Ns, 2Nα, 2Nβ, h; kwargs...))...)
end

# translates from previous redundant parameterization
#sa = s1 + s01
#sb = s11 - 2s01
#s = 2s1 + s11 = 2sa + sb
#h = (s1 + s01)/(2s1 + s11)  = sa/(2sa + sb)
#sb = (1-2h)*s
#sa = s*h
#A  = 2Ne*α
#B  = 2Ne*β
#Ns = Ne*(2sa + sb)
#h  = sa/(2sa + sb) 
#Wright(Ns, A, B, h; kwargs...)

# Normalizing constant of Wright's distribution, using IBP
# could just use Z0fun/Z1fun/Zxfun below (but implemented this one first...)
function Zfun(Ns, A, B, h; kwargs...)
	# c = 0.5^(A + B - 1) * exp(-N*(sa + 3sb/4)) * (1/A + 1/B)
    c = 0.5^(A + B - 1) * exp(-Ns*(h + 3*(1-2h)/4)) * (1/A + 1/B)
	d, _ = quadgk(p->((    p^A)/A)*fpfun(p, B, Ns, h), 0.0, 0.5; kwargs...)
	e, _ = quadgk(p->(((1-p)^B)/B)*gpfun(p, A, Ns, h), 0.5, 1.0; kwargs...)
	c - d + e
end

Cfun(p, Ns, h) = exp(-p*Ns*(2h + (1-2h)*(2-p)))
logCfun(p, Ns, h) = -Ns*p*(2h + (1-2h)*(2-p))

function fpfun(p, B, Ns, h)
    #-(1-p)^(B-1)*(N * (-p*sb + 2sa + sb*(2-p)) + (B-1)/(1-p)) * Cfun(p, N, sa, sb) 
    -(1-p)^(B-1)*(2Ns * (1-h-p*(1-2h)) + (B-1)/(1-p)) * Cfun(p, Ns, h) 
end

function gpfun(p, A, Ns, h)
    #p^(A-1)*(N*(p*sb - 2sa + sb*(p-2)) + (A-1)/p) * Cfun(p, N, sa, sb) 
    p^(A-1)*(-2Ns * (1-h-p*(1-2h)) + (A-1)/p) * Cfun(p, Ns, h) 
end

function gspfun(p, A, Ns, h)
    #p^A*(N*(p*sb - 2sa + sb*(p-2)) + A/p) * Cfun(p, N, sa, sb)
    p^A*(-2Ns*(1-h-p*(1-2h)) + A/p) * Cfun(p, Ns, h)
end

function Yfun(Ns, A, B, h; kwargs...)
    C(p) = Cfun(p, Ns, h)
	L, _ = quadgk(p->p^A * (1-p)^(B-1) * C(p), 0., 0.5; kwargs...)
	R1 = (1/B)*(1/2)^(B+A)*C(0.5)
	R2, _ = quadgk(p->(((1-p)^B)/B)*gspfun(p, A, Ns, h), 0.5, 1; kwargs...)
	return L + R1 + R2
end

function Distributions.mean(d::Wright; kwargs...) 
	@unpack Ns, Nα, Nβ, h, Z = d
    Yfun(Ns, 2Nα, 2Nβ, h; kwargs...) / Z
end

function Distributions.var(d::Wright; kwargs...) 
    Ep = mean(d; kwargs...)
    Epq = expectedpq(d; kwargs...)
    Ep*(1-Ep) - Epq   
    # E[pq] = E[p] - E[p^2] <=> E[p^2] = E[p] - E[pq] 
    # => V[p] = E[p^2] - E[p]^2 
    #         = E[p] - E[pq] - E[p]^2 
    #         = E[p]E[q] - E[pq]  
end

Epq(d::Wright; kwargs...) = expectedpq(d; kwargs...)

function expectedpq(d::Wright; kwargs...)
	@unpack Ns, Nα, Nβ, h, Z = d
    N, _ = quadgk(p -> p^2Nα * (1-p)^2Nβ * Cfun(p, Ns, h), 0., 1.)
    return N/Z
end

function Distributions.pdf(d::Wright, p::Real) 
    @unpack Ns, Nα, Nβ, h, Z = d
    p^(2Nα-1) * (1-p)^(2Nβ-1) * Cfun(p, Ns, h) / Z
end

function Distributions.logpdf(d::Wright, p) 
    @unpack Ns, Nα, Nβ, h, Z = d
    (2Nα-1)*log(p) + (2Nβ-1)*log(1-p) - Ns*p*(2h + (1-2h)*(2-p)) - log(Z)
end

function unnormpdf(d::Wright, p)
    @unpack Ns, Nα, Nβ, h, Z = d
    p^(2Nα-1) * (1-p)^(2Nβ-1) * Cfun(p, Ns, h)
end

function unnormlogpdf(d::Wright, p)
    @unpack Ns, Nα, Nβ, h, Z = d
    (2Nα-1)*log(p) + (2Nβ-1)*log(1-p) - Ns*p*(2h + (1-2h)*(2-p))
end

function sfs(d::Wright, bins; kwargs...)
    ys = map(i->Zxfun(d, bins[i], bins[i+1]), 1:length(bins)-1)
    xs = bins[1:end-1] .+ step(bins)/2
    xs, ys
end

function Zxfun(d, x, y; kwargs...)
    x ≈ 0 && y ≈ 1 && return d.Z
    x ≈ 0 && return Z0fun(d, y; kwargs...)
    y ≈ 1 && return Z1fun(d, x; kwargs...)
    I, _ = quadgk(p->pdf(d, p), x, y; kwargs...)
    return I
end

# 0 to x integral
function Z0fun(d, x; kwargs...)
	@unpack Ns, Nα, Nβ, h = d
    a = (1/2Nα)*(x)^2Nα*(1-x)^(2Nβ-1)*Cfun(x, Ns, h)
	b, _ = quadgk(p->((p^2Nα)/2Nα)*fpfun(p, 2Nβ, Ns, h), 0, x; kwargs...)
    return a - b
end

# x to 1 integral
function Z1fun(d, x; kwargs...)
	@unpack Ns, Nα, Nβ, h = d
    #la = B * log(1-x) + (A-1)*log(x) + logCfun(x, Ne, sa, sb)
    #b, _ = quadgk(p->((1-p)^B)*gpfun(p, A, Ne, sa, sb), x, 1; kwargs...)
    #logsumexp(la, log(b))
    a = (1/2Nβ)*(1-x)^2Nβ*x^(2Nα-1)*Cfun(x, Ns, h)
    b, _ = quadgk(p->(((1-p)^2Nβ)/2Nβ)*gpfun(p, 2Nα, Ns, h), x, 1; kwargs...)
    return a + b
end


end # module WrightDistribution
