
Ns = -2.0
Nm = 1.0
Nu = 0.01
d  = Wright(Ns, Nu, Nu, 1/2)
mean(d)

using WrightDistribution, FastGaussQuadrature, LinearAlgebra

function gjZ(Ns, Nα, Nβ, h, n=5)
    a = (1/2)^(Nα + Nβ - 1) 
    x, w = gaussjacobi(n, Nβ-1, Nα-1)
    f(x) = WrightDistribution.Cfun((1+x)/2, Ns, h)
    return a*dot(w, f.(x))
end

@btime Wright(Ns, Nu, Nu, 1/2)
@btime gjZ(d, 20)

Nss = [0.0, 0.01, 0.1, 1.0, 10., 100., 1000., 10000.]
Nαs = [0.0, 1e-7, 1e-5, 1e-3, 0.01, 0.1, 1.0, 10.0, 100., 1000., 10000.]
hs  = [0.0, 0.25, 0.5, 0.75, 1.0]
X = map(Iterators.product(Nss, Nαs, Nαs, hs)) do (Ns, Na, Nb, h)
    Z = try 
        WrightDistribution.Zfun(Ns, Na, Nb, h)
    catch
        Inf
    end
    Zs = map([3,5,10,20,100]) do n
        try 
            gjZ(Ns, Na, Nb, h, n)
        catch
            Inf
        end
    end
    [Z; Zs...]
end

xs = X[:,5,3,1]

# Should determine in detail what is the best way to proceed for a particular
# combination of Ns Nm (Nu) and h. It seems to depend mostly on Ns. max(3,
# Ns/10) or something like that might work?
function nnodes(Ns, Nα, Nβ, h)
    min(max(3, ceil(Int64, Ns/5)), 1000)
end

Nss = [0.0, 0.01, 0.1, 1.0, 10., 100., 1000.]
Nαs = [1e-7, 1e-5, 1e-3, 0.01, 0.1, 1.0, 10.0, 100., 1000., 10000.]
hs  = [0.0, 0.25, 0.5, 0.75, 1.0]
X = map(Iterators.product(Nss, Nαs, Nαs, hs)) do (Ns, Na, Nb, h)
    Z1 = @timed WrightDistribution.Zfun(Ns, Na, Nb, h)
    n = nnodes(Ns, Na, Nb, h)
    Z2 = @timed gjZ(Ns, Na, Nb, h, n)
    n, Z1.time - Z2.time, abs(Z1.value - Z2.value)/Z1.value
end

