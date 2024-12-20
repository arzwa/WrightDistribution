using WrightDistribution
using Plots

# diploid
N = 200
s = 0.02
u = s/100
m = s/10
p̄ = 0.9
h = 0.3
d = Wright(2N*s, 2N*(m*p̄ + u), 2N*(m*(1-p̄) + u), h)

function simulate(N, s, u, m, p̄, h, ngen, p)
    map(1:ngen) do _
        p_ = p*(1-m) + m*p̄
        p_ = p_*(1-u) + u*(1-p_)
        q_ = 1 - p_
        w1 = (1 + s*h)*p_ + (1 + s)*q_
        w̄  = p_^2 + 2p_*q_*(1 + s*h) + (1 + s)*q_^2
        q′ = (w1/w̄)*q_
        p′ = 1-q′
        p  = rand(Binomial(2N, p′)) / 2N
    end
end

ps = simulate(N, s, u, m, p̄, h, 10_000_000, rand())[1:10:end]
stephist(ps, bins=0:0.01:1, normalize=true)
plot!(0:0.001:1, x->pdf(d, x))


# haploid
N = 500
s = 0.02
u = s/100
m = s/6
p̄ = 0.75
d = Wright(2N*s, N*(m*p̄ + u), N*(m*(1-p̄) + u), 0.5)

function hap_simulate(N, s, u, m, p̄, ngen, p)
    map(1:ngen) do _
        p_ = p *(1-m) + m*p̄
        p_ = p_*(1-u) + u*(1-p_)
        q_ = 1 - p_
        w1 = (1 + s)
        w̄  = p_ + (1 + s)*q_
        q′ = (w1/w̄)*q_
        p′ = 1 - q′
        p  = rand(Binomial(N, p′)) / N
    end
end

ps = hap_simulate(N, s, u, m, p̄, 1_000_000, rand())
stephist(ps, bins=0:0.01:1, normalize=true)
plot!(0:0.001:1, x->pdf(d, x))


