module MetaBias
export MixModelLikelihoodPrior, MixModel, NullPosterior, reset_τ, import_sampledata, density, nulldensity, mass, nullmass, mean, var, std, rand, pdf, bayesfactor

using Distributions: Normal, Bernoulli, cdf, ContinuousUnivariateDistribution
using HDF5: h5open
using Cubature: hquadrature, hcubature

# extending
import Distributions: rand, pdf, mean, var, std

immutable MixModel <: ContinuousUnivariateDistribution
    bd::Bernoulli
    nd::Normal
    σ::Float64
    Z::Float64

    function MixModel(η::Real, μ::Real, σ2::Real, Z::Real=1.96)
        σ = sqrt(σ2)
        new(Bernoulli(η), Normal(μ,σ), σ, Z)
    end
end
heaviside(x) = 0.5*(sign(x)+1)
pdf(d::MixModel,x::Real) = pdf(d.nd,x) * (d.bd.p + (1-d.bd.p)*norm_const(d.nd,d.σ,d.Z)*heaviside(abs(x) - d.Z*d.σ))
function rand_dishonest(d::MixModel)
    x = rand(d.nd)
    d.Z*d.σ < abs(x) ? x : rand_dishonest(d)
end
function rand(d::MixModel)
    honest = rand(d.bd)
    honest==1 ? rand(d.nd) : rand_dishonest(d)
end

immutable NullDensityParam
    # paramaeters for likelihood x prior
    # L(x) prior(x) =  exp[-(a x^2 + b x + c)]
    a::Float64
    b::Float64
    c::Float64
    σ_prior::Float64
end
function NullDensityParam{S<:Real,T<:Real}(y::Array{S,1},var::Array{T,1},var_prior::Real)
    σ = sqrt(var)
    N = length(y)
    @assert N==length(σ)
    @assert N>0
    #var = σ.^2
    σ_prior = sqrt(var_prior)
    a = 0.5 * (sum(1./var) + 1/σ_prior^2)
    b = - sum(y./var)
    c = 0.5*sum(y.^2./var) + sum(log(σ)) + log(σ_prior) + 0.5*(N+1)*log(2π)
    NullDensityParam(a,b,c,σ_prior)
end
var(ndp::NullDensityParam) = 1 / (2*ndp.a)
mean(ndp::NullDensityParam) = - var(ndp) * ndp.b
std(ndp::NullDensityParam) = sqrt(var(ndp))

type MixModelLikelihoodPrior
    ndp::NullDensityParam
    z::Array{Float64,1} # all data
    z₁::Array{Float64,1} # significant z values
    σ::Array{Float64,1} # all
    σ₁::Array{Float64,1} # significant σ = sqrt(var))
    N₀::Int64 # number of non significant data, lenght(z) = length(z₁) + N₀
    Z::Float64 # z-score corresponding to test significance level
end

function MixModelLikelihoodPrior{S<:Real,T<:Real}(z::Array{S,1},var::Array{T,1},τ::Float64=2.0,Z::Float64=1.96)
    σ = sqrt(var)
    idx =  abs(z) .> (σ * Z)
    MixModelLikelihoodPrior(NullDensityParam(z,var,τ), z,z[idx],σ,σ[idx],length(idx)-sum(idx),Z)
end

function reset_τ(ndp::NullDensityParam, var_prior::Real)
    σ_prior = sqrt(var_prior)
    a = ndp.a - 1.0/(2ndp.σ_prior^2) + 1.0/(2σ_prior^2)
    c = ndp.c - log(ndp.σ_prior) + log(σ_prior)
    NullDensityParam(a,ndp.b,c,σ_prior)
end
function reset_τ(mm::MixModelLikelihoodPrior, var_prior::Real)
    mm.ndp = reset_τ(mm.ndp, var_prior)
    mm
end

function norm_const(nd::Normal, σ::Real, Z::Real)
    @assert nd.σ == σ
    1/(1.0 - cdf(nd,σ*Z) + cdf(nd,-σ*Z))
end
function norm_const(μ::Real, σ::Real, Z::Real)
    nd = Normal(μ, σ)
    norm_const(nd, σ, Z)
end

logdensity(ndp::NullDensityParam,x::Real) = -(ndp.a*x^2 + ndp.b*x + ndp.c)
function logdensity{T<:Real}(mm::MixModelLikelihoodPrior, x::Array{T,1})
    @assert length(x) == 2
    η, μ, σ₁, N₀, Z = x[1], x[2], mm.σ₁, mm.N₀, mm.Z
    @assert 0. <= η <= 1.
    modification = reduce(+,[log(η + (1-η)*norm_const(μ,σᵢ,Z)) for σᵢ in σ₁]) + N₀*log(η)
    logdensity(mm.ndp, μ) + modification
end
density{T<:Real}(mm::MixModelLikelihoodPrior, x::Array{T,1}) = exp(logdensity(mm,x))
density(ndp::NullDensityParam,x::Real) = exp(logdensity(ndp,x))



nulldensity(mm::MixModelLikelihoodPrior, x::Real) = exp(logdensity(mm.ndp,x))

function nulldensity{T<:Real}(mm::MixModelLikelihoodPrior, x::Array{T,1})
    N = length(x)
    res = Array(eltype(x),N)
    for i in 1:N
        res[i] = nulldensity(mm, x[i])
    end
    res
end
function density{T<:Real}(mm::MixModelLikelihoodPrior, x::Array{T,2})
    N = size(x)[2]
    res = Array(eltype(x),N)
    for i in 1:N
        res[i] = density(mm, x[:,i])
    end
    res
end

function logdensity{T<:Real}(mm::MixModelLikelihoodPrior, x::Array{T,2})
    N = size(x)[2]
    res = Array(eltype(x),N)
    for i in 1:N
        res[i] = logdensity(mm, x[:,i])
    end
    res
end

function nullmass(mm::MixModelLikelihoodPrior)
    f(x) = nulldensity(mm, x)
    m, s = mean(mm.ndp), std(mm.ndp)
    (val,err) = hquadrature(f, m - 10*s, m + 10*s)
    val
end

function mass(mm::MixModelLikelihoodPrior)
    f(x) = density(mm, x)
    m, s = mean(mm.ndp), std(mm.ndp)
    (val,err) = hcubature(f, [0,m - 10*s], [1,m + 10*s])
    val
end

function mass_fixed_η(mm::MixModelLikelihoodPrior, η::Real)
    f(x) = density(mm, [η,x])
    m, s = mean(mm.ndp), std(mm.ndp)
    (val,err) = hquadrature(f, m - 10*s, m + 10*s)
    val
end

function mass_fixed_μ(mm::MixModelLikelihoodPrior, η::Real)
    f(x) = density(mm, [η,x])
    m, s = mean(mm.ndp), std(mm.ndp)
    (val,err) = hquadrature(f, m - 10*s, m + 10*s)
    val
end

function pdf_given_η(mm::MixModelLikelihoodPrior, η::Real)
    norm = mass_fixed_η(mm, η)
    f(μ) = density(mm, [η, μ]) / norm
end

function log10_bayesfactor(mm::MixModelLikelihoodPrior)
    log10(mass(mm)) - log10(nullmass(mm))
end

function bayesfactor(mm::MixModelLikelihoodPrior)
    10^log10_bayesfactor(mm)
end

# functions for test purposes
function NullPosterior{S<:Real,T<:Real}(z::Array{S,1},var::Array{T,1},τ::Real=2.0)
    postvar = 1 /(sum(1./var) + 1./τ)
    postmean = postvar * sum(z./var)
    Normal(postmean, sqrt(postvar))
end

function import_sampledata(h5file, name)
    out = h5open(h5file, "r") do file
        zi = read(file, name*"/zi")[:]
        vi = read(file, name*"/vi")[:]
        return zi, vi
    end
    return out
end
end
