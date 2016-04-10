module MetaBias
export NullLikelihoodPrior,MixModelLikelihoodPrior, MixModel, NullPosterior, reset_τ, import_sampledata, density, length, rand, pdf

using Distributions: Normal, Bernoulli, cdf, ContinuousUnivariateDistribution
using HDF5: h5open

# extending
import Base: length
import Distributions: rand, pdf

abstract LikelihoodPrior

immutable MixModel <: ContinuousUnivariateDistribution
    bd::Bernoulli
    nd::Normal
    σ::Float64
    Z::Float64
    
    function MixModel(η::Real, μ::Real, σ2::Real, Z::Real)
        σ = sqrt(σ2)
        new(Bernoulli(η), Normal(μ,σ), σ, Z)
    end
end
MixModel(η::Real, μ::Real, σ2::Real) = MixModel(η, μ, σ2, 1.96)
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
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    τ::Float64
end

function NullDensityParam{S<:Real,T<:Real}(z::Array{S,1},var::Array{T,1},τ::Real)
    N = length(z)
    @assert N==length(var)
    @assert N>0
    a = (2pi)^(-(N+1)/2.0) / sqrt(reduce(*,var)) / sqrt(τ)
    b = sum(1./var)/2 + 1.0/(2τ)
    c = sum(z./var)
    d = sum(z.^2./var) / 2.0
    NullDensityParam(a,b,c,d,τ)
end

type NullLikelihoodPrior <: LikelihoodPrior
    ndp::NullDensityParam
end

function NullLikelihoodPrior{S<:Real,T<:Real}(z::Array{S,1},var::Array{T,1},τ::Float64=2.0)
    NullLikelihoodPrior(NullDensityParam(z,var,τ))
end

type MixModelLikelihoodPrior <: LikelihoodPrior
    ndp::NullDensityParam
    z₁::Array{Float64,1} # significant z values
    σ₁::Array{Float64,1} # significant σ = sqrt(var))
    N₀::Int64 # number of non significant data, lenght(z) = length(z₁) + N₀
    Z::Float64 # z-score corresponding to test significance level
end

function MixModelLikelihoodPrior{S<:Real,T<:Real}(z::Array{S,1},var::Array{T,1},τ::Float64=2.0,Z::Float64=1.96)
    σ = sqrt(var)
    idx =  abs(z) .> (σ * Z)
    MixModelLikelihoodPrior(NullDensityParam(z,var,τ), z[idx],σ[idx],length(idx)-sum(idx),Z)
end

function reset_τ(ndp::NullDensityParam, τ)
    a = ndp.a * sqrt(ndp.τ) / sqrt(τ)
    b = ndp.b - 1.0/(2ndp.τ) + 1.0/(2τ)
    NullDensityParam(a,b,ndp.c,ndp.d,τ)
end
function reset_τ(lp::LikelihoodPrior,τ)
    lp.ndp = reset_τ(lp.ndp, τ)
    lp
end

function norm_const(nd::Normal, σ::Real, Z::Real)
    @assert nd.σ == σ
    1/(1.0 - cdf(nd,σ*Z) + cdf(nd,-σ*Z))
end
function norm_const(μ::Real, σ::Real, Z::Real)
    nd = Normal(μ, σ)
    norm_const(nd, σ, Z)
end

density(ndp::NullDensityParam,x::Real) = ndp.a * exp(-(ndp.b*x^2 - ndp.c*x + ndp.d))
density(null::NullLikelihoodPrior,μ::Real) = density(null.ndp,μ)
function density{T<:Real}(null::NullLikelihoodPrior, x::Array{T,1})
    N = length(x)
    res = Array(eltype(x),N)
    for i in 1:N
        res[i] = density(null, x[i])
    end
    res
end

function density{T<:Real}(mm::MixModelLikelihoodPrior, x::Array{T,1})
    @assert length(x) == 2
    η, μ, σ₁, N₀, Z = x[1], x[2], mm.σ₁, mm.N₀, mm.Z
    @assert 0. <= η <= 1.
    modification = reduce(*,[η + (1-η)*norm_const(μ,σᵢ,Z) for σᵢ in σ₁]) * η^N₀
    density(mm.ndp,μ) * modification
end
function density{T<:Real}(mm::MixModelLikelihoodPrior, x::Array{T,2})
    N = size(x)[2]
    res = Array(eltype(x),N)
    for i in 1:N
        res[i] = density(mm, x[:,i])
    end
    res
end

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
