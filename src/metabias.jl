module MetaBias
export NullLikelihoodPrior,MixModelLikelihoodPrior, NullPosterior, reset_τ, import_sampledata, density

using Distributions: Normal, cdf
using HDF5: h5open

abstract LikelihoodPrior

immutable NullDensityParam
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    τ::Float64
end

type NullLikelihoodPrior <: LikelihoodPrior
    ndp::NullDensityParam
end

type MixModelLikelihoodPrior <: LikelihoodPrior
    ndp::NullDensityParam
    z₁::Array{Float64,1} # significant z values
    σ₁::Array{Float64,1} # significant σ = sqrt(var))
    N₀::Int64 # number of non significant data, lenght(z) = length(z₁) + N₀
    Z::Float64 # z-score corresponding to test significance level
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

function reset_τ(ndp::NullDensityParam, τ)
    a = ndp.a * sqrt(ndp.τ) / sqrt(τ)
    b = ndp.b - 1.0/(2ndp.τ) + 1.0/(2τ)
    NullDensityParam(a,b,ndp.c,ndp.d,τ)
end

function reset_τ(lp::LikelihoodPrior,τ)
    lp.ndp = reset_τ(lp.ndp, τ)
    lp
end

function NullLikelihoodPrior{S<:Real,T<:Real}(z::Array{S,1},var::Array{T,1},τ::Float64=2.0)
    NullLikelihoodPrior(NullDensityParam(z,var,τ))
end

function MixModelLikelihoodPrior{S<:Real,T<:Real}(z::Array{S,1},var::Array{T,1},τ::Float64=2.0,Z::Float64=1.96)
    σ = sqrt(var)
    idx =  abs(z) .> (σ * Z)
    MixModelLikelihoodPrior(NullDensityParam(z,var,τ), z[idx],σ[idx],length(idx)-sum(idx),Z)
end

function Ni(μ, σ, Z)
    nd = Normal(μ, σ)
    1.0 - cdf(nd,σ*Z) + cdf(nd,-σ*Z)
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
    η, μ = x[1], x[2]
    @assert 0. <= η <= 1.
    modification = reduce(*,[η + (1-η)*Ni(μ,σᵢ,mm.Z) for σᵢ in mm.σ₁]) * η^mm.N₀
    println(η, "\t",μ,"\t", modification)
    density(mm.ndp,μ) * modification
end
function density{T<:Real}(mm::MixModelLikelihoodPrior, x::Array{T,2})
    N = size(x)[2]
    res = Array(eltype(x),N)
    for i in 1:N
        x[i] = density(mm, x[:,i])
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
