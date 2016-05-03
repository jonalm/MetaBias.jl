module MetaBias
export MixModelLikelihoodPrior, MixModel, NullPosterior, reset_σ_prior, import_sampledata, density, nulldensity, mass, nullmass, mean, var, std, rand, pdf, bayesfactor, log10_bayesfactor, marginal_μ_density, marginal_η_density, plotpyramide, plotjoint, plotbayesfactor

using Distributions: Normal, Bernoulli, cdf, ContinuousUnivariateDistribution
using HDF5: h5open
using Cubature: hquadrature, hcubature

# for plot routines
using PyPlot
using PyCall
@pyimport matplotlib.patches as patch
PyPlot.plt[:style][:use]("ggplot")

import Distributions: rand, pdf, mean, var, std # extending these


# TODO:
# limits of the posterior effect (μ) for mixturemodel is currently set to
# (mean-10std, mean+10std), where mean and std correspond to the null distribution,
# these limits should be calculated or controlled explicitly


# core types and functions

immutable MixModel <: ContinuousUnivariateDistribution
    bd::Bernoulli
    nd::Normal
    σ::Float64
    Z::Float64

    function MixModel(η::Real, μ::Real, σ::Real, Z::Real=1.96)
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
function NullDensityParam{S<:Real,T<:Real}(effect::Array{S,1},σ::Array{T,1},σ_prior::Real)
    N = length(effect)
    @assert N==length(σ)
    @assert N>0
    var = σ.^2
    var_prior = σ_prior^2
    a = 0.5 * (sum(1./var) + 1/var_prior)
    b = - sum(effect./var)
    c = 0.5*sum(effect.^2./var) + sum(log(σ)) + log(σ_prior) + 0.5*(N+1)*log(2π)
    NullDensityParam(a,b,c,σ_prior)
end
var(ndp::NullDensityParam) = 1 / (2*ndp.a)
mean(ndp::NullDensityParam) = - var(ndp) * ndp.b
std(ndp::NullDensityParam) = sqrt(var(ndp))

type MixModelLikelihoodPrior
    ndp::NullDensityParam
    # sorted data, such that significant σ can be accessed by sliceing the array
    sorted_effect::Array{Float64,1}
    sorted_σ::Array{Float64,1}
    N_non_significant::Int64 # number of non significant data
    Z::Float64 # z-score corresponding to test significance level
    nullposterior::Normal
    name::ASCIIString
end

function MixModelLikelihoodPrior{S<:Real,T<:Real}(effect::Array{S,1},
                                                  σ::Array{T,1},
                                                  σ_prior::Float64=2.0,
                                                  Z::Float64=1.96,
                                                  name::ASCIIString="unset")
    ndp = NullDensityParam(effect,σ,σ_prior)

    significance = abs(effect) - σ * Z
    s = sortperm(significance)
    sorted_significance = significance[s]
    ff = findfirst(x->x>0.0, sorted_significance)
    N_non_significant = ff==0 ? length(sortet_significane) : ff-1

    sorted_effect = effect[s]
    sorted_σ = σ[s]

    MixModelLikelihoodPrior(ndp,
                            sorted_effect,
                            sorted_σ,
                            N_non_significant,
                            Z,
                            Normal(mean(ndp), std(ndp)),
                            name)
end

function reset_σ_prior(ndp::NullDensityParam, σ_prior::Real)
    a = ndp.a - 1.0/(2ndp.σ_prior^2) + 1.0/(2σ_prior^2)
    c = ndp.c - log(ndp.σ_prior) + log(σ_prior)
    NullDensityParam(a,ndp.b,c,σ_prior)
end
function reset_σ_prior(mm::MixModelLikelihoodPrior, σ_prior::Real)
    mm.ndp = reset_σ_prior(mm.ndp, σ_prior)
    mm.nullposterior = Normal(mean(mm.ndp), std(mm.ndp))
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

# Key function! Definition of density:
logdensity(ndp::NullDensityParam,x::Real) = -(ndp.a*x^2 + ndp.b*x + ndp.c)
function logdensity{T<:Real}(mm::MixModelLikelihoodPrior, x::Array{T,1})
    @assert length(x) == 2
    η, μ = x[1], x[2]
    Nns, Z = mm.N_non_significant, mm.Z
    sσ = mm.sorted_σ[Nns+1:end]
    @assert 0. <= η <= 1.
    modification = reduce(+,[log(η + (1-η)*norm_const(μ,σᵢ,Z)) for σᵢ in sσ]) + Nns*log(η)
    logdensity(mm.ndp, μ) + modification
end
function logdensity{T<:Real}(mm::MixModelLikelihoodPrior, x::Array{T,2})
    N = size(x)[2]
    res = Array(eltype(x),N)
    for i in 1:N
        res[i] = logdensity(mm, x[:,i])
    end
    res
end

density{T<:Real}(mm::MixModelLikelihoodPrior, x::Array{T,1}) = exp(logdensity(mm,x))
density(ndp::NullDensityParam,x::Real) = exp(logdensity(ndp,x))
function density{T<:Real}(mm::MixModelLikelihoodPrior, x::Array{T,2})
    N = size(x)[2]
    res = Array(eltype(x),N)
    for i in 1:N
        res[i] = density(mm, x[:,i])
    end
    res
end

nulldensity(mm::MixModelLikelihoodPrior, x::Real) = exp(logdensity(mm.ndp,x))
function nulldensity{T<:Real}(mm::MixModelLikelihoodPrior, x::Array{T,1})
    N = length(x)
    res = Array(eltype(x),N)
    for i in 1:N
        res[i] = nulldensity(mm, x[i])
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

function marginal_μ_density(mm::MixModelLikelihoodPrior, μ::Real)
    # accuracy not important, only used for plots
    f(η) = density(mm, [η, μ])
    (val,err) = hquadrature(f, 0, 1, reltol=1e-3, abstol=1e-3)
    val
end
function marginal_μ_density{T<:Real}(mm::MixModelLikelihoodPrior,x::Array{T,1})
    N = length(x)
    res = Array(eltype(x),N)
    for i in 1:N
        res[i] = marginal_μ_density(mm, x[i])
    end
    res
end

function marginal_η_density(mm::MixModelLikelihoodPrior, η::Real)
    # accuracy not important, only used for plots
    f(μ) = density(mm, [η,μ])
    m, s = mean(mm.ndp), std(mm.ndp)
    (val,err) = hquadrature(f, m - 10*s, m + 10*s,reltol=1e-3, abstol=1e-3)
    val
end
function marginal_η_density{T<:Real}(mm::MixModelLikelihoodPrior,x::Array{T,1})
    N = length(x)
    res = Array(eltype(x),N)
    for i in 1:N
        res[i] = marginal_η_density(mm, x[i])
    end
    res
end

function log10_bayesfactor(mm::MixModelLikelihoodPrior)
    log10(mass(mm)) - log10(nullmass(mm))
end
function bayesfactor(mm::MixModelLikelihoodPrior)
    10^log10_bayesfactor(mm)
end

function bayesfactor{T<:Real}(mm::MixModelLikelihoodPrior,σ_priors::Array{T,1})
    original_prior  = mm.ndp.σ_prior
    N = length(σ_priors)
    res = Array(eltype(σ_priors), N)

    for i in 1:N
        reset_σ_prior(mm, σ_priors[i])
        res[i] = bayesfactor(mm)
    end

    reset_σ_prior(mm, original_prior)
    res
end

# functions for test purposes
function NullPosterior{S<:Real,T<:Real}(z::Array{S,1},σ::Array{T,1},σ_prior::Real=2.0)
    var = σ .^ 2
    var_prior = σ_prior ^ 2
    postvar = 1 /(sum(1./var) + 1./var_prior)
    postmean = postvar * sum(z./var)
    Normal(postmean, sqrt(postvar))
end

# misc
function import_sampledata(h5file, name)
    out = h5open(h5file, "r") do file
        z = read(file, name*"/Z_transformed_effect")[:]
        v = read(file, name*"/variance")[:]
        return z, v
    end
    return out
end

# plotting
const BLUE="#348ABD"
const RED="#A60628"
const ORANGE="#E24A33"

function plotpyramide(mm::MixModelLikelihoodPrior)
    sef = mm.sorted_effect
    sσ = mm.sorted_σ
    Nsf = mm.N_non_significant

    y1 = maximum(sσ)
    margin_y = 0.08*y1
    ylow, yhigh = y1 + margin_y, -margin_y
    ylowp = ylow + margin_y
    x1,x2 = minimum(sef), maximum(sef)
    margin_x = 0.08*(x2-x1)
    xlow, xhigh = x1 - margin_x, x2 + margin_x
    axmargin = 0.01
    separator = 0.7
    fig = figure()
    ax1 = fig[:add_axes]([0+axmargin,0+axmargin,
                          1-2*axmargin,separator-2*axmargin])
    ax2 = fig[:add_axes]([0+axmargin,separator+axmargin,
                          1-2*axmargin,1 - separator - 2*axmargin])

    ax1[:invert_yaxis]()
    c = patch.Polygon([0 0; -ylowp*mm.Z ylowp; ylowp*mm.Z ylowp],
                      alpha=0.5,fc="white",ec="k", ls="dashed")
    ax1[:add_artist](c)
    ax1[:plot](sef[Nsf+1:end],sσ[Nsf+1:end],"o", color=RED, label="Significant")
    ax1[:plot](sef[1:Nsf],sσ[1:Nsf],"o", color=BLUE, label="Non-significant")
    ax1[:set_ylim]([ylow,yhigh])
    ax1[:set_xlim]([xlow,xhigh])
    ax1[:legend](numpoints=1)
    ax1[:set_ylabel]("measured error (std)")
    ax1[:set_xlabel]("measured effect")

    a = mean(mm.ndp)
    b = std(mm.ndp)
    xmin, xmax = a - 10*b, a + 10*b
    xx = collect(linspace(xmin, xmax,100))
    yy = pdf(mm.nullposterior,xx)
    zz = marginal_μ_density(mm, xx) / mass(mm)

    ax2[:fill_between](xx,0,yy, color=ORANGE,
                       alpha=0.7, label="Null posterior")
    ax2[:fill_between](xx,0,zz, color="k",
                       alpha=0.7, label="MixModel marginal posterior")
    ax2[:set_xlim]([xlow,xhigh])
    ax2[:legend](numpoints=1)
    ax2[:set_xticklabels]([])
    ax2[:set_ylabel](L"pdf posterior effect, $p(\mu|$data$)$")
    fig, (ax1,ax2)
end

function plotjoint(mm::MixModelLikelihoodPrior, etamin::Real=0.5, μstdwidth::Real=6.0)
    @assert 0.0 <= etamin <= 1.0
    @assert 0.0 < μstdwidth
    a, b = mean(mm.ndp), std(mm.ndp)
    murange = [a-μstdwidth*b, a+μstdwidth*b]
    etarange = [etamin,1.0]

    mass_ = mass(mm)

    NY, NX = 100, 100
    eta = linspace(etarange[1],etarange[2],NY) |> collect
    mu = linspace(murange[1], murange[2], NX) |> collect
    ETA = Float64[e for e in eta, m in mu]
    MU = Float64[m for e in eta, m in mu]
    coord = [ETA[:] MU[:]]'
    jointlogdensity = reshape(logdensity(mm, coord), NY, NX)
    jointpdf = exp(jointlogdensity) / mass_
    marginal_mu = marginal_μ_density(mm, mu) / mass_
    marginal_eta = marginal_η_density(mm, eta) / mass_

    axmargin = 0.01
    separator = 0.8
    fig = figure(figsize=(8,8))
    ax1 = fig[:add_axes]([0+axmargin,0+axmargin,
                          separator-2*axmargin,separator-2*axmargin])
    ax2 = fig[:add_axes]([0+axmargin,separator+axmargin,
                          separator-2*axmargin , 1 - separator - 2*axmargin])
    ax3 = fig[:add_axes]([separator+axmargin,axmargin,
                          1-separator-2*axmargin, separator - 2*axmargin])
    ax2[:set_xticklabels]([])
    ax3[:set_yticklabels]([])
    ax1[:contour](MU,ETA, jointlogdensity,10,colors=RED, alpha=0.5)
    ax1[:contour](MU,ETA, jointpdf,10,colors="k")
    ax2[:fill_between](mu,0,marginal_mu,color="k")
    ax2[:set_xlim](murange)
    ax3[:fill_betweenx](eta,0,marginal_eta,color="k")
    ax3[:set_ylim](etarange)

    ax1[:legend](numpoints=1)
    fig[:text](2*axmargin,5*axmargin,
               "Joint Posterior Distributions",
               verticalalignment="bottom", horizontalalignment="left")
    fig[:text](2*axmargin,2*axmargin,
               L"Black solid: $p(\mu, \eta\, |$ data $) \quad $ Stapled red : $\log p(\mu, \eta\, |$ data $)$",
               verticalalignment="bottom", horizontalalignment="left")
    ax1[:set_xlabel](L"$\mu$")
    ax1[:set_ylabel](L"$\eta$")
    ax2[:set_ylabel](L"$p(\mu\,|$ data$)$")
    ax3[:set_xlabel](L"$p(\eta\,|$ data$)$")
    fig, (ax1, ax2, ax3)
end

function add_bayes2ax{T<:Real}(mm::MixModelLikelihoodPrior, ax::PyCall.PyObject, priors::Array{T,1}) 
    res = bayesfactor(mm, priors)
    ax[:loglog](priors,res)
    ax   
end

function plotbayesfactor(mm::MixModelLikelihoodPrior)
    plotbayesfactor([mm])
end

function plotbayesfactor(mmm::Array{MixModelLikelihoodPrior, 1})
    priors = collect(logspace(-2,2))
    fig = figure(figsize=(8,8))
    ax = fig[:add_subplot](111)

    for mm in mmm
        add_bayes2ax(mm,ax,priors)
    end
    
    ax[:set_ylabel]("Bayes Factor")
    ax[:set_xlabel](L"$\sigma$ prior")
    fig, ax    
end




end
