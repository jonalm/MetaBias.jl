using MetaBias
using FactCheck
using Distributions

srand(42) # fixed random seed

# data used in tests
zi, vi = import_sampledata("../data/sampledata.hdf5", "/konstantopoulos");
y = zi
σ = sqrt(vi)
np = NullPosterior(zi,vi)
r = randn(5)*2*std(np) + mean(np)

facts("updating τ is consistent with calculating from scratch") do
    lp0 = MixModelLikelihoodPrior(zi,vi,2.0)
    lp0 = reset_τ(lp0, 3.0)
    lp1 = MixModelLikelihoodPrior(zi,vi,3.0)
    @fact lp0.ndp --> lp1.ndp
end

facts("mean and var are consistent for NullDensityParam and NullPosterior") do
    τ = 3.3
    ndp = MetaBias.NullDensityParam(zi,vi,τ)
    np = NullPosterior(zi,vi,τ)
    @fact var(ndp) --> roughly(var(np))
    @fact mean(ndp) --> roughly(mean(np))
end

facts("Null likelihood prior density proportioal to null pdf") do
    τ = 3.0
    np = NullPosterior(zi,vi,τ)
    mm = MixModelLikelihoodPrior(zi,vi,τ)
    ratio_vec = pdf(np, r) ./ nulldensity(mm, r)
    ratio = ratio_vec[1]
    for i in 2:length(r)
        @fact ratio_vec[i] --> roughly(ratio)
    end
end

facts("Normalised Null likelihood prior is consistent with null pdf") do
    τ = 3.0
    np = NullPosterior(zi,vi,τ)
    mm = MixModelLikelihoodPrior(zi,vi,τ)
    m = nullmass(mm)
    unity_vec = (nulldensity(mm, r) / m) ./ pdf(np, r)
    for u in unity_vec
        @fact u --> roughly(1.0)
    end
end

facts("Test η=1 corresponds to Null") do
    mm = MixModelLikelihoodPrior(zi,vi)
    for ri in r
        @fact nulldensity(mm,ri) --> roughly(density(mm, [1., ri]))
    end
end

facts("Test single vs multiple input for density(NullLikelihoodPrior, x)") do
    mm = MixModelLikelihoodPrior(zi,vi)
    for ri in r
        @fact nulldensity(mm, ri) --> roughly(nulldensity(mm, [ri])[1])
    end
end

facts("Test single vs multiple input for density(MixModelLikelihoodPrior, x)") do
    mmlp = MixModelLikelihoodPrior(zi,vi)
    for ri in r
        a = [0.9, ri]
        b = a''
        @fact density(mmlp, a) --> roughly(density(mmlp, b)[1])
    end
end

facts("Test logpdf(MixModel) is consisitent with logdensity(MixModelLikelihoodPrior))") do
    #test various parameters
    for (η,μ,σ_prior,Z) in ((0.9,0.2,2.0,1.96), (0.1,-1.0,4.0, 1.4), (0.5,0.0,2.4,1.2))
        logres1 = logpdf(Normal(0.0, σ_prior), μ) #prior
        for (zii, vii) in zip(zi, vi)
            logres1 += logpdf(MixModel(η,μ,sqrt(vii),Z), zii)
        end

        mmlp = MixModelLikelihoodPrior(zi,vi,σ_prior,Z)
        logres2 = MetaBias.logdensity(mmlp, [η,μ])
        @fact logres1 --> roughly(logres2)
    end
end

facts("Test pdf(MixModel) is consisitent with density(MixModelLikelihoodPrior)") do
    #test various parameters
    for (η,μ,σ_prior,Z) in ((0.9,0.2,2.0,1.96), (0.1,-1.0,4.0, 1.4), (0.5,0.0,2.4,1.2))
        logres1 = logpdf(Normal(0.0,σ_prior), μ) #prior
        for (zii, vii) in zip(zi, vi)
            logres1 += logpdf(MixModel(η,μ,sqrt(vii),Z), zii)
        end

        mmlp = MixModelLikelihoodPrior(zi,vi,σ_prior,Z)
        res2 = density(mmlp, [η,μ])
        @fact exp(logres1) --> roughly(res2)
    end
end

