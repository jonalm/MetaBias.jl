using MetaBias
using FactCheck
using Distributions

srand(42) # fixed random seed

# data used in tests
zi, vi = import_sampledata("../data/sampledata.hdf5", "/konstantopoulos");
np = NullPosterior(zi,vi)
r = randn(5)*2*std(np) + mean(np)

facts("abcd parameters equal in Null and MixModel") do
    lp0 = NullLikelihoodPrior(zi,vi)
    lp1 = MixModelLikelihoodPrior(zi,vi)
    @fact lp0.ndp --> lp1.ndp
end

facts("equal default hyper parameter τ in Null and MixModel") do
    lp0 = NullLikelihoodPrior(zi,vi)
    lp1 = MixModelLikelihoodPrior(zi,vi)
    @fact lp0.ndp.τ --> lp1.ndp.τ
end

facts("updating τ is consistent with calculating from scratch") do
    for LP in (NullLikelihoodPrior,MixModelLikelihoodPrior)
        lp0 = LP(zi,vi,2.0)
        lp0 = reset_τ(lp0, 3.0)
        lp1 = LP(zi,vi,3.0)
        @fact lp0.ndp --> lp1.ndp
    end
end

facts("mean and var are consistent for NullDensityParam and NullPosterior") do
    τ = 3.3
    ndp = MetaBias.NullDensityParam(zi,vi,τ)
    np = NullPosterior(zi,vi,τ)
    @fact var(ndp) --> roughly(var(np))
    @fact mean(ndp) --> roughly(mean(np))
end

facts("Null likelihod prior density proportioal to null pdf") do
    τ = 3.0
    np = NullPosterior(zi,vi,τ)
    nlp = NullLikelihoodPrior(zi,vi,τ)
    ratio_vec = pdf(np, r) ./ density(nlp, r)
    ratio = ratio_vec[1]
    for i in 2:length(r)
        @fact ratio_vec[i] --> roughly(ratio)
    end
end


facts("Test η=1 corresponds to Null") do
    mmlp = MixModelLikelihoodPrior(zi,vi)
    nlp  = NullLikelihoodPrior(zi,vi)
    for ri in r
        @fact density(nlp,ri) --> roughly(density(mmlp, [1., ri]))
    end
end

facts("Test single vs multiple input for density(NullLikelihoodPrior, x)") do
    nlp = NullLikelihoodPrior(zi,vi)
    for ri in r
        @fact density(nlp, ri) --> roughly(density(nlp, [ri])[1])
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
    for (η, μ, τ, Z) in ((0.9,0.2,2.0,1.96), (0.1,-1.0,4.0, 1.4), (0.5,0.0,2.4,1.2))

        logres1 = logpdf(Normal(0.0,sqrt(τ)), μ) #prior
        for (zii, vii) in zip(zi, vi)
            logres1 += logpdf(MixModel(η,μ,vii,Z), zii)
        end

        mmlp = MixModelLikelihoodPrior(zi,vi,τ,Z)
        logres2 = logdensity(mmlp, [η,μ])
        @fact logres1 --> roughly(logres2)
    end
end

facts("Test pdf(MixModel) is consisitent with density(MixModelLikelihoodPrior)") do
    #test various parameters
    for (η, μ, τ, Z) in ((0.9,0.2,2.0,1.96), (0.1,-1.0,4.0, 1.4), (0.5,0.0,2.4,1.2))

        logres1 = logpdf(Normal(0.0,sqrt(τ)), μ) #prior
        for (zii, vii) in zip(zi, vi)
            logres1 += logpdf(MixModel(η,μ,vii,Z), zii)
        end

        mmlp = MixModelLikelihoodPrior(zi,vi,τ,Z)
        res2 = density(mmlp, [η,μ])
        @fact exp(logres1) --> roughly(res2)
    end
end
