using MetaBias
using FactCheck
using Distributions

srand(42) # fixed random seed

# data used in tests
zi, vi = import_sampledata("../data/sampledata.hdf5", "/konstantopoulos");


facts("abcd parameters equal in Null and MixModel") do
    lp0 = NullLikelihoodPrior(zi,vi)
    lp1 = MixModelLikelihoodPrior(zi,vi)
    @fact lp0.ndp --> lp1.ndp
end

facts("equal default hyper parameter τ in Null and MixModel") do
    lp0 = NullLikelihoodPrior(zi,vi)
    lp1 = MixModelLikelihoodPrior(zi,vi)
    @fact lp0.ndp.τ --> lp1.ndp.τ
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

facts("Null likelihod prior density proportioal to null pdf") do
    N = 10
    τ = 3.0
    np = NullPosterior(zi,vi,τ)
    nlp = NullLikelihoodPrior(zi,vi,τ)
    r = randn(10)*2*std(np) + mean(np)
    ratio_vec = pdf(np, r) ./ density(nlp, r)
    ratio = ratio_vec[1]
    for i in 2:N
        @fact ratio_vec[i] --> roughly(ratio)
    end
end

