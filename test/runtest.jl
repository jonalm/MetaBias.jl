using MetaBias
using FactCheck
using Distributions

srand(42) # fixed random seed

# data used in tests
zi, vi = import_sampledata("../data/sampledata.hdf5", "/konstantopoulos");
r = randn(5)

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

facts("Null likelihod prior proportioal to density") do
    τ = 3.0
    nullposterior = NullPosterior(zi,vi,τ)
    display(pdf(nullposterior, r))
    #nulllikelihoodprior = 
    #@fact lp0.ndp --> lp1.ndp
end

