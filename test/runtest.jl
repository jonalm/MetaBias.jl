using MetaBias
using FactCheck

zi, vi = import_sampledata("../data/sampledata.hdf5", "/konstantopoulos");

facts("abcd parameters equal in Null and MixModel") do
    lp0 = Null(zi,vi)
    lp1 = MixModel(zi,vi)
    @fact lp0.ndp --> lp1.ndp
end

facts("equal default hyper parameter τ in Null and MixModel") do
    lp0 = Null(zi,vi)
    lp1 = MixModel(zi,vi)
    @fact lp0.ndp.τ --> lp1.ndp.τ
end

facts("updating τ is consistent with calculating from scratch") do
    for LP in (MixModel,)
        lp0 = LP(zi,vi,2.0)
        lp0 = reset_τ(lp0, 3.0)
        lp1 = LP(zi,vi,3.0)
        @fact lp0.ndp --> lp1.ndp
    end
end
