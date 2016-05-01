using MetaBias
using HDF5
using Formatting

h5fn = "data/sampledata.hdf5"

namevec = h5open(h5fn, "r") do file
    names(file)
end

header = lpad("dataset", 16," ")*" |  Bayes Factor   "
println()
println(header)
println("-"^length(header))
for name in namevec
    effect, variance = import_sampledata(h5fn, name)
    σ = sqrt(variance)
    mm = MixModelLikelihoodPrior(effect, σ, 2.)
    bf = bayesfactor(mm)
    println(lpad(name, 16," ")*" | "*sprintf1("%10.3f", bf ))
end
println()



