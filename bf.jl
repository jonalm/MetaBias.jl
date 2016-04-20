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
    zi, vi = import_sampledata(h5fn, name)
    mm = MixModelLikelihoodPrior(zi, vi, 4.)
    bf = bayesfactor(mm)
    println(lpad(name, 16," ")*" | "*sprintf1("%10.3f", bf ))
end
println()



