using CumBandSel
if true
  addprocs()
  @everywhere using CumBandSel
end
using Distributions
using NPZ
import CumBandSel: cumulants, optfunction3, optfunction34, optfunction4, getcum

#srand()
##
#data generator
testapprox(3)
testapprox(4)

pixel_n = 350000
band_n = 128
cum_max = 4
bls = 3

asymbands = [5,15,25,35,45,55,65,75,85,95,105]
symbands = [10,20,30,40,50,60,70,80,90,100,110]


asymbands = [5,15,25,35,45]
symbands = [10,20,30,40,50]

data = CumBandSel.datagen(asymbands, symbands, pixel_n, band_n);

npzwrite("finalsnpz/datatest.npy", data)
data = npzread("finalsnpz/datatest.npy")
c = getcum(data, 4, 2);
using JLD
save("cums/tests.jld", "c", c, "x", data, "order", cum_max)
d = load("cums/tests.jld")
carray = [convert(Array, c[i]) for i in 1:4];

@time a = greedesearchdata(carray, optfunction3, 127)
@time b = greedesearchdata(carray, optfunction4, 127)
@time c = greedesearchdata(carray, optfunction34, 127)


find(c[106][1])

#plots distributions

using JLD
using SymmetricTensors
using PyPlot

function co(ls)
  ret = zeros(size(ls,1))
  for i in 1:size(ls, 1)
    ret[i] = ls[i][2]
  end
  ret
end

function readf(files = ["b1.jld", "b2.jld", "b3.jld"])
  l = []
  for file in files
    push!(l, load("cums_hole/"*file))
  end
  l
end


function plotres(l, files = ["b1.jld", "b2.jld", "b3.jld"], opts = ["optfunction34"],
                                                                pltlim::Int = 1,
                                                                lim::Int = 108)
  col = ["k", "r", "b", "g", "y", "c"]
  lin = ["-", "--", ":"]
  k = 1
  for opt in opts
    line = (pltlim  < 115)? lin[k]: "-d"
    k += 1
    for i in 1:length(l)
      f = replace(files[i], ".jld", "")*" "*replace(opt, "optfunction", "")
      println(f)
      plot(co(l[i][opt])[pltlim:end], color = col[i], line, label = f)
      println(find(l[i][opt][lim][1]))
    end
  end
  plt[:legend]()
end

files = ["b1_clean.jld", "b1.jld", "b2.jld", "b3.jld"]
files = ["c6_clean.jld", "c1.jld", "c2.jld", "c3.jld"]
files = ["d2_clean.jld", "d1.jld", "d2.jld", "d3.jld"]
files = ["g6_clean.jld", "g1.jld", "g2.jld", "g3.jld"]
files = ["k2_clean.jld", "k1.jld", "k2.jld", "k3.jld"]
files = ["p1_clean.jld", "p1.jld", "p2.jld", "p3.jld"]
files = ["s1_clean.jld", "s1.jld", "s2.jld", "s3.jld"]
files = ["w4_clean.jld", "w1.jld", "w2.jld", "w3.jld"]
l = readf(files)
plotres(l, files, ["optfunction34", "optfunction4", "optfunction3"], 110, 110)


ls = readf(["c6_clean.jld", "c1.jld", "c2.jld", "c3.jld"]);

c6_clean = ls[1]["optfunction34"]
c1= ls[2]["optfunction34"]
c2= ls[3]["optfunction34"]
c3 = ls[4]["optfunction34"]


c = c3
ret = zeros(Int, 127)
temp = Int[]
for i in 1:127
  a = find(!c[i][1])
  for k in a
    if !(k in temp)
      ret[i] = k
    end
  end
  temp = a
end
ret

println("c3")
println(ret)



using PyPlot
plot(data[:,10], data[:,15], "o")
plot(data[:,7], data[:,14], "o")
plot(data[:,8], data[:,9], "o")
plt[:hist](data[:,7], 20)
plt[:hist](data[:,8], 20)
plt[:hist](data[:,9], 20)
plt[:hist](data[:,10], 20)
plt[:hist](data[:,14], 20)
plt[:hist](data[:,15], 20)

n = "_10"
l = 34
fig, ax = subplots()
y = log.(targfval34[1:end])
l3 = length(asymbands)
l4 = length(symbands)
k = band_n - l4 -l3
ax[:plot](y, color = "blue", "--D", label= "\$ C_{$l} \\;asym =$asymbands \\; kurt=$symbands\$")
ax[:legend](fontsize = 10, loc = 2, ncol = 2)
ax[:set_xlabel]("\$ k - number \\; of \\; eliminated \\; bands \$")
ax[:set_ylabel]("\$\ log(f(C..., k)) \$")
ax[:vlines](k, minimum(y), maximum(y))

savefig("targetf_C_$l$n")
