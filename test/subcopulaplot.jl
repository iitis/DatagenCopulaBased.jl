using DatagenCopulaBased
using NPZ

t = 350000
n = 128


asymn = [5,15,25,35,45]
symn = [10,20,30,40,50]

data = subcopdatagen(asymn, symn, t, n);

npzwrite("finalsnpz/datatest.npy", data)
data = npzread("finalsnpz/datatest.npy")
#plots distributions
data

using PyPlot
plot(data[:,5], data[:,15], "o")
plot(data[:,10], data[:,20], "o")
plot(data[:,8], data[:,9], "o")
plt[:hist](data[:,5], 20)
plt[:hist](data[:,15], 20)
plt[:hist](data[:,10], 20)
plt[:hist](data[:,20], 20)
plt[:hist](data[:,8], 20)
plt[:hist](data[:,9], 20)
