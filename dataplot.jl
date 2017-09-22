
#srand()
##
#data generator

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
#plots distributions


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
