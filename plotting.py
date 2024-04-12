import numpy as np
import matplotlib.pyplot as plt


#Plot the difference between the first timesteps for the host kernel and cuda
#kernel, using n_steps=10

data1 = np.loadtxt("./data/host_u00000001.dat")
data2 = np.loadtxt("./data/cuda_u00000001.dat")
plt.plot(data1-data2)
plt.savefig("fig_Step1.png")
plt.clf()

data1 = np.loadtxt("./data/cuda_uInit.dat")
data2 = np.loadtxt("./data/shared_uInit.dat")
plt.plot(data1-data2)
plt.savefig("fig_initial.png")
plt.clf()


#Find the maximum difference between the host kernel and cuda, using n_steps=10
max_diff = 0
for i in np.arange(0,9):
    data1 = np.loadtxt("./data/host_u0000000{}.dat".format(i))
    data2 = np.loadtxt("./data/cuda_u0000000{}.dat".format(i))
    max_diff = np.max((np.max(np.abs(data1-data2)),max_diff))
print("Maximum cuda and host diff: ",max_diff)

#Find the maximum difference between the cuda kernel and cuda kernel using shared memory
#using n_steps=10

# max_diff = 0
# for i in np.arange(0,9):
#     data1 = np.loadtxt("data/shared_u0000000{}.dat".format(i))
#     data2 = np.loadtxt("data/cuda_u0000000{}.dat".format(i))
#     max_diff = np.max((np.max(np.abs(data1-data2)),max_diff))
# print("Maximum cuda and cuda shared diff: ",max_diff)


#Plot 

for i in np.arange(0,9):
    data = np.loadtxt("./data/shared_u0000000{}.dat".format(i))
    plt.plot(data,linestyle="--",label = str(i))
    # data = np.loadtxt("./data/host_u0000000{}.dat".format(i))
    # p = plt.plot(data)
    plt.legend()
plt.savefig("fig_10Steps.png")
plt.clf()

for i in np.arange(0,9):
    data = np.loadtxt("./data/shared_u000000{}0.dat".format(i))
    plt.plot(data,linestyle="--",label = str(i))
    # data = np.loadtxt("./data/host_u0000000{}.dat".format(i))
    # p = plt.plot(data)
    plt.legend()
plt.savefig("fig_100Steps.png")
plt.clf()

for i in np.arange(0,9):
    data = np.loadtxt("./data/host_u00{}00000.dat".format(i))
    p = plt.plot(data)
    # color = p[0].get_color()
    # data = np.loadtxt("./data/cuda_u00{}00000.dat".format(i))
    # plt.plot(data,linestyle=":")
    # plt.legend()
    data = np.loadtxt("./data/shared_u00{}00000.dat".format(i))
    plt.plot(data,linestyle="--")
plt.savefig("fig_1000000Steps.png")


