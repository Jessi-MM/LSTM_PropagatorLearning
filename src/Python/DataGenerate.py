from ProtonTransferDVR import ProtonTransfer

# Example code to create m trajectories with the next:
# n=32 points on the grid defined in the interval [a=-1.5,b=1.5] angstroms
# using a time-dependent potential: time=True
# random variables in the system of potential: var_random=True
# data will be saving in save_dir

m = 100  # number of trajectories to generate
for i in range(m):
    d = ProtonTransfer(n=32, a=-1.5, b=1.5, time=True, var_random=True, save_dir='./Trajectories/data'+str(i))
    d.vector_potential(t=200, step=1)  # 200 time steps in the trajectory, delta t = 1
    d.evolution_wp(t=200, step=1, gaussiana=True)  # 200 time steps in the trajectory, delta t = 1, Gaussian as initial wave packet
