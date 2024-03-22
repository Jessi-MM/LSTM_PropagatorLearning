import random
import numpy as np
from numpy import ogrid
from numpy.linalg import eig
import matplotlib.pyplot as plt
import cmath
from tabulate import tabulate
import matplotlib.animation as anim


plt.style.use('seaborn')

r_n = np.linspace(-1.5,1.5,100)
onda = np.zeros((100,301),dtype = 'complex_')
pote = np.zeros((100,301))

for i in range(0,301,1):
    onda[:,i] = np.load('./data0_g/Wavepacket/'+str(i)+'-wave.npy')
    pote[:,i] = np.load('./data0_g/Potential/'+str(i)+'-potential.npy')*(1/1.5936e-3)

# Animation 

x = []
y = []

x1 = []
y1 = []


fig, ax = plt.subplots()

ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-15,50])
ax.set_xlabel('Distancia $r$ $[\AA]$')
#ax.set_ylabel('$\Psi(r,t)$')
ax.set_title(r"Evolución temporal")






lwave,  = ax.plot(x, y, "-", color = "C1")
potent, = ax.plot(x1, y1, "-", color = "C2")

def shift(frame):
    lwave.set_xdata(r_n)
    lwave.set_ydata((onda[:,frame].real)*10)
    lwave.set_label('$\Psi(r,t)*10$ t = '+str(frame)+' fs')
    
    potent.set_xdata(r_n)
    potent.set_ydata(pote[:,frame])
    potent.set_label('$V(r,t)$ t = '+str(frame)+' fs')
    
    ax.legend()
    

    
    return (potent, lwave,)

ani = anim.FuncAnimation(fig, shift, frames = 300, blit = False, interval=10, repeat = False)
#ani.save('animation', writer="ffmpeg")
fig.show()
