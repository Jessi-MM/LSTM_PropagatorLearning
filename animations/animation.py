import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import h5py
import matplotlib.animation as anim
color = ['#83b692','#f9ada0', '#f9627d', '#c65b7c', '#5b3758']


LSTM_compar = False

# Loading data LSTM predited
if LSTM_compar:
    data_path = '../Predictions/'
    h5f = h5py.File(data_path+'20241003-202051.h5', 'r')  # name of file
    X_vis = h5f.get('dataset_X')
    y_vis = h5f.get('dataset_y')
    y_lstm = h5f.get('dataset_p')
else:
    # Loading data DVR data generated
    data_path = '../data/DataNew/'  #input("Give the data path: ")
    h5f = h5py.File(data_path+'ngrid32_delta_20250116-193307.h5', 'r')  # name of file
    X_vis = h5f.get('dataset_X')
    y_vis = h5f.get('dataset_y')





# General parameters
n_grid = len(X_vis[0][0])//3  # number of points on the grid
seq_len = len(X_vis[0])  # number of steps in trajectories
a = -1.5  # initial point in angstroms
b = 1.5  # final point in angstroms
r_n = np.linspace(a,b,n_grid)

dat = 0 # Choosing a data

if LSTM_compar:
    X_r = X_vis[dat,:,0:n_grid]*np.sqrt(1/0.5291775)
    X_i = X_vis[dat,:,n_grid:2*n_grid]*np.sqrt(1/0.5291775)
    delta_r = y_lstm[dat,:,0:n_grid]*np.sqrt(1/0.5291775)
    delta_i = y_lstm[dat,:,n_grid:2*n_grid]*np.sqrt(1/0.5291775)
    pred_r = X_r + delta_r
    pred_i = X_i + delta_i

    
    X_de = np.vectorize(complex)(X_r,X_i)
    X_dens = ((np.abs(X_de))**2)*10# or *(1/0.5291775)

    pred_de = np.vectorize(complex)(pred_r,pred_i)
    pred_dens = ((np.abs(X_de))**2)*10

    pote = X_vis[dat,:,2*n_grid:3*n_grid]*(1/1.5936e-3)  # au to kcal/mol

    # Empty lists to use in shift function

    x = []   
    y = []  

    x1 = []  
    y1 = []  

    x2 = []  
    y2 = []

    x3 = []
    y3 = []


    fig, ax = plt.subplots()

    # Canvas settings
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-20,50])
    #ax.set_# XXX: label('$r$ $[\AA]$')
    #ax.set_title(r"Evolución temporal")

    # Defining real and imag ax plot

    #lwaver,  = ax.print()lot(x, y, "-", color = "green")
    #lwavei,  = ax.plot(x2, y2, "-", color = "blue")


    # Defining density ax plot

    dens_t, = ax.plot(x3, y3, "-")
    dens_lstm, = ax.plot(x2, y2, "-", color =color[4])
    potent, = ax.plot(x1, y1,color=color[3])

    def shift(frame):
        """
        frame: t (from 0 to 199)
        """
    
        '''
        lwaver.set_xdata(r_n)
        lwaver.set_ydata((onda[:,frame].real))
        lwaver.set_label('$\Psi(r,t)*10$_real t = '+str(frame)+' fs')
           
        lwavei.set_xdata(r_n)
        lwavei.set_ydata((onda[:,frame].imag))
        lwavei.set_label('$\Psi(r,t)*10$_imag t = '+str(frame)+' fs')
        '''
        potent.set_xdata(r_n)
        potent.set_ydata(pote[frame,:])
        potent.set_label('$V(r,t)$ t = '+str(frame)+' fs')
    
                  
        dens_t.set_xdata(r_n)
        dens_t.set_ydata(X_dens[frame,:])
        dens_t.set_label('$|\psi_{True}|$ t = '+str(frame)+' fs')
    
        dens_lstm.set_xdata(r_n)
        dens_lstm.set_ydata(pred_dens[frame,:])
        dens_lstm.set_label('$|\psi_{LSTM}|$ t = '+str(frame)+' fs')
    
        ax.legend()
    
        # Each time t it returns in the canvas:
    
        #return (lwaver, lwavei, potent, dens,)
        return (dens_t, dens_lstm, potent,)

    ani = anim.FuncAnimation(fig, shift, frames = 199, blit = False)
    plt.show()


    
else:

    X_r = X_vis[dat,:,0:n_grid]*np.sqrt(1/0.5291775)
    X_i = X_vis[dat,:,n_grid:2*n_grid]*np.sqrt(1/0.5291775)
    X_de = np.vectorize(complex)(X_r,X_i)
    X_dens = ((np.abs(X_de))**2)*10# or *(1/0.5291775)

    pote = X_vis[dat,:,2*n_grid:3*n_grid]*(1/1.5936e-3)  # au to kcal/mol


    # Empty lists to use in selfhift function

    x = []   
    y = []  
    
    x1 = []  
    y1 = []  
    
    x2 = []  
    y2 = []
    
    x3 = []
    y3 = []
    
    
    fig, ax = plt.subplots()
    
    # Canvas settings
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-20,50])
    #ax.set_xlabel('$r$ $[\AA]$')
    #ax.set_title(r"Evolución temporal")

    # Defining real and imag ax plot

    #lwaver,  = ax.plot(x, y, "-", color = "green")
    #lwavei,  = ax.plot(x2, y2, "-", color = "blue")


    # Defining density ax plot

    dens_t, = ax.plot(x3, y3, "-")
    #dens_lstm, = ax.plot(x2, y2, "-", color =color[4])
    potent, = ax.plot(x1, y1,color=color[3])

    def shift(frame):
        """
        frame: t (from 0 to 199)
        """
        
        '''
        lwaver.set_xdata(r_n)
        lwaver.set_ydata((onda[:,frame].real))
        lwaver.set_label('$\Psi(r,t)*10$_real t = '+str(frame)+' fs')
           
        lwavei.set_xdata(r_n)
        lwavei.set_ydata((onda[:,frame].imag))
        lwavei.set_label('$\Psi(r,t)*10$_imag t = '+str(frame)+' fs')
        '''
        potent.set_xdata(r_n)
        potent.set_ydata(pote[frame,:])
        potent.set_label('$V(r,t)$ t = '+str(frame)+' fs')
    
                  
        dens_t.set_xdata(r_n)
        dens_t.set_ydata(X_dens[frame,:])
        dens_t.set_label('$|\psi_{True}|$ t = '+str(frame)+' fs')
    
        #dens_lstm.set_xdata(r_n)
        #dens_lstm.set_ydata(pred_dens[frame,:])
        #dens_lstm.set_label('$|\psi_{LSTM}|$ t = '+str(frame)+' fs')
    
        ax.legend()
    
        # Each time t it returns in the canvas:
    
        #return (lwaver, lwavei, potent, dens,)
        return (dens_t, potent,)

    ani = anim.FuncAnimation(fig, shift, frames = 199, blit = False)
    plt.show()
