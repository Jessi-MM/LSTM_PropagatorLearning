import numpy as np
import torch
import random
import os
from scipy import integrate

class Potential_System:
    """
    This class generates the matrix of time-dependent Potentials modeled with Proton Transfer System
    """
    
    def __init__(self, time = True, var_random = True):
        
        
        if time == True:  # Time Dependent Potential
            if var_random == True:
                self.V = 10*1.5936e-3  # Electronic coupling [kcal/mol] -> au
                self.w1 = random.uniform(1500,4000)*4.556e-6   # Frecuencies of the harmonic proton potentials [cm^-1] -> au
                self.w2 = random.uniform(1500,4000)*4.556e-6   # Frecuencies of the harmonic proton potentials [cm^-1] -> au
                self.l = random.uniform(0,10)*1.5936e-3  # Amplitude of the energy bias [kcal/mol] -> au
                self.x_eq = random.uniform(-10,10)*1.5936e-3  # Equilibrium energy bias [kcal/mol] -> au
                self.w_x = 0.0148*(1/41.34144)  # Frequency of the energy bias oscillations [fs^-1] -> au
                self.th_x = random.uniform(0, 2*np.pi)  # Initial phase. Zero for the time-independent potentials
                self.R_eq = random.uniform(0.2, 1.0)*(1/0.5291775)  # Equilibrium distance between the minima of the harmonic potential [0.1nm] -> au 
                self.R_0 = random.uniform(0, self.R_eq)  # Initial displacement from equilibrium
                self.w_R = random.uniform(100, 300)*4.556e-6  # Frecuency of the proton-donor-acceptor distance oscillation [cm^-1] -> au
                self.th_R = random.uniform(0, 2*np.pi)  # Random initial phase 
                self.m = 1836  #The proton mass [au]
            
            if var_random == False:
                self.V = 10*1.5936e-3  # Electronic coupling [kcal/mol] -> au
                self.w1 = float(input('Enter w1 (Frecuencies of the harmonic proton potentials [cm^-1]): \n'))*4.556e-6
                self.w2 = float(input('Enter w2 (Frecuencies of the harmonic proton potentials [cm^-1]): \n'))*4.556e-6
                self.l = float(input('Enter l (Amplitude of the energy bias [kcal/mol]): \n'))*1.5936e-3  # Amplitude of the energy bias [kcal/mol] -> au
                self.x_eq = float(input('Enter x_eq (Equilibrium energy bias [kcal/mol]): \n'))*1.5936e-3  # Equilibrium energy bias [kcal/mol] -> au
                self.w_x = 0.0148*(1/41.34144)  # Frequency of the energy bias oscillations [fs^-1] -> au
                self.th_x = float(input('Enter th_x (Initial phase from 0 to 2pi [rad]): \n'))  # Initial phase. Zero for the time-independent potentials
                self.R_eq = float(input('Enter R_eq (Equilibrium distance between the minima of the harmonic potential [0.1nm]): \n'))*(1/0.5291775)  # Equilibrium distance between the minima of the harmonic potential [0.1nm] -> au 
                self.R_0 = float(input('Enter R_0 (Initial displacement from equilibrium from 0 to R_eq [0.1nm]): \n'))*(1/0.5291775)  # Initial displacement from equilibrium
                self.w_R = float(input('Enter w_r (Frecuency of the proton-donor-acceptor distance oscillation [cm^-1]): \n'))*4.556e-6  # Frecuency of the proton-donor-acceptor distance oscillation [cm^-1] -> au
                self.th_R = float(input('Enter th_R (Random initial phase from 0 to 2pi [rad])'))  # Random initial phase 
                self.m = 1836  #The proton mass [au]
                
                
                
        if time == False:  # Time Independent Potential
            if var_random == False:
                self.V = 10*1.5936e-3  # Electronic coupling [kcal/mol] -> au
                self.w1 = float(input('Enter w1 (Frecuencies of the harmonic proton potentials [cm^-1]): \n'))*4.556e-6   # Frecuencies of the harmonic proton potentials [cm^-1] -> au
                self.w2 = float(input('Enter w2 (Frecuencies of the harmonic proton potentials [cm^-1]): \n'))*4.556e-6   # Frecuencies of the harmonic proton potentials [cm^-1] -> au
                self.x_eq = float(input('Enter x_eq (Equilibrium energy bias [kcal/mol]): \n'))*1.5936e-3 # Equilibrium energy bias [kcal/mol] -> au
                self.R_eq = float(input('Enter R_eq (Equilibrium distance between the minima of the harmonic potential [0.1nm]): \n'))*(1/0.5291775) # Equilibrium distance between the minima of the harmonic potential [0.1nm] -> au 
                self.m = 1836  #The proton mass [au]
                
            if var_random == True:
                self.V =  10*1.5936e-3 # Electronic coupling [kcal/mol] -> au
                self.w1 = random.uniform(1500,4000)*4.556e-6# Frecuencies of the harmonic proton potentials [cm^-1] -> au
                self.w2 = random.uniform(1500,4000)*4.556e-6   # Frecuencies of the harmonic proton potentials [cm^-1] -> au
                self.x_eq = random.uniform(-10,10)*1.5936e-3  # Equilibrium energy bias [kcal/mol] -> au
                self.R_eq = random.uniform(0.2, 1.0)*(1/0.5291775)  # Equilibrium distance between the minima of the harmonic potential [0.1nm] -> au 
                self.m = 1836  #The proton mass [au]
                
                
    #=== System Functions ================================== 
    
    #--- Time Dependent Potential --------------------------
    def X(self, t):
        """
        Variables:
        l : Amplitude of the energy bias [a_0]
        w_x: Frequency of the energy bias oscillations [Jiffy^-1]
        th_x: Initial phase
        x_eq: Equilibrium energy bias [au]
    
        Input:
        t: Time [Jiffy]
    
        Output:
        X_t: Collective energy gap coordinate [au]
        """
        X_t = self.l*np.cos(self.w_x*t+self.th_x) + self.x_eq
        return X_t
    
    
    def R(self, t):
        """
        Variables:
        R_0: Initial displacement from equilibrium [a_0]
        R_eq: Equilibrium distance between the minima of the harmonic potential [a_0]
        w_R: Frecuency of the proton-donor-acceptor distance oscillation [cm^-1]
        th_R: Random initial phase
    
        Input:
        t : Time [Jiffy]
    
        Output:
        R_t: Vibrations of the proton donor and acceptor [a_0]
        """
        R_t = (self.R_0-self.R_eq)*np.cos(self.w_R*t+self.th_R) + self.R_eq
        return R_t
    
    def u1(self, r, t):
        """
        Variables:
        m : Proton mass [m_e]
        w1: Frecuencies of the harmonic proton potentials [cm^-1]
        R(t): Vibrations of the proton donor and acceptor [a_0]
    
        Input:
        r: Proton coordinate [a_0]
        t: Time [Jify]
    
        Output:
        u1_t: Harmonic oscillator potential [au]
    
        """
        u1_t = (1/2)*self.m*(self.w1**2)*(r + (self.R(t)/2))**2
        return u1_t


    def u2(self, r, t):
        """
        Variables:
        m : Proton mass [m_e]
        w2: Frecuencies of the harmonic proton potentials [cm^-1]
        R(t): Vibrations of the proton donor and acceptor [a_0]
    
        Input:
        r: Proton coordinate [a_0]
        t: Time [Jiffy]
    
        Output:
        u2_t: Harmonic oscillator potential [au]
    
        """
        u2_t = (1/2)*self.m*(self.w2**2)*(r - (self.R(t)/2))**2
        return u2_t
    
    #=== Potential of system ==============================
    #--- Time Dependent Potential
    
    def matrix_potential(self, r, t):
        """
        Input:
        r: Proton coordinate [a_0]
        t: Time [Jiffy]
    
        Output:
        Potential Matrix
    
        """
        matrix = torch.tensor([[self.u1(r,t),self.V],[self.V,self.u2(r,t) + self.X(t)]]).type(torch.float64)
        return matrix
    
    def potential(self, r, t):
        """
        Input:
        r: Proton coordinate [a_0]
        t: Time [Jiffy]
    
        Output:
        The lowest eigenvalue of potential matrix  
        """
        e_val = torch.linalg.eigvals(self.matrix_potential(r,t)).type(torch.float64)
    
        if e_val[0] < e_val[1]:
            pot = e_val[0]
        else:
            pot = e_val[1]
        return pot
    
    #--- Time Independent Potential ----------------------------
    def u1_TI(self, r):
        """
        Variables:
        m : Proton mass [m_e]
        w1: Frecuencies of the harmonic proton potentials [cm^-1]
        R_eq: Vibrations of the proton donor and acceptor [a_0]
    
        Input:
        r: Proton coordinate [a_0]
    
        Output:
        u1_t: Harmonic oscillator potential [au]
    
        """
        u1_t = (1/2)*self.m*(self.w1**2)*(r + (self.R_eq/2))**2
        return u1_t


    def u2_TI(self, r):
        """
        Variables:
        m : Proton mass [m_e]
        w2: Frecuencies of the harmonic proton potentials [cm^-1]
        R_eq: Vibrations of the proton donor and acceptor [a_0]
    
        Input:
        r: Proton coordinate [a_0]
    
        Output:
        u2_t: Harmonic oscillator potential [au]
    
        """
        u2_t = (1/2)*self.m*(self.w2**2)*(r - (self.R_eq/2))**2
        return u2_t
    
    #=== Potential of system ==============================
    #--- Time Independent Potential
    
    def matrix_potential_TI(self, r):
        """
        Input:
        r: Proton coordinate [a_0]
    
        Output:
        Potential Matrix
    
        """
        matrix = torch.tensor([[self.u1_TI(r),self.V],[self.V,self.u2_TI(r) + self.x_eq]])
        return matrix
    
    def potential_TI(self, r):
        """
        Input:
        r: Proton coordinate [a_0]
    
        Output:
        The lowest eigenvalue of potential matrix  
        """
        e_val = torch.linalg.eigvals(self.matrix_potential_TI(r))
    
        if e_val[0] < e_val[1]:
            pot = e_val[0]
        else:
            pot = e_val[1]
        return pot

            
class ProtonTransfer(Potential_System):
    """
    This class propagates wavepacket under time-dependent Potential_System
    
    
    
    """
    
    def __init__(self, n, a, b, time, var_random, seq_len):
        """
        n: Number of points on the grid to DVR method
        a: Initial point of grid [angstroms]
        b: Final point of grid [angstroms]
        time:True if Time Dependent Potential / False if Time Independent Potential
        var_random: True: Initialization with random values / False: To give the values of potential parameters
        seq_len: Time of wavepacket propagation
        """
        
        self.n = n  # Number of points on the grid to DVR method
        self.a = a*(1/0.5291775)  # Initial point of grid [au]
        self.b = b*(1/0.5291775)  # Final point of grid [au]
        self.time = time  # True: Time Dependent Potential / False: Time Independent Potential
        self.var_random = var_random  # True: Initialization with random values 
        self.seq_len = seq_len

        # To save data:
        # X[i]: (wavepacket_realpart, wavepacket_imagpart, potential) at time t
        # y[i]: Difference between (wavepacket_realpart, wavepacket_imagpart) at time t+step and (wavepacket_realpart, wavepacket_imagpart) time t
        # with i in seq_len
        self.Xdat = torch.empty((self.seq_len,self.n*3), dtype=torch.float64 )
        self.ydat = torch.empty((self.seq_len,self.n*2), dtype=torch.float64 )
        
        self.r_n = np.linspace(self.a, self.b, self.n) # Grid position [au]
        super().__init__(time = self.time, var_random = self.var_random)

        
    
    def get_values_Potential(self):
        """
        Print the values of the potential parameters
        """
        header = f"{'Variable':<35} {'Value':>20}"
        print("=" * len(header))
        if self.time:
            print("====== Time Dependent Potential ======\n")
            rows = [
                ["V [kcal/mol]", self.V*(1/1.5936e-3)],
                ["w1 [cm^-1]", self.w1*(1/4.556e-6)],
                ["w2 [cm^-1]", self.w2*(1/4.556e-6)],
                ["l [kcal/mol]", self.l*(1/1.5936e-3)],
                ["x_eq [kcal/mol]", self.x_eq*(1/1.5936e-3)],
                ["w_x [fs^-1]", self.w_x*(41.34144)],
                ["th_x [rad]", self.th_x],
                ["R_eq [0.1 nm]", self.R_eq*(0.5291775)],
                ["R_0 [0.1 nm]", self.R_0*(0.5291775)],
                ["w_R [cm^-1]", self.w_R*(1/4.556e-6)],
                ["th_R [rad]", self.th_R],
                ["m (Proton mass [au])", self.m]
            ]
        else:
            print("====== Time Independent Potential ======\n")
            rows = [
                ["V [kcal/mol]", self.V*(1/1.5936e-3)],
                ["w1 [cm^-1]", self.w1*(1/4.556e-6)],
                ["w2 [cm^-1]", self.w2*(1/4.556e-6)],
                ["x_eq [kcal/mol]", self.x_eq*(1/1.5936e-3)],
                ["R_eq [0.1 nm]", self.R_eq*(0.5291775)],
                ["m (Proton mass [au])", self.m]
            ]

        print(header)
        print("=" * len(header))
        for var, val in rows:
            print(f"{var:<35} {val:>20.6f}")

        
    #=== Random wavepacket initial as a sum of spectral functions basis =================
        
    def __eigenestados__(self, r, m):
        a1 = self.a*0.5291775  # au -> angstroms
        b1 = self.b*0.5291775  # au -> angstroms
        e_m =((2/(self.b-self.a))**(1/2))*torch.sin((m*np.pi*(r-self.a))/(self.b-self.a))
        return e_m
        
    
    def __phi_i__(self, m):
        phi = self.__eigenestados__(self.r_n, m)
        return phi
    
    def __random_Ci__(self):
        cix = random.random()
        ciy = random.random()
        Ci = complex(cix,ciy)
        return Ci
    
    def Wavepacket_Init(self,k):
        #random.seed(0)
        Ci_norm = []
        wavei = torch.empty((k,self.n), dtype =torch.complex64)
        for i in range(1,k+1):
            Ci = self.__random_Ci__()
            Ci_norm.append(np.abs(Ci)**2)
            wavei[i-1,:] = Ci*(self.__phi_i__(i))
        wave = torch.sum(wavei[i] for i in range(k))         
        
        return wave/torch.sqrt(sum(Ci_norm))
    
    #=== Random wavepacket initial as a sum of pseudospectral functions basis =================
    def gaussi(self):
        r_n = torch.from_numpy(self.r_n)
        mu = np.random.uniform(-0.5,0.5)*(1/0.5291775)
        sigma = np.random.uniform(0.1,0.3)*(1/0.5291775)
        Ci = self.__random_Ci__()
        g = Ci*(1/sigma*np.sqrt(2*np.pi))*torch.exp(-((r_n-mu)**2)/(2*sigma**2))
        
        
    
        norm = torch.trapz((torch.abs(g)**2), r_n)
        return g/np.sqrt(norm)
    
    
    
    #=== Vector Potential =======================================
    
    def vector_potential(self, t, step=1):
        """
        Input:
        t: Time [fs]
    
        Output:
        Potential in the grid at time t (lenght: 32)
        """
        for j in range(0,t+1, step):
            v_pot = np.zeros(self.n)
            for i, item in enumerate(self.r_n):
                v_pot[i] = self.potential(item,j*41.34)  # fs -> au
        return v_pot
        
    
    def vector_potential_TI(self):
        """
    
        Output:
        Potential in the grid (lenght: 32)
        """
        v_pot = np.zeros(self.n)
        for i, item in enumerate(self.r_n):
            v_pot[i] = self.potential_TI(item)
        return v_pot
    
    
    #=== Kinetic Energy Matrix =================================
    
    def T_spec(self):
        """
        Kinetic Energy Matrix in the spectral basis
        """
        h_bar = 1  # Planck constant [au]
        N = self.n
        L = self.b - self.a
    
        T = torch.empty((N,N))
    
        for i in range(1,N+1):
            for j in range(1,N+1):
            
                if i != j:
                    T[i-1,j-1] = 0
                else:
                    T[i-1,j-1] = (h_bar**2/(2*self.m))*((j*np.pi)/L)**2 
                
        return T
    
    def Unitary_Matrix(self):
        """
        Unitary Matrix of transformation from spectral to pseudospectral basis 
        """
        N = self.n
    
        U = torch.empty((N, N))
    
        for i in range(1,N+1):
            for j in range(1,N+1):
    
                U[i-1,j-1] = np.sqrt(2/(N+1))*np.sin((i*j*np.pi)/(N+1))
    
        return U
    

    def KINETIC_DVR(self):
        """
        Kinetic Energy Matrix in the pseudospectral basis
        """
        N = self.n
        
        U = self.Unitary_Matrix()
        T_spectral = self.T_spec()
        
        T_DVR = torch.matmul(torch.matmul(U,T_spectral),U)
        
        return T_DVR
    
    #----------------------- old version 
    def KINETIC_DVR_old(self):
        """
        This is an *approximation* to Kinetic Energy Matrix in pseudospectral basis
        """
        h_bar = 1  # Planck constant [au]
        co = ((h_bar**2)/(2*self.m))*((np.pi**2)/(2))*(1/((self.b-self.a)**2))  # coeff
        N = self.n
        
        T_DVR = torch.zeros(self.n,self.n)
        
        for i in range(1,N-1):
            for j in range(1,N-1):
                
                if i == j:
                    T_DVR[i,i] = co*( ((2*N**2+1)/3) - (1/((np.sin((np.pi*i)/N))**2)) )
                else:
                    T_DVR[i,j] = ((-1)**(i-j))*co*( (1/((np.sin((np.pi*(i-j))/(2*N)))**2)) - (1/((np.sin((np.pi*(i+j))/(2*N)))**2)) )
        
        return T_DVR
    #----------------------------- old version
    

    
    #=== Potential Matrix ==============================
    def V_DVR(self, t):
        """
        Input: t [au]
    
        Output: Matriz de potencial en el grid: V_DVR [au]
        """
        t = t*41.34  # fs -> au
        
        V = torch.zeros(self.n,self.n, dtype=torch.float64)
    
        for i, r in enumerate(self.r_n):
            V[i,i] = self.potential(r,t)#*(1/1.5936e-3)  # time: fs -> au, energy: Hartree -> kcal/mol
        
        return V
    
    #=== Hamiltonian Matrix ==============================
    def H_DVR(self,t, T_DVR):
        """
        input: tiempo [fs]
        T: kinetic energy matrix (constant everytime)
    
        output: Matriz Hamiltoniana DVR [au]
        """
        #T_DVR = self.KINETIC_DVR()
        
        H = T_DVR + self.V_DVR(t)
        return H
    
    
    
    #=== Propagation wavepacket Time Dependent Potential ==========================
    # Find eigenvalues and eigenvectors:
    def eigenN(self,t, T_DVR):
        
        H_DVR = self.H_DVR(t, T_DVR).type(torch.complex128)
        
        Eigen_n, U = torch.linalg.eig(self.H_DVR(t, T_DVR))
    
        U_inv = torch.linalg.inv(U)
        
        
        D = torch.mm(torch.mm(U_inv, H_DVR),U)
        D = (complex(0,-1))*t*D

        
        return Eigen_n, U, U_inv, D
    
    
    def Psi_VDR_t(self, t, Psi_DVR_inicial, T_DVR):
        """
        Input:
        t: time 
        Psi_DVR_inicial: wavepacket
        T_DVR: Kinetic Energy Matrix
        
        Output:
        Psi_DVR_final: wavepacket propagated one step next under time-dependent potential
        """
    
        Eigen_n, U, U_inv, D = self.eigenN(t, T_DVR)
    
        Diag = torch.zeros((self.n,self.n), dtype =torch.complex128)
        for i in range(self.n):
            Diag[i][i] = np.exp(torch.diag(D))[i]
    
    
        Psi_DVR_final = torch.matmul((torch.mm(torch.mm(U, Diag),U_inv)),Psi_DVR_inicial)
        
        # Normalization
        norm = torch.trapz((torch.abs(Psi_DVR_final)**2), torch.from_numpy(self.r_n))
        
    
        return Psi_DVR_final/np.sqrt(norm)
    
    
    def evolution_wp(self, time, step, gaussiana, T_DVR):
        """
        Function that calculates the evolution of the initial wavepacket at a time t under the given potential.
        Time intervals are 1 fs
    
        Input:
        t: evolution time [fs]
        step: Time intervals (1 recomended)
        gaussiana: Initial wavepacket
        T_DVR: Kinetic Energy Matrix (always the same)
    
        Output:
        wp: Wavepacket evolved with DVR method under potential V(t)
        with a special format to save data to train the network
   
        """
        n = self.n  # Points on grid
        t = time
        
    
        if t == 0:
            if gaussiana == True:
            
                wp = self.gaussi()  # wavepacket at time t=0 (random gaussian)

                self.Xdat[t,0:n] = wp.real
                self.Xdat[t,n:n*2] = wp.imag
                self.Xdat[t,n*2:n*3] = torch.diag(self.V_DVR(t))

            return wp
        
        else:
            # If t is greater than 0, the wavepacket at time t is the result of propagating the previous wavepacket at time t-1
            
            wp = self.Psi_VDR_t(t, self.evolution_wp(t-step, step, gaussiana, T_DVR),T_DVR)

            if t < self.seq_len:
                # Saving wavepacket and potential at time t
                self.Xdat[t,0:n] = wp.real
                self.Xdat[t,n:n*2] = wp.imag
                self.Xdat[t,n*2:n*3] = torch.diag(self.V_DVR(t))
                
                # Saving difference of wavepacket at time t+step and wavepacket at time t
                self.ydat[t-1,0:n] = wp.real - self.Xdat[t-1,0:n]
                self.ydat[t-1,n:n*2] = wp.imag - self.Xdat[t-1,n:n*2]

            elif t == self.seq_len:

                self.ydat[t-1,0:n] = wp.real - self.Xdat[t-1,0:n]
                self.ydat[t-1,n:n*2] = wp.imag - self.Xdat[t-1,n:n*2]



            return wp

                   
    
    #=== Time Independent Propagation wavepacket ==========================
    
    #=== Time Independent Potential Matrix ==============================
    def V_DVR_TI(self):
        """
        Output: Matriz de potencial en el grid: V_DVR [au]
        """
        V = np.zeros((self.n,self.n))
    
        for i, r in enumerate(self.r_n):
            V[i,i] = self.potential_TI(r)#*(1/1.5936e-3)  # time: fs -> au, energy: Hartree -> kcal/mol
        
        return V
    
    #=== Time Independent Hamiltonian Matrix ==============================
    def H_DVR_TI(self):
        """
    
        output: Matriz Hamiltoniana DVR [au]
        """
        T_DVR = self.KINETIC_DVR()
        H = T_DVR + self.V_DVR_TI()
        return H
    
    
    # Find eigenvalues and eigenvectors:
    def eigenN_TI(self):
        Eigen_n, U = eig(self.H_DVR_TI())
    
        U_inv = np.linalg.inv(U)
    
        D = np.dot(np.dot(U_inv, self.H_DVR_TI()),U)
        D = (complex(0,-1))*t*D
        
        return Eigen_n, U, U_inv, D
    
    
    def Psi_VDR_t_TI(self, Psi_DVR_inicial):
    
        Eigen_n, U, U_inv, D = self.eigenN_TI()
    
        Diag = np.zeros([self.n,self.n], dtype = 'complex_')
        for i in range(self.n):
            Diag[i][i] = np.exp(np.diagonal(D))[i]
    
    
        Psi_DVR_final = np.dot((np.dot(np.dot(U, Diag),U_inv)),Psi_DVR_inicial)
    
        # Normalizacion:
        norm = integrate.simpson((np.abs(Psi_DVR_final)**2), self.r_n)

        
    
        return Psi_DVR_final/np.sqrt(norm)
    
    
    def evolution_wp_TI(self, t, step, gaussiana):

        if t == 0:
            if gaussiana == False:
                wp = self.Wavepacket_Init()
                with open(os.path.join('./'+self.save_dir+'/Wavepacket', str(t)+'-wave.npy'), 'wb') as f:
                    np.save(f, wp)
                return wp 
            if gaussiana == True:
                wp = self.gaussi()
                with open(os.path.join('./'+self.save_dir+'/Wavepacket', str(t)+'-wave.npy'), 'wb') as f:
                    np.save(f, wp)
                return wp 
        else:
            wp = self.Psi_VDR_t_TI(self.evolution_wp_TI(t-step, step, gaussiana))
            with open(os.path.join('./'+self.save_dir+'/Wavepacket', str(t)+'-wave.npy'), 'wb') as f:
                np.save(f, wp)
            return wp
    
    
    def density(self, wavepacket):
        dens = (np.abs(wavepacket))**2
        return dens
            