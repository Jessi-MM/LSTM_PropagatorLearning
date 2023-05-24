import random
import numpy as np
from numpy import ogrid
from numpy.linalg import eig
import matplotlib.pyplot as plt
import cmath
from scipy import integrate
from tabulate import tabulate
import matplotlib.animation as anim
import os

class Potential_System:
    
    def __init__(self, time = True, var_random = True):
        
        # Potential System variables:
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
        matrix = np.array([[self.u1(r,t),self.V],[self.V,self.u2(r,t) + self.X(t)]])
        return matrix
    
    def potential(self, r, t):
        """
        Input:
        r: Proton coordinate [a_0]
        t: Time [Jiffy]
    
        Output:
        The lowest eigenvalue of potential matrix  
        """
        e_val = np.linalg.eigvals(self.matrix_potential(r,t))
    
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
        matrix = np.array([[self.u1_TI(r),self.V],[self.V,self.u2_TI(r) + self.x_eq]])
        return matrix
    
    def potential_TI(self, r):
        """
        Input:
        r: Proton coordinate [a_0]
    
        Output:
        The lowest eigenvalue of potential matrix  
        """
        e_val = np.linalg.eigvals(self.matrix_potential_TI(r))
    
        if e_val[0] < e_val[1]:
            pot = e_val[0]
        else:
            pot = e_val[1]
        return pot

class ProtonTransfer(Potential_System):
    
    def __init__(self, n, a, b, time, var_random, save_dir):
        """
        Input:
        n: Number of points on the grid to DVR method
        a: Initial point of grid [angstroms]
        b: Final point of grid [angstroms]
        k: Number of eigenestatos included in the initial wavepacket
        
        Output:
        a,b & grid r_n in atomic units: au 
        """
        
        self.n = n  # Number of points on the grid to DVR method
        self.a = a*(1/0.5291775)  # Initial point of grid [au]
        self.b = b*(1/0.5291775)  # Final point of grid [au]
        #self.k = k  # Number of eigenestatos included in the initial wavepacket
        self.time = time  # True: Time Dependent Potential / False: Time Independent Potential
        self.var_random = var_random  # True: Initialization with random values 
        self.save_dir = save_dir
        
        os.makedirs('./'+self.save_dir+'/Wavepacket')
        os.makedirs('./'+self.save_dir+'/Potential')
        
        self.r_n = ogrid[self.a:self.b:(0+1j)*self.n]  # Grid position [au]
        super().__init__(time = self.time, var_random = self.var_random)
        
    
    def get_values_Potential(self):
        if self.time == True:
            with open('./'+self.save_dir+'/Values_PotentialSystem.txt', 'w') as f:
                f.write('====== Time Dependent Potential ======\n')
                f.write(tabulate([["V [kcal/mol]", self.V*(1/1.5936e-3)],["w1 [cm^-1]", self.w1*(1/4.556e-6)], ["w2 [cm^-1]", self.w2*(1/4.556e-6)], ["l [kcal/mol]", self.l*(1/1.5936e-3)], ["x_eq [kcal/mol]", self.x_eq*(1/1.5936e-3)], ["w_x [fs^-1]", self.w_x*(41.34144)], ["th_x [rad]", self.th_x], ["R_eq [0.1 nm]", self.R_eq*(0.5291775)], ["R_0 [0.1 nm]", self.R_0*(0.5291775)], ["w_R [cm^-1]", self.w_R*(1/4.556e-6)], ["th_R [rad]", self.th_R], ["m (Proton mass [au])", self.m]], headers=['Variable', 'Value'], tablefmt='orgtbl'))
            print('====== Time Dependent Potential ======\n')
            return print(tabulate([["V [kcal/mol]", self.V*(1/1.5936e-3)],["w1 [cm^-1]", self.w1*(1/4.556e-6)], ["w2 [cm^-1]", self.w2*(1/4.556e-6)], ["l [kcal/mol]", self.l*(1/1.5936e-3)], ["x_eq [kcal/mol]", self.x_eq*(1/1.5936e-3)], ["w_x [fs^-1]", self.w_x*(41.34144)], ["th_x [rad]", self.th_x], ["R_eq [0.1 nm]", self.R_eq*(0.5291775)], ["R_0 [0.1 nm]", self.R_0*(0.5291775)], ["w_R [cm^-1]", self.w_R*(1/4.556e-6)], ["th_R [rad]", self.th_R], ["m (Proton mass [au])", self.m]], headers=['Variable', 'Value'], tablefmt='orgtbl'))
        
        if self.time == False:
            with open('./'+self.save_dir+'/Values_PotentialSystem.txt', 'w') as f:
                f.write('====== Time Independent Potential ======')
                f.write((tabulate([["V [kcal/mol]", self.V*(1/1.5936e-3)],["w1 [cm^-1]", self.w1*(1/4.556e-6)], ["w2 [cm^-1]", self.w2*(1/4.556e-6)], ["x_eq [kcal/mol]", self.x_eq*(1/1.5936e-3)], ["R_eq [0.1 nm]", self.R_eq*(0.5291775)], ["m (Proton mass [au])", self.m]], headers=['Variable', 'Value'], tablefmt='orgtbl')))
            print("====== Time Independent Potential ======")
            return print(tabulate([["V [kcal/mol]", self.V*(1/1.5936e-3)],["w1 [cm^-1]", self.w1*(1/4.556e-6)], ["w2 [cm^-1]", self.w2*(1/4.556e-6)], ["x_eq [kcal/mol]", self.x_eq*(1/1.5936e-3)], ["R_eq [0.1 nm]", self.R_eq*(0.5291775)], ["m (Proton mass [au])", self.m]], headers=['Variable', 'Value'], tablefmt='orgtbl'))
        
        
    def get_values_Trajectory(self, t, step):
        with open('./'+self.save_dir+'/Values_Trajectory.txt', 'w') as f:
                f.write(tabulate([["Divisiones en el grid (n):", self.n],["Number of eigenstates to generate the initial wavepacket (k):", self.k],["Time of trajectory (t):", t], ["Step of trajectory:", step]],headers=['Variable', 'Value'], tablefmt='orgtbl'))


    #=== Random wavepacket initial =============================================
        
    def __eigenestados__(self, r, m):
        a1 = self.a*0.5291775  # au -> angstroms
        b1 = self.b*0.5291775  # au -> angstroms
        e_m =((2/(self.b-self.a))**(1/2))*np.sin((m*np.pi*(r-self.a))/(self.b-self.a))
        return e_m
        
    
    def __phi_i__(self, m):
        phi = np.zeros(self.n)
        j = 0
        for j, i in enumerate(self.r_n):
            phi[j] = self.__eigenestados__(i, m)
        return phi
    
    def __random_Ci__(self):
        cix = random.random()
        ciy = random.random()
        Ci = complex(cix,ciy)
        return Ci
    
    def Wavepacket_Init(self,k):
        #random.seed(0)
        Ci_norm = []
        wavei = np.zeros([k,self.n], dtype = 'complex_')
        for i in range(1,k+1):
            Ci = self.__random_Ci__()
            Ci_norm.append(np.abs(Ci)**2)
            wavei[i-1,:] = Ci*(self.__phi_i__(i))
        wave = sum(wavei[i] for i in range(k))         
        
        return wave/np.sqrt(sum(Ci_norm))
    
    
    def gaussi(self):
        mu = np.random.uniform(-0.5,0.5)*(1/0.5291775)
        sigma = np.random.uniform(0.1,0.3)*(1/0.5291775)
        Ci = self.__random_Ci__()
        g = Ci*(1/sigma*np.sqrt(2*np.pi))*np.exp(-((self.r_n-mu)**2)/(2*sigma**2))
    
        norm = integrate.simpson((np.abs(g)**2), self.r_n)
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
            with open(os.path.join('./'+self.save_dir+'/Potential', str(j)+'-potential.npy'), 'wb') as f:
                np.save(f, v_pot)
            
        
        return v_pot
        
    
    def vector_potential_TI(self):
        """
    
        Output:
        Potential in the grid (lenght: 32)
        """
        v_pot = np.zeros(self.n)
        for i, item in enumerate(self.r_n):
            v_pot[i] = self.potential_TI(item)
            
        with open(os.path.join('./'+self.save_dir+'/Potential', str(t)+'-potential.npy'), 'wb') as f:
                np.save(f, v_pot)
        
        return v_pot
    
    
    #=== Kinetic Energy Matrix =================================
    def __senno__(self, k,l,n):
        y = (n**2)*(np.sin((n*np.pi*k)/self.n))*(np.sin((n*np.pi*l)/self.n))
        return y


    def KINETIC_DVR(self):
        h_bar = 1  # Constante de Planck en au
        T_c = ((h_bar**2)/(2*self.m))*((np.pi/(self.b-self.a))**2)*(2/self.n)
        T1 = np.arange(1,self.n-1)
    
        T_DVR = np.zeros((self.n,self.n))  # Matriz de enerfía cinética (fija) [au]

        for i in range(1,self.n-1):
            for j in range(1,self.n-1):
                c = []
                for n in T1:
                    c.append(self.__senno__(i,j,n))
                T_DVR[i,j] = T_c*sum(c)
        
        return T_DVR
    
    #=== Potential Matrix ==============================
    def V_DVR(self, t):
        """
        Input: t [au]
    
        Output: Matriz de potencial en el grid: V_DVR [au]
        """
        V = np.zeros((self.n,self.n))
    
        for i, r in enumerate(self.r_n):
            V[i,i] = self.potential(r,(t))#*(1/1.5936e-3)  # time: fs -> au, energy: Hartree -> kcal/mol
        
        return V
    
    #=== Hamiltonian Matrix ==============================
    def H_DVR(self,t):
        """
        input: tiempo [fs]
    
        output: Matriz Hamiltoniana DVR [au]
        """
        T_DVR = self.KINETIC_DVR()
        t = t*41.34  # fs -> au
        H = T_DVR + self.V_DVR(t)
        return H
    
    
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
    
    
    
    
    
    
    #=== Propagation wavepacket Time Dependent Potential ==========================
    # Find eigenvalues and eigenvectors:
    def eigenN(self,t):
        Eigen_n, U = eig(self.H_DVR(t))
    
        U_inv = np.linalg.inv(U)
    
        D = np.dot(np.dot(U_inv, self.H_DVR(t)),U)
        D = (complex(0,-1))*t*D
        
        return Eigen_n, U, U_inv, D
    
    
    def Psi_VDR_t(self, t, Psi_DVR_inicial):
    
        Eigen_n, U, U_inv, D = self.eigenN(t)
    
        Diag = np.zeros([self.n,self.n], dtype = 'complex_')
        for i in range(self.n):
            Diag[i][i] = np.exp(np.diagonal(D))[i]
    
    
        Psi_DVR_final = np.dot((np.dot(np.dot(U, Diag),U_inv)),Psi_DVR_inicial)
        
        # Normalization
        norm = integrate.simpson((np.abs(Psi_DVR_final)**2), self.r_n)
        
    
        return Psi_DVR_final/np.sqrt(norm)
    
    
    def evolution_wp(self, t, step, gaussiana):
        """
        Función que calcula la evolución del wavepacket inicial a un tiempo t bajo el potencial dado.
        Los intervalos de tiempo son de 1 fs
    
        Input:
        t: tiempo de evolución [fs]
    
        Output:
        wp: Wavepacket evolucionado con el método DVR bajo el potencial V(t)
        """
        
        #-if t == 0:
            #-wp = self.Wavepacket_Init()
            #-with open(os.path.join('./'+self.save_dir+'/Wavepacket', str(t)+'-wave.npy'), 'wb') as f:
                #-np.save(f, wp)
            #-return wp
        if t == 0:
            if gaussiana==True:
                wp = self.gaussi()
                with open(os.path.join('./'+self.save_dir+'/Wavepacket', str(t)+'-wave.npy'), 'wb') as f:
                    np.save(f, wp)
                return wp
            if gaussiana==False:
                k = int(input('Give the number of eigenestates k to generate the wavepacket initial'))
                wp = self.Wavepacket_Init(k)
                with open(os.path.join('./'+self.save_dir+'/Wavepacket', str(t)+'-wave.npy'), 'wb') as f:
                    np.save(f, wp)
                return wp
                
        
        else:
            wp = self.Psi_VDR_t(t, self.evolution_wp(t-step, step, gaussiana))
            with open(os.path.join('./'+self.save_dir+'/Wavepacket', str(t)+'-wave.npy'), 'wb') as f:
                np.save(f, wp)
                
        
            return wp

    
    
    #=== Time Independent Propagation wavepacket ==========================
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
        """
        Función que calcula la evolución del wavepacket inicial a un tiempo t bajo el potencial dado.
        Los intervalos de tiempo son de 1 fs
        * Colocar múltiplos enteros de step
    
        Input:
        t: tiempo de evolución [fs]
    
        Output:
        wp: Wavepacket evolucionado con el método DVR bajo el potencial V(t)
        """
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
