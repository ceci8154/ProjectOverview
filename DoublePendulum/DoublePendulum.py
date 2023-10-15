from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
from mpl_toolkits import mplot3d
from scipy import optimize
plt.close('all')

#Choose yes or no 

#Video simulation of pendulum
Simulation ='yes'
#Plot of energy as function of time.
Energy ='no'
#Plot of theta 2 for two simulations with different time steps
Quality ='no'
#Creates a poincare section holding theta 1 = 0.
Poincare ='no'
#Detailed calculations of Lyapunov exponents for each 
#coordinate and prints the largest one
DetailLyapunov ='no'
#Runs an experiment where some of the exponents are increased stepwise 
#and for each increase a Lyapunov exponent is calculated.
Experiment ='no'

#Parameter conditions are specified.
G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg


#Defining the derivatives of the four coordinates, 
#to be used in the differential equation solver function
#for instance, dydx[0] is the derivative of theta 1. In this case that 
#is omega 1. The more complicated derivatives are from the differential
#equations.
def derivs(state, t):

    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    del_ = state[2] - state[0]
    den1 = (M1 + M2)*L1 - M2*L1*cos(del_)*cos(del_)
    dydx[1] = (M2*L1*state[1]*state[1]*sin(del_)*cos(del_) +
               M2*G*sin(state[2])*cos(del_) +
               M2*L2*state[3]*state[3]*sin(del_) -
               (M1 + M2)*G*sin(state[0]))/den1

    dydx[2] = state[3]

    den2 = (L2/L1)*den1
    dydx[3] = (-M2*L2*state[3]*state[3]*sin(del_)*cos(del_) +
               (M1 + M2)*G*sin(state[0])*cos(del_) -
               (M1 + M2)*L1*state[1]*state[1]*sin(del_) -
               (M1 + M2)*G*sin(state[2]))/den2

    return dydx


#Creates a time array for use throughout the analysis.
dt = 0.05
length = 10
t = np.arange(0.0, length, dt)

#th1 and th2 are the initial angles (degrees)
#w10 and w20 are the initial angular velocities (degrees per second)
th1= 145
w1 = 0
th2= 145
w2 = 0

#The initial state is defined and transformed to radians
state = np.radians([th1, w1, th2, w2])

#The coordinates are simulated using pythons simulation method.
y = integrate.odeint(derivs, state, t)

#Changing back the coordinated to cartesian coordinated for video
#simulation.
x1 = L1*sin(y[:, 0])
y1 = -L1*cos(y[:, 0])
x2 = L2*sin(y[:, 2]) + x1
y2 = -L2*cos(y[:, 2]) + y1

#Simulates the double pendulum.
if Simulation == 'yes':
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), \
    ylim=(-2, 2))
    ax.grid()
    
    line, = ax.plot([], [], 'o-', lw=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text
    
    def animate(i):
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]
        
        line.set_data(thisx, thisy)
        time_text.set_text(time_template % (i*dt))
        return line, time_text
    
    ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)), \
    interval=25, blit=True, init_func=init)
    
    ani.save('double_pendulum.mp4', fps=15)
    plt.show()


if Energy == 'yes':
    #Calculates and plots the energi of the system. The energy is the 
    #same that is defined in the Lagrangian. Since the system is 
    #conservative, the energy should remain constant.
    kinenergi = 1/2*M1*L1**2*(y[:,1])**2+1/2*M2*(L1**2*(y[:,1])**2+ \
    L2**2*(y[:,3])**2+2*L1*L2*y[:,3]*y[:,1]*np.cos(y[:,0]-y[:,2]))
    potenergi = -M1*G*L1*np.cos(y[:,0])-M2*G*L1*np.cos(y[:,0])- \
    M2*G*L2*np.cos(y[:,2])
    energi = kinenergi+potenergi
    
    fig = plt.figure()
    plt.plot(t,energi,c='r',label='Total Energy')
    plt.plot(t,potenergi,c='g',label='Potential Energy')
    plt.plot(t,kinenergi,c='b',label='Kinetic Energy')
    plt.xlabel('Time (s)',fontsize=60)
    plt.ylabel('Energy (J)',fontsize=60)
    #plt.title('Energi of system over time',fontsize=30)
    #plt.legend(fontsize=50)
    plt.tick_params(axis='both', which='major', labelsize=50)
    plt.tick_params(axis='both', which='minor', labelsize=50)


if Quality == 'yes':
    #Plots two simulations with different time steps to check quality.
    #dtny is the new time step to be compared with the one defined 
    #earlier.
    dtny = 0.005
    tny = np.arange(0.0,length,dtny)
    #yny is the new coordinates simulated with the new time step.
    yny = integrate.odeint(derivs, state, tny)
    #Now theta 2 is plottet for both simulations to see if they diverge.
    fig = plt.figure()
    plt.plot(t,y[:,2]/np.pi,c='r',label='Simulation with time step=%f'%(dt))
    plt.plot(tny,yny[:,2]/np.pi,c='g',\
    label='Simulation with time step=%f'%(dtny))
    plt.xlabel('Time (s)',fontsize=60)
    plt.ylabel(r'$\theta_2 (\pi)$',fontsize=60)
    #plt.title('',fontsize=30)
    #plt.legend(fontsize=40)
    plt.tick_params(axis='both', which='major', labelsize=50)
    plt.tick_params(axis='both', which='minor', labelsize=50)


#Creating Poincare section for theta1 = 0
if Poincare == 'yes':
    zero = np.array([])
    #We sort through theta 1 to find the points in which it passes 0 
    #degrees.
    for i in range(len(y[:,0])-1):
        if np.sin(y[i-1,0])<0 and np.sin(y[i,0])>0 and np.cos(y[i,0])>0 \
        or np.sin(y[i-1,0])>0 and np.sin(y[i,0])<0 and np.cos(y[i,0])>0:
            zero = np.append(zero,int(i-1))
    #zero is the array that contains the place in the array right before 
    #theta 1 crosses 0. 
    zero = zero.astype(int)
    tz = np.array([])
    ynew = np.zeros((len(zero),4))
    #We now use the points before theta 1 crosses 0 degrees 
    #to use linear equations to estimate what the coordinates were 
    #at the exact time theta 1 crossed 0.
    for i,k in zip(zero,range((len(zero)))):
        a = (y[i+1,0]-y[i,0])/(dt)
        if a > 0:
            mody = (y[i,0]%(2*np.pi))
            tzero = (2*np.pi-mody)/a    
        else:
            mody = (y[i,0]%(2*np.pi))
            tzero = (-mody)/a
        tz = np.append(tz,tzero)
        for j in range(1,4,1):
            at = (y[i+1,j]-y[i,j])/(dt)
            newvalue = y[i,j]+at*tzero
            ynew[k,j] = newvalue
    #We now plot the points in a 3d diagram to see 
    #if there are any obvious symmetries

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter3D(ynew[:,2],ynew[:,3], ynew[:,1],c='r',s=100)
    #plt.title('',fontsize=30)
    ax.set_zlabel(r'$\omega_1 (rad/s) $',labelpad=80,fontsize=60);
    ax.set_xlabel(r'$\theta_2 (rad) $',labelpad=80,fontsize=60)
    ax.set_ylabel(r'$\omega_2 (rad/s)$',labelpad=80,fontsize=60)
    #plt.tick_params(axis='both', which='major', labelsize=50)
    #plt.tick_params(axis='both', which='minor', labelsize=50)
    plt.tick_params(axis='both', which='major', labelsize=50)
    plt.tick_params(axis='both', which='minor', labelsize=50)


#Calculates the Lyapunov exponent for the pendulum defined in the 
#beginning.
if DetailLyapunov == 'yes':
    #Defines the small changes in starting conditions
    dth1 = 0.001
    dw1 = 0.001
    dth2 = 0.001
    dw2 = 0.001
    
    #Creates a new simulation with the initial conditions plus the 
    #small change.
    nystate = state + np.radians([dth1, dw1, dth2, dw2])
    nyy = integrate.odeint(derivs, nystate, t)
    
    #Calculating difference between each coordinate at all times.
    dy = nyy-y
    
    #Title used to create plots of exponents later.
    title = [r'$\theta_1 $',r'$\omega_1 $',r'$\theta_2 $',r'$\omega_1 $']
    
    #Array for the Lyapunov exponents for each coordinate.
    lya = np.array([])
    #Loop that finds the Lyapunov exponents for each coordinate.
    for i in range(4):
        #Finds the norm between coordinates
        norm = np.sqrt(dy[:,i]**2)
        #Plotting the norm.
        fig = plt.figure()
        plt.plot(t,norm,c='r')
        plt.xlabel('Time (s)',fontsize=60)
        plt.ylabel('Norm of difference in %s (rad)'%(title[i]), \
        fontsize=60)
        plt.tick_params(axis='both', which='major', labelsize=50)
        plt.tick_params(axis='both', which='minor', labelsize=50)
        
        #Fitting a linear regression to log(dy/dy0) 
        #to find lyaponow exponent as the slope.
        #Calculates log(dy/dy0) 
        logt1 = np.log(norm/norm[0])
        
        #Plots the log(dy/dy0) against time
        fig = plt.figure()
        plt.plot(t,logt1[:],c='r',label='Data from simulation')
        plt.xlabel('Time (s)',fontsize=60)
        plt.ylabel('Log of norm of difference in %s' %(title[i]), \
        fontsize=60)
        #plt.title('' ,fontsize = 30)
        plt.tick_params(axis='both', which='major', labelsize=50)
        plt.tick_params(axis='both', which='minor', labelsize=50)
        
        #Fits a double linear regression to the data to find the slope
        #which is the Lyapunov exponent.
        def piecewise_linear(x, x0, y0, k1, k2):
            return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0,\
            lambda x:k2*x + y0-k2*x0])
        
        val , cov = optimize.curve_fit(piecewise_linear, t, logt1)
        #Ensures that the initial slope is not picked it is does not 
        #last at least 2 seconds.
        if val[0] > 2:
            lyacoef = val[2]
        else:
            lyacoef = val[3]
        
        plt.plot(t,piecewise_linear(t,*val),color='g',\
        label='Fit with Lyaponov exponent of %f' %(lyacoef))
        plt.legend(fontsize='60')
        lya = np.append(lya,lyacoef)
        
    print(max(lya))


#Defines Lyapunov function that basically takes the initial state and 
#a small change and calculates the Lyapunov exponent exactly like in 
#the detailed case
def exponent(initial, change):
    y = integrate.odeint(derivs, initial, t)
    nyy = integrate.odeint(derivs, initial+change, t)
    dy = nyy-y
    lya = np.array([])
   
    for i in range(4):
        norm = np.sqrt(dy[:,i]**2)
    
        #Fitting a linear regression to log(dy/dy0) 
        #to find lyaponow exponent as the slope.
        logt1 = np.log(norm/norm[0])
        
        def piecewise_linear(x, x0, y0, k1, k2):
            return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0,\
            lambda x:k2*x + y0-k2*x0])
        
        val , cov = optimize.curve_fit(piecewise_linear, t, logt1)
        
        if val[0] > 2:
            lyacoef = val[2]
        else:
            lyacoef = val[3]
        lya = np.append(lya,lyacoef)
        
    return(max(lya))


#Actual experiment where Lyapunov exponents are calculated for a variety 
#of starting conditions
if Experiment == 'yes':    
    #Defines the first case
    th1 = 0.0
    w1 = 0.0
    th2 = 0.0
    w2 = 0.0
    
    #Defines the small change in coordinates.
    dth1 = 0.001
    dw1 = 0.001
    dth2 = 0.001
    dw2 = 0.001
    
    #Groups them in arrays.
    startstate = np.radians([th1, w1, th2, w2])
    delta = np.radians([dth1, dw1, dth2, dw2])
    
    #How many steps should be investigated
    ac = 180
    #Variable number 0-3. for instance 1 means that the variable will 
    #be increased by one for each step. -1/2 means that the variable is 
    #changed by minus a half for each step.
    v1 = 1
    v2 = 0
    v3 = 1
    v4 = 0
    
    #Creates array for Lyapunov exponents.
    lyafinal = ([])
    
    #Runs the experiment.
    for i in range(ac):
        state = startstate + np.radians([i*v1, i*v2, i*v3, i*v4])
        Lyapunov = exponent(state,delta)
        Lyafinal = np.append(Lyafinal,Lyapunov)
    #Creates a graph over Lyapunov exponent against step number.
    steps = np.linspace(1,ac,ac)
    plt.figure()
    plt.scatter(steps,lyafinal)
    plt.xlabel('Step #',fontsize=60)
    plt.ylabel('Lyaponov exponent',fontsize=60)
    plt.tick_params(axis='both', which='major', labelsize=50)
    plt.tick_params(axis='both', which='minor', labelsize=50)
