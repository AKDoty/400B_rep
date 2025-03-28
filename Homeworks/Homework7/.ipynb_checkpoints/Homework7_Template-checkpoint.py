
# # Homework 7 Template
# 
# Rixin Li & G . Besla
# 
# Make edits where instructed - look for "****", which indicates where you need to 
# add code. 




# import necessary modules
# numpy provides powerful multi-dimensional arrays to hold and manipulate data
import numpy as np
# matplotlib provides powerful functions for plotting figures
import matplotlib.pyplot as plt
# astropy provides unit system and constants for astronomical calculations
import astropy.units as u
import astropy.constants as const
# import Latex module so we can display the results with symbols
from IPython.display import Latex

# **** import CenterOfMass to determine the COM pos/vel of M33
from CenterOfMass2 import CenterOfMass

# **** import the GalaxyMass to determine the mass of M31 for each component
from GalaxyMass import ComponentMass


class M33AnalyticOrbit:
    """ Calculate the analytical orbit of M33 around M31 """
    
    def __init__(self, filename): # **** add inputs
        """
        Function that takes a filename and stores the integrated orbit to this file. 
        PARAMETERS
        Inputs: filename -> 'str', name of a file in which we will store integrated orbit.
        """
        
        ### get the gravitational constant (the value is 4.498502151575286e-06)
        self.G = const.G.to(u.kpc**3/u.Msun/u.Gyr**2).value
        
        ### **** store the output file name
        self.filename = filename
        
        ### get the current pos/vel of M33 
        # **** create an instance of the  CenterOfMass class for M33 
        M33_COM = CenterOfMass("M33_000.txt", 2)
        
        # **** store the position VECTOR of the M33 COM (.value to get rid of units)
        posM33 = M33_COM.COM_P(.1, 4)
        
        # **** store the velocity VECTOR of the M33 COM (.value to get rid of units)
        velM33 = M33_COM.COM_V(posM33[0], posM33[1], posM33[2])
            #getting 3 position components for velocity
        
        ### get the current pos/vel of M31 
        # **** create an instance of the  CenterOfMass class for M31 
        M31_COM = CenterOfMass("M31_000.txt", 2)
        
        # **** store the position VECTOR of the M31 COM (.value to get rid of units)
        posM31 = M31_COM.COM_P(.1, 2)
        
        # **** store the velocity VECTOR of the M31 COM (.value to get rid of units)
        velM31 = M31_COM.COM_V(posM31[0], posM31[1], posM31[2])
        
        ### store the DIFFERENCE between the vectors posM33 - posM31
        # **** create two VECTORs self.r0 and self.v0 and have them be the
        # relative position and velocity VECTORS of M33

        self.r0 = (posM33 - posM31).value
        self.v0 = (velM33 - velM31).value
        
        ### get the mass of each component in M31 
        
        ### disk
        # **** self.rdisk = scale length (no units)
        self.rdisk = 5 #in kpc

        # **** self.Mdisk set with ComponentMass function. Remember to *1e12 to get the right units. Use the right ptype
        self.Mdisk = ComponentMass("M31_000.txt", 2)*1e12
        
        ### bulge
        # **** self.rbulge = set scale length (no units)
        self.rbulge = 1 #in kpc

        # **** self.Mbulge  set with ComponentMass function. Remember to *1e12 to get the right units Use the right ptype
        self.Mbulge = ComponentMass("M31_000.txt", 3)*1e12
        
        # Halo
        # **** self.rhalo = set scale length from HW5 (no units)
        self.rhalo = 62.0 #in kpc, based on answer in hw5 solution
        
        # **** self.Mhalo set with ComponentMass function. Remember to *1e12 to get the right units. Use the right ptype
        self.Mhalo = ComponentMass("M31_000.txt", 1)*1e12
    
    
    def HernquistAccel(self, M, r_a, r): # it is easiest if you take as an input the position VECTOR 
        """
        Function that computes gravitational acceleration vectors from 3 components of the M31 galaxy,
        based on what is induced by a Hernquist profile. 
        PARAMETERS
        Inputs: M -> 'float', total halo or bulge mass (Msun)
                r_a -> 'float', corresponding length scale (kpc)
                r -> 'vector', relative position vector of relevant object (kpc)
        Outputs: Hern -> 'vector', acceleration vector from a Hernquist potential
        """
        
        ### **** Store the magnitude of the position vector
        rmag = np.linalg.norm(r)
        
        ### *** Store the Acceleration
        Hern =  -r*self.G*M/(rmag *(r_a + rmag)**2)
        #follow the formula in the HW instructions
        # NOTE: we want an acceleration VECTOR so you need to make sure that in the Hernquist equation you 
        # use  -G*M/(rmag *(ra + rmag)**2) * r --> where the last r is a VECTOR 
        
        return Hern
    
    
    
    def MiyamotoNagaiAccel(self, M, r_d, r):# it is easiest if you take as an input a position VECTOR  r 
        """
        Function that returns the acceleration vector of a disk from a Miyamoto-Nagai profile. 
        PARAMETERS
        Inputs: M -> 'float', total disk mass (Msun)
                r_a -> 'float', corresponding length scale (kpc)
                r -> 'vector', relative position vector of relevant object (kpc)
        Outputs: accel -> 'vector', acceleration vector from a Miyamoto-Nagai potential
        """

        ### Acceleration **** follow the formula in the HW instructions
        # AGAIN note that we want a VECTOR to be returned  (see Hernquist instructions)
        # this can be tricky given that the z component is different than in the x or y directions. 
        # we can deal with this by multiplying the whole thing by an extra array that accounts for the 
        # differences in the z direction:
        #  multiply the whle thing by :   np.array([1,1,ZSTUFF]) 
        # where ZSTUFF are the terms associated with the z direction

        x, y, z = r
        r_d = self.rdisk
        z_d = self.rdisk/5
        #Defining vars we need per instructions

        R = np.sqrt(x**2 + y**2)
        B = r_d + np.sqrt(z**2 + z_d**2)
        #defining R & B vars

        ZSTUFF = B/np.sqrt(z**2 + z_d**2) #terms associated with z direction
        accel = -r*np.array([1,1,ZSTUFF])*self.G*M/((R**2 + B**2)**1.5)
       
        return accel
        # the np.array allows for a different value for the z component of the acceleration
     
    
    def M31Accel(self, r): # input should include the position vector, r
        """
        Function that sums all acceleration vectors of M31 from each galaxy component. 
        PARAMETERS
        Inputs: r -> 'vector', 3D position vector of M33
        Outputs: accel_tot -> 'vector', 3D vector of total acceleration of M33 because of M31
        """

        ### Call the previous functions for the halo, bulge and disk
        # **** these functions will take as inputs variable we defined in the initialization of the class like 
        # self.rdisk etc. 
        
        accel_halo = self.HernquistAccel(self.Mhalo, self.rhalo, r)
        accel_bulge = self.HernquistAccel(self.Mbulge, self.rbulge, r)
        accel_disk = self.MiyamotoNagaiAccel(self.Mdisk, self.rdisk, r)

        sum = accel_halo + accel_bulge + accel_disk
        
            # return the SUM of the output of the acceleration functions - this will return a VECTOR 
        return sum
    
    
    
    def LeapFrog(self, dt, r, v): # take as input r and v, which are VECTORS. Assume it is ONE vector at a time
        """
        Function that sets up an integrator and executes it by determining sequential integration steps. 
        PARAMETERS
        Inputs: dt -> 'float', time interval for integration
                r -> 'vector', starting position vector for M33 COM position relative to M31
                v -> 'vector', starting velocity vector for M33 relative to M31
        Outputs: rnew -> 'vector', new integrated position vector
                 vnew -> 'vector', new integrated velocity vector 
        """
        
        # predict the position at the next half timestep
        rhalf = r + v*(dt/2)
        
        # predict the final velocity at the next timestep using the acceleration field at the rhalf position 
        vnew = v + (self.M31Accel(rhalf))*dt
        
        # predict the final position using the average of the current velocity and the final velocity
        # this accounts for the fact that we don't know how the speed changes from the current timestep to the 
        # next, so we approximate it using the average expected speed over the time interval dt. 
        rnew = rhalf + vnew*(dt/2)
        
        return rnew, vnew # **** return the new position and velcoity vectors
    
    
    def OrbitIntegration(self, t0, dt, tmax):
        """
        Function that loops over LeapFrog integrator and solves equations of motion, used to compute
        future orbit of M33. 
        PARAMETERS
        Inputs: t0 -> 'float', starting time
                dt -> 'float', time interval
                tmax -> 'float', final time
        """

        # initialize the time to the input starting time
        #t = np.arange(t0, tmax, dt)
        t = t0
        r = self.r0
        v = self.v0
        
        # initialize an empty array of size :  rows int(tmax/dt)+2  , columns 7
        orbit = np.zeros((int(tmax/dt)+2, 7))
        
        # initialize the first row of the orbit
        orbit[0] = t0, *tuple(self.r0), *tuple(self.v0)
        # this above is equivalent to 
        # orbit[0] = t0, self.r0[0], self.r0[1], self.r0[2], self.v0[0], self.v0[1], self.v0[2]
        
        
        # initialize a counter for the orbit.  
        i = 1 # since we already set the 0th values, we start the counter at 1
        
        # start the integration (advancing in time steps and computing LeapFrog at each step)
        while (t <= tmax):  # as long as t has not exceeded the maximal time 
            
            # **** advance the time by one timestep, dt
            t = t + dt
            
            # ***** advance the position and velocity using the LeapFrog scheme
            # remember that LeapFrog returns a position vector and a velocity vector  
            # as an example, if a function returns three vectors you would call the function and store 
            # the variable like:     a,b,c = function(input)
            r, v = self.LeapFrog(dt, r, v)
            orbit[i] = t, *tuple(r), *tuple(v)
            
            # **** update counter i , where i is keeping track of the number of rows (i.e. the number of time steps)
            i += 1
        
        
        # write the data to a file
        np.savetxt(self.filename, orbit, fmt = "%11.3f"*7, comments='#', 
                   header="{:>10s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}"\
                   .format('t', 'x', 'y', 'z', 'vx', 'vy', 'vz'))
        
        # there is no return function



