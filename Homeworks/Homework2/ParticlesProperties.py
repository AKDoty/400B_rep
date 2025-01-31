import numpy as np
import astropy.units as u
#importing necessary packages

def ParticleInfo(filename, ptype, pN):
    '''
    This function takes a MW file and returns information about a specific
        particle's 3D distance, 3D velocity, and mass.
    Inputs: filename = (file) MW.txt file from nimoy database
                Here, only using MW_000.txt
            ptype = (number) particle's 'type', labeled 1-3 corresponding to 
                different objects (haloes, disks, etc)
            pN = (number) particle's number, line in MW.txt file
    Outputs: distance = (astropy quantity) magnitude of 3D distance (kpc)
            velocity = (astropy quantity) magnitude of 3D velocity (km/s)
            mass = (astropy quantity) mass (M_sun)
    '''
    from ReadFile import Read
    #Fetching Read function from ReadFile code
    time, N, data = Read(filename)
    #Getting the returns from Read function

    index = np.where(data['type'] == ptype)
    #Creating index representing 'type', input will be 1-3

    x = data['x'][index]*(u.kpc) #x position
    y = data['y'][index]*(u.kpc) #y position
    z = data['z'][index]*(u.kpc) #z position
    #Reading in position coordinates, in kpc

    vx = data['vx'][index]*(u.km/u.s) #velocity in x
    vy = data['vy'][index]*(u.km/u.s) #velocity in y
    vz = data['vz'][index]*(u.km/u.s) #velocity in z
    #Reading in velocity values, in km/s

    distance = np.sqrt((x[pN - 1])**2 + (y[pN - 1])**2 + (z[pN - 1])**2)
    #Calculating 3D distance using distance formula. 
    #Using [particle number - 1] b/c index starts at 0

    velocity = np.sqrt((vx[pN - 1])**2 + (vy[pN - 1])**2 + (vz[pN - 1])**2)
    #Calculates 3D velocity using distance formula. 
    #Using [particle number - 1] b/c index starts at 0

    distance = np.around(distance, 3)
    velocity = np.around(velocity, 3)
    #Rounding outputs to 3 decimal places

    mass = data['m'][index]*u.M_sun*10**10
    mass = mass[pN-1]
    #Converting generic mass to be in M_sun, then indexing it

    return distance, velocity, mass
    #Retuns desired quantities

distance, velocity, mass = ParticleInfo("MW_000.txt", 2, 100)
print(ParticleInfo("MW_000.txt", 2, 100))
#calling test cases and printing result

distLY = distance.to(u.lyr)
#Converts 3D distance to lightyears
print(np.round(distLY, 3))
#Prints distLY and rounds to 3 decimal places