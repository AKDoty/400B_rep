import numpy as np
import astropy.units as u
#Importing necessary packages

def Read(filename):
#Function that reads in a file
#Input: A file from Milky Way data
#Output: Returning time, total particle number, and 
    file = open(filename, 'r')
    #Opening file
    
    line1 = file.readline()
    label, value = line1.split()
    time = float(value)*u.Myr
    #Reading in first line
        #time in Myr
    
    line2 = file.readline()
    label, value = line2.split()
    N = float(value)
    #Reading in second line
        #total number of particles, unitless

    file.close()
    #closing file
    
    data = np.genfromtxt(filename, dtype = None, names = True, skip_header = 3)
    #storing remainder of data in file into an array
    
    return time, N, data
    #returning time, total number of particles, and data array
