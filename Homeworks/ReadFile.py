import numpy as np
import astropy.units as u
#Importing necessary packages

def Read(filename):
#Function that reads in a file
  file = open(filename, 'r')
  #Opening file
  
  line1 = file.readline()
  label, value = line1.split()
  time = float(value)*u.Myr
  #Reading in first line
  
  line2 = file.readline()
  label, value = line2.split()
  N = float(value)
    #Should be in moles?
  #Reading in second line

  file.close()
  data = np.genfromtxt(filename, dtype = None, names = True, skip_header = 3)
  return time, N
  #Full data array?
