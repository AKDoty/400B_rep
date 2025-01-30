import numpy as np
import astropy.units as u

def ParticleInfo(filename, ptype, pN):
  from ReadFile import Read
  time, N, data = Read(filename)

  index = np.where(data['type'] == ptype)

  x = data['x'][index]*(u.kpc)
  y = data['y'][index]*(u.kpc)
  z = data['z'][index]*(u.kpc)
  #Reading in position coordinates

  vx = data['vx'][index]*(u.km/u.s)
  vy = data['vy'][index]*(u.km/u.s)
  vz = data['vz'][index]*(u.km/u.s)
  #Reading in velocity values

  3Ddistance = np.sqrt((x[pN - 1])**2 + (y[pN - 1])**2 + (z[pN - 1])**2)
  #Calculating 3D distance using distance formula. Using [particle number - 1] b/c index starts at 0

  3Dvelocity = np.sqrt((vx[pN - 1])**2 + (vy[pN - 1])**2 + (vz[pN - 1])**2)
  #Calculates 3D velocity using distance formula. 

  3Ddistance = np.around(3Ddistance, 3)
  3Dvelocity = np.around(3Dvelocity, 3)

  mass = data['m'][index]*u.M_sun*10**10

  return 3Ddistance, 3Dvelocity, mass



  #data has particle type, mass, x, y, z, vx, vy, vz
  #want: mag of distance in kpc
    #mag of velocity in km/s
    #mass in units of Msun
      #round distance and velocity to 3 decimals using np.around(value, 3)
  
