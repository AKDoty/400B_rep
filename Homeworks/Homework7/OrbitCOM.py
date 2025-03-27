
# import modules
import numpy as np
import astropy.units as u
from astropy.constants import G

# import plotting modules
import matplotlib.pyplot as plt
import matplotlib

# my modules
from ReadFile import Read
# Step 1: modify CenterOfMass so that COM_P now takes a parameter specifying 
# by how much to decrease RMAX instead of a factor of 2
from CenterOfMass2 import CenterOfMass


def OrbitCOM(galaxy, start, end, n):
    """function that loops over all the desired snapshots to compute the COM pos and vel as a function of time.
    inputs: galaxy -> (string) name of galaxy
            start -> (int) number of first snapshot to be read in
            end -> (int) number of last snapshot to be read in
            n -> (int) interval over which COM is returned
    outputs:
            fileout -> (file) saves time, COM position, and velocity vectors of 
                a given galaxy in each snapshot
    """
    
    # compose the filename for output
    fileout = "Orbit_" + galaxy + ".txt"
    
    #  set tolerance and VolDec for calculating COM_P in CenterOfMass
    delta = .1
    volDec = 2
    
    # for M33 that is stripped more, use different values for VolDec
    if galaxy == "M33":
        volDec = 4
    
    # generate the snapshot id sequence 
    snap_ids = np.arange(start, end + 1, n)
        #.arange is exclusive of final value, so we do end + 1

    if snap_ids.size == 0:
        print("No more snap_ids")
        return
        
    # initialize the array for orbital info: t, x, y, z, vx, vy, vz of COM
    orbit = np.zeros([(snap_ids.size), 7])
    
    # a for loop 
    # loop over files
    for i, snap_id in enumerate(snap_ids):
        #Defining filename -> :03 defines max length we want the 
        #name to be. "If input is not the same size, fill it with
        #0s until it's the same size"
        filename = f"VLowRes/{galaxy}/{galaxy}_{snap_id:03}.txt"

        # Initialize an instance of CenterOfMass class, using disk particles
        COM = CenterOfMass(filename, 2)
            #disk is ptype = 2

        #Storing time in Gyr
        orbit[i, 0] = (COM.time).to_value()/1000

        #Position
        posCOM = COM.COM_P(.1, volDec)
            #inputting delta and volDEC
        x, y, z = posCOM[0], posCOM[1], posCOM[2]
        orbit[i, 1] = x.value
        orbit[i, 2] = y.value
        orbit[i, 3] = z.value
        #Velocity
        velCOM = COM.COM_V(x, y, z)
        vx, vy, vz = velCOM[0], velCOM[1], velCOM[2]
        orbit[i, 4] = vx.value
        orbit[i, 5] = vy.value
        orbit[i, 6] = vz.value

        # Store the COM pos and vel. Remember that now COM_P required VolDec
        # store the time, pos, vel in ith element of the orbit array,  without units (.value) 
        # note that you can store 
        # a[i] = var1, *tuple(array1)
        # print snap_id to see the progress
        print(snap_id)
        
    # write the data to a file
    # we do this because we don't want to have to repeat this process 
    # this code should only have to be called once per galaxy.
    np.savetxt(fileout, orbit, fmt = "%11.3f"*7, comments='#',
               header="{:>10s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}"\
                      .format('t', 'x', 'y', 'z', 'vx', 'vy', 'vz'))



# Recover the orbits and generate the COM files for each galaxy
# read in 800 snapshots in intervals of n=5
# Note: This might take a little while - test your code with a smaller number of snapshots first! 

M31orbit = OrbitCOM('M31', 0, 800, 5)
M33orbit = OrbitCOM('M33', 0, 800, 5)
MWorbit = OrbitCOM('MW', 0, 800, 5)


# Read in the data files for the orbits of each galaxy that you just created
# headers:  t, x, y, z, vx, vy, vz
# using np.genfromtxt
M31outfile = 'Orbit_M31.txt'
M33outfile = 'Orbit_M33.txt'
MWoutfile = 'Orbit_MW.txt'

M31data = np.genfromtxt(M31outfile, dtype = None, names = True, skip_header = 0, usecols = (0,1,2,3,4,5,6))
M33data = np.genfromtxt(M33outfile, dtype = None, names = True, skip_header=0, usecols = (0,1,2,3,4,5,6))
MWdata = np.genfromtxt(MWoutfile, dtype = None, names = True, skip_header = 0, usecols = (0,1,2,3,4,5,6))



# function to compute the magnitude of the difference between two vectors 
# You can use this function to return both the relative position and relative velocity for two 
# galaxies over the entire orbit  

def vecdifference(pos1, pos2, vel1, vel2):
    '''
    Function to compute the magnitude of the difference between two vectors
    Inputs: pos1, pos2 -> (arrays) position vectors of 2 different galaxies
            vel1, vel2 -> (arrays) velocity vectors of 2 different galaxies
    Outputs: pmag, vmag -> (floats) magnitudes of position, velocity differences
                                    between 2 galaxies
    '''
    px = pos2[0] - pos1[0]
    py = pos2[1] - pos1[1]
    pz = pos2[2] - pos1[2]
    pmag = np.sqrt(px**2 + py**2 + pz**2) #magnitude of relative position difference
    
    vx = vel2[0] - vel1[0]
    vy = vel2[1] - vel1[1]
    vz = vel2[2] - vel1[2]
    vmag = np.sqrt(vx**2 + vy**2 + vz**2) #magnitude of relative velocity difference

    return pmag, vmag


# Determine the magnitude of the relative position and velocities 

# of MW and M31

# of M33 and M31

def vecdifference(pos1, pos2, vel1, vel2):
    '''
    Function to compute the magnitude of the difference between two vectors
    Inputs: pos1, pos2 -> (arrays) position vectors of 2 different galaxies
            vel1, vel2 -> (arrays) velocity vectors of 2 different galaxies
    Outputs: pmag, vmag -> (floats) magnitudes of position, velocity differences
                                    between 2 galaxies
    '''
    px = pos2[0] - pos1[0]
    py = pos2[1] - pos1[1]
    pz = pos2[2] - pos1[2]
    pmag = np.sqrt(px**2 + py**2 + pz**2) #magnitude of relative position difference
    
    vx = vel2[0] - vel1[0]
    vy = vel2[1] - vel1[1]
    vz = vel2[2] - vel1[2]
    vmag = np.sqrt(vx**2 + vy**2 + vz**2) #magnitude of relative velocity difference

    return pmag, vmag


# Plot the Orbit of the galaxies 
#################################
time = np.genfromtxt(M33outfile, dtype = None, names = True, skip_header = 0, usecols = 0)
#pulling time (column 0) from the output file
    #Using M31, but they'll all cover same time range so could be any outfile

#MW and M31
fig = plt.figure(figsize = (10,10))  # sets the scale of the figure
ax = plt.subplot(111)
plt.plot(time, sepW_31, color = 'green', linewidth = 5, label = 'M31 - MW')
plt.xlabel('Time (Gyr)', fontsize=22)
plt.ylabel('Separation (kpc)', fontsize=22)
    #adjust tick label font size
label_size = 22
matplotlib.rcParams['xtick.labelsize'] = label_size 
matplotlib.rcParams['ytick.labelsize'] = label_size
legend = ax.legend(loc = 'upper right', fontsize = 'x-large')
plt.show()

#M33 and M31
fig = plt.figure(figsize = (10,10))  # sets the scale of the figure
ax = plt.subplot(111)
plt.plot(time, sep33_31, color = 'purple', linewidth = 5, label = 'M31 - M33')
plt.xlabel('Time (Gyr)', fontsize=22)
plt.ylabel('Separation (kpc)', fontsize=22)
    #adjust tick label font size
label_size = 22
matplotlib.rcParams['xtick.labelsize'] = label_size 
matplotlib.rcParams['ytick.labelsize'] = label_size
legend = ax.legend(loc = 'upper right', fontsize = 'x-large')
plt.show()



# Plot the orbital velocities of the galaxies 
#################################
#MW and M31
fig = plt.figure(figsize = (10,10))  # sets the scale of the figure
ax = plt.subplot(111)
plt.plot(time, velW_31, color = 'green', linewidth = 5, label = 'M31 - MW')
plt.xlabel('Time (Gyr)', fontsize=22)
plt.ylabel('Velocity (kpc/s)', fontsize=22)
    #adjust tick label font size
label_size = 22
matplotlib.rcParams['xtick.labelsize'] = label_size 
matplotlib.rcParams['ytick.labelsize'] = label_size
legend = ax.legend(loc = 'upper right', fontsize = 'x-large')
plt.show()

#M33 and M31
fig = plt.figure(figsize = (10,10))  # sets the scale of the figure
ax = plt.subplot(111)
plt.plot(time, vel33_31, color = 'purple', linewidth = 5, label = 'M31 - M33')
plt.xlabel('Time (Gyr)', fontsize=22)
plt.ylabel('Velocity (kpc/s)', fontsize=22)
    #adjust tick label font size
label_size = 22
matplotlib.rcParams['xtick.labelsize'] = label_size 
matplotlib.rcParams['ytick.labelsize'] = label_size
legend = ax.legend(loc = 'upper right', fontsize = 'x-large')
plt.show()
