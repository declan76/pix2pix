""" 
Dr Hannah Schunker

18/08/2023 
"""

import numpy as np
import array
from astropy.io import ascii
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import colors
import os.path
import glob, os
import math

# RUN
# > three_fits_processor.py
#

def distance_to_disk_centre(crlt_obs, crln_obs, crlt_ref, crln_ref):
    # input values in degrees

    # convert to radians
    crln_ref=crln_ref*np.pi/180.
    crlt_ref=crlt_ref*np.pi/180.
    crln_obs=crln_obs*np.pi/180.
    crlt_obs=crlt_obs*np.pi/180.
    
    dlon=abs(crln_obs-crln_ref)
    
    distance=math.acos( math.sin(crlt_obs)*math.sin(crlt_ref) + math.cos(crlt_obs)*math.cos(crlt_ref)*math.cos(dlon)  )
    #distance=acos( cos(!dpi/2.-CRLT_OBS)*cos(!dpi/2.-CRLT_REF) + sin(!dpi/2.-CRLT_OBS)*sin(!dpi/2.-CRLT_REF)*cos(dlon)  )
    return distance # in radians    (*180./!dpi for degrees)

def remove_2dplane(array):
    m=array.shape[0] # assume it is square!

    X1, X2 = np.mgrid[:m, :m]
    X = np.hstack(   ( np.reshape(X1, (m*m, 1)) , np.reshape(X2, (m*m, 1)) ) )
    X = np.hstack(   ( np.ones((m*m, 1)) , X ))
    YY = np.reshape(array, (m*m, 1))
    theta = np.dot(np.dot( np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), YY)
    plane = np.reshape(np.dot(X, theta), (m, m));
    sub = array - plane
            
    #plt.imshow(array)
    #plt.show()
    #plt.imshow(sub)
    #plt.show()
    return sub

######  START CODE PROPER  #######


######  HARD CODED STUFF  #######
ODIR='/Users/schunker/OneDrive/RESEARCH/EARS/ChannelCubes/' # output directory
m=256 # pixels, crop size of output maps
MDIR='/Users/schunker/Sol/AVEM/'
mfac=4000. # gauss
IDIR='/Users/schunker/Sol/AVEIC/'
ifac=50000.   # ?
VDIR='/Users/schunker/Sol/HOLOG_EARS/'
vfac=100.   # cm/s

# get list of active regions to loop over
h1='/Users/schunker/OneDrive/RESEARCH/EARS/HARP_output_good.txt'
data = ascii.read(h1) 
#print(data)
# columns are
# ar, harp, t0_TAI, t0_days,  latitude (degrees),   longitude (degrees),   CMD (degrees)
arc = data['col1']
ars = arc.data
h2='/Users/schunker/OneDrive/RESEARCH/EARS/HARP_output_good_set2.txt'
data = ascii.read(h2)
arc = data['col1']
arc = arc.data
ars=np.append(ars,arc)
#print(ars)

# up to 11242
ars=ars[49:]

################################################
# loop over EARS 
for ar in ars:
    ar_str=str(ar)

    # loop over time intervals manually
    for ti in range(-20,29):
        ti_str=str(ti)
       
        # read in average magnetogram        
        mfilename=MDIR+'mps_schunker.avem_ears_AR'+ar_str+'_TI'+ti_str+'_QSUN0.fits'
                   
        # read in corresponding intensity        
        ifilename=IDIR+'mps_schunker.aveic_ears_AR'+ar_str+'_TI'+ti_str+'_QSUN0.fits'   
               
        # read in divergence map
        if ti >= 0:
            ti_str='+'+"{:02d}".format(ti)
        if ti < 0:
            ti_str="{:03d}".format(ti)
            
        vfiles=VDIR+'HOLOG_AR'+ar_str+'/DT_OI_TD3_*TI'+ti_str

        # if all the files exist then go through and make a 3-channel datacube
        if glob.glob(vfiles) and os.path.isfile(ifilename) and os.path.isfile(mfilename):
            print(ifilename)       
            hdu=fits.open(ifilename)
            idata = hdu[1].data
            print(idata.shape)
        
            print(mfilename)
            hdu=fits.open(mfilename)
            mdata = hdu[1].data
            print(mdata.shape)
            mhdr = hdu[1].header
            # divide by cos theta
            theta = distance_to_disk_centre(mhdr['CRLT_OBS'], mhdr['CRLN_OBS'], mhdr['CRLT_REF'], mhdr['CRLN_REF'])
            if theta*180/np.pi > 80:
                print("WARNING: distance to disk is greater than 60 degrees! theta = ",theta*180/np.pi," degrees. Exiting.")
                exit()
            mdata = mdata / math.cos(theta)
            
            junk= glob.glob(vfiles)
            vfilename=junk[0] # assuming there is only one file
            print(vfilename)
            hdu=fits.open(vfilename)
            vdata = hdu[0].data
            print(vdata.shape)
            # strictly should also filter flows
        
            # crop all to 256x256
            idata = idata[int(m-m/2):int(m+m/2),int(m-m/2):int(m+m/2)]
            mdata = mdata[int(m-m/2):int(m+m/2),int(m-m/2):int(m+m/2)]
            vdata = vdata[int(m-m/2):int(m+m/2),int(m-m/2):int(m+m/2)]

            # remove background mean from intensity
            #X1, X2 = np.mgrid[:m, :m]
            #X = np.hstack(   ( np.reshape(X1, (m*m, 1)) , np.reshape(X2, (m*m, 1)) ) )
            #X = np.hstack(   ( np.ones((m*m, 1)) , X ))
            #YY = np.reshape(idata, (m*m, 1))
            #theta = np.dot(np.dot( np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), YY)
            #plane = np.reshape(np.dot(X, theta), (m, m));
            isub = remove_2dplane(idata)
            
            
            print(' IC range: ',np.min(isub),np.max(isub))
            print(' B range: ',np.min(mdata),np.max(mdata))
            print(' V range: ',np.min(vdata),np.max(vdata))
        
            # normalise
            isub = isub / ifac
            mdata = mdata / mfac
            vdata = vdata / vfac

            # show an image and close it straight away
            #plt.imshow(vdata)
            #plt.show(block=False)
            #plt.pause(0.1)
            #plt.close()
            #plt.imshow(isub)
            #plt.show(block=False)
            #plt.pause(0.1)
            #plt.close()
            
            print(' IC norm range: ',np.min(isub),np.max(isub))
            print(' B norm range: ',np.min(mdata),np.max(mdata))
            print(' V norm range: ',np.min(vdata),np.max(vdata))
        
            if np.min(isub) < -1 or np.min(mdata) < -1 or np.min(vdata) < -1:
                print('WARNING: minimum value of something is less than -1. Not good for pix2pix. Truncating.')
                plt.imshow(vdata)
                plt.show(block=False)
                plt.pause(0.1)
                plt.close()
                
                plt.imshow(isub)
                plt.show(block=False)
                plt.pause(0.1)
                plt.close()
                
                zz = np.where(isub < -1)
                isub[zz] = -1
                zz = np.where(mdata < -1)
                mdata[zz] = -1
                zz = np.where(vdata < -1)
                vdata[zz] = -1
                
            if  np.max(isub) > 1 or np.max(mdata) > 1 or np.max(vdata) > 1:
                print('WARNING: maximum value of something is greater than 1. Not good for pix2pix. Truncating.')
                plt.imshow(vdata)
                plt.show(block=False)
                plt.pause(0.1)
                plt.close()

                plt.imshow(isub)
                plt.show(block=False)
                plt.pause(0.1)
                plt.close()
                
                zz = np.where(isub > 1)
                isub[zz] = 1
                zz = np.where(mdata > 1)
                mdata[zz] = 1
                zz = np.where(vdata > 1)
                vdata[zz] = 1
                

            # create 3D datacube and write out
            cube=np.dstack((mdata,vdata,isub))
            print(cube.shape)

            ofile=ODIR+"channels_AR"+ar_str+"_TI"+ti_str+".fits"
            print("Writing "+ofile+"...")
            hdu = fits.PrimaryHDU(data=cube)
            hdr = hdu.header
            hdr['IM1'] = mfilename
            hdr['IM2'] = vfilename
            hdr['IM3'] = ifilename
            hdr['MFAC'] = mfac
            hdr['IFAC'] = ifac
            hdr['VFAC'] = vfac
            
            #    fits.writeto(odir+"slice_tau_1.000_"+var+".fits",cube,overwrite=True)
            hdu.writeto(ofile,overwrite=True)

                
print('Done')