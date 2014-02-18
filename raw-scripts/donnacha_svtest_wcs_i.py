import numpy as np
import subprocess
import sys
import os
import matplotlib.pyplot as plt
import sys
import scipy.ndimage.filters as filt
import math
import pyfits
from matplotlib.colors import LogNorm

# read in epoch file: EXPOSURE_E1, EXPOSURE_E2, WCS_G1, WCS_G2, EXPOSURE_INDEX
x  = pyfits.open('wcs_im3shape_v3_epoch_i_000001.fits')
g1_1 = x[1].data.field('exposure_e1')
g2_1 = x[1].data.field('exposure_e2')
wcs_g1_1 = x[1].data.field('wcs_g1')
wcs_g2_1 = x[1].data.field('wcs_g2')
exposure_index_1 = x[1].data.field('exposure_index')

N1 = len(g1_1)
g1=g1_1
g2=g2_1
wcs_g1=wcs_g1_1
wcs_g2=wcs_g2_1
exposure_index=exposure_index_1

x  = pyfits.open('wcs_im3shape_v3_epoch_i_000002.fits')
g1_2 = x[1].data.field('exposure_e1')
g2_2 = x[1].data.field('exposure_e2')
wcs_g1_2 = x[1].data.field('wcs_g1')
wcs_g2_2 = x[1].data.field('wcs_g2')
exposure_index_2 = x[1].data.field('exposure_index')

N2 = len(g1_2)
g1=np.append(g1,g1_2)
g2=np.append(g2,g2_2)
wcs_g1=np.append(wcs_g1,wcs_g1_2)
wcs_g2=np.append(wcs_g2,wcs_g2_2)
exposure_index=np.append(exposure_index,exposure_index_2)

x  = pyfits.open('wcs_im3shape_v3_epoch_i_000003.fits')
g1_3 = x[1].data.field('exposure_e1')
g2_3 = x[1].data.field('exposure_e2')
wcs_g1_3 = x[1].data.field('wcs_g1')
wcs_g2_3 = x[1].data.field('wcs_g2')
exposure_index_3 = x[1].data.field('exposure_index')

N3 = len(g1_3)
g1=np.append(g1,g1_3)
g2=np.append(g2,g2_3)
wcs_g1=np.append(wcs_g1,wcs_g1_3)
wcs_g2=np.append(wcs_g2,wcs_g2_3)
exposure_index=np.append(exposure_index,exposure_index_3)

x  = pyfits.open('wcs_im3shape_v3_epoch_i_000004.fits')
g1_4 = x[1].data.field('exposure_e1')
g2_4 = x[1].data.field('exposure_e2')
wcs_g1_4 = x[1].data.field('wcs_g1')
wcs_g2_4 = x[1].data.field('wcs_g2')
exposure_index_4 = x[1].data.field('exposure_index')

N4 = len(g1_4)
g1=np.append(g1,g1_4)
g2=np.append(g2,g2_4)
wcs_g1=np.append(wcs_g1,wcs_g1_4)
wcs_g2=np.append(wcs_g2,wcs_g2_4)
exposure_index=np.append(exposure_index,exposure_index_4)

x  = pyfits.open('wcs_im3shape_v3_epoch_i_000005.fits')
g1_5 = x[1].data.field('exposure_e1')
g2_5 = x[1].data.field('exposure_e2')
wcs_g1_5 = x[1].data.field('wcs_g1')
wcs_g2_5 = x[1].data.field('wcs_g2')
exposure_index_5 = x[1].data.field('exposure_index')

N5 = len(g1_5)
g1=np.append(g1,g1_5)
g2=np.append(g2,g2_5)
wcs_g1=np.append(wcs_g1,wcs_g1_5)
wcs_g2=np.append(wcs_g2,wcs_g2_5)
exposure_index=np.append(exposure_index,exposure_index_5)

Ntot=N1+N2+N3+N4+N5

# average over the exposures of every object to get an average g1 & g2 for each object
# Do with for both exposure ellipticities and wcs ellipticities

g1s_averaged=[]
g2s_averaged=[]
g1s_wcs_averaged=[]
g2s_wcs_averaged=[]
count=0
# start with no shear
g1_running=0.0
g2_running=0.0
g1_wcs_running=0.0
g2_wcs_running=0.0
for i in range(1,Ntot):

	if (np.mod(100*(i+1)/float(Ntot),1)<1e-6): print str(int(100*(i+1)/float(Ntot)))+' percent complete'

	g1_current = g1
	g2_current = g2
	g1_wcs_current = wcs_g1
	g2_wcs_current = wcs_g2
	index_current = exposure_index[i]
	index_previous = exposure_index[i-1]


	if (index_current==1): # index is one so we're on a new object
		# first average the running total and store the last object
		g1_av = g1_running/index_previous
		g2_av = g2_running/index_previous
		g1_wcs_av = g1_wcs_running/index_previous
		g2_wcs_av = g2_wcs_running/index_previous
		g1s_averaged.append(g1_av)
		g2s_averaged.append(g2_av)
		g1s_wcs_averaged.append(g1_wcs_av)
		g2s_wcs_averaged.append(g2_wcs_av)

		# now reset the running total
		g1_running = g1[i]
		g2_running = g2[i]
		g1_wcs_running = wcs_g1[i]
		g2_wcs_running = wcs_g2[i]

		# add a new object to the count
		count+=1

	if (index_current>1): # index is >1 so we're on another exposure of the same object
		# add to the running total
		g1_running = g1_running + g1[i]
		g2_running = g2_running + g2[i]
		g1_wcs_running = g1_wcs_running + wcs_g1[i]
		g2_wcs_running = g2_wcs_running + wcs_g2[i]

g1s_averaged = np.array(g1s_averaged)
g2s_averaged = np.array(g2s_averaged)
g1s_wcs_averaged = np.array(g1s_wcs_averaged)
g2s_wcs_averaged = np.array(g2s_wcs_averaged)

plt.figure()
plt.subplot(2,2,1)
plt.hist(g1s_averaged,500)
plt.xlabel('$e_1$')
plt.title('Epoch i-band, Exposure Averaged')
plt.subplot(2,2,2)
plt.hist(g2s_averaged,500)
plt.xlabel('$e_2$')
plt.subplot(2,2,3)
plt.plot(g1s_averaged,g2s_averaged,'+')
plt.xlabel('$e_1$')
plt.ylabel('$e_2$')
plt.show()

plt.figure()
plt.subplot(2,2,1)
plt.hist(g1s_wcs_averaged,500)
plt.xlabel('$e_1$')
plt.title('Epoch i-band, WCS Exposure Averaged')
plt.subplot(2,2,2)
plt.hist(g2s_wcs_averaged,500)
plt.xlabel('$e_2$')
plt.subplot(2,2,3)
plt.plot(g1s_wcs_averaged,g2s_wcs_averaged,'+')
plt.xlabel('$e_1$')
plt.ylabel('$e_2$')
plt.xlim([-0.01,0.01])
plt.ylim([-0.01,0.01])
plt.show()

plt.figure()
plt.plot(g1s_averaged,g1s_wcs_averaged,'k+',label='$g_1$')
plt.plot(g2s_averaged,g2s_wcs_averaged,'k+',label='$g_2$')
plt.xlabel('$g_{exposure}$')
plt.ylabel('$g_{wcs}$')
plt.title('Epoch i-band')
plt.legend()
plt.show()

# average in bins of g_wcs
w=0.0005
bin_l = np.arange(np.min(g1s_wcs_averaged),np.max(g1s_wcs_averaged)-w,w)
bin_u = np.arange(np.min(g1s_wcs_averaged)+w,np.max(g1s_wcs_averaged),w)
bin_c = (bin_l+bin_u)/2.0
Nbin = len(bin_c)

binned_av_g1 = np.zeros(np.shape(bin_c))
binned_N1 = np.zeros(np.shape(bin_c))
binned_stdev_g1 = np.zeros(np.shape(bin_c))
binned_av_g2 = np.zeros(np.shape(bin_c))
binned_N2 = np.zeros(np.shape(bin_c))
binned_stdev_g2 = np.zeros(np.shape(bin_c))

for ibin in range(0,Nbin):
	g1s_use = g1s_averaged[np.asanyarray(np.where((g1s_wcs_averaged>=bin_l[ibin])&(g1s_wcs_averaged<bin_u[ibin])))]
	g2s_use = g2s_averaged[np.asanyarray(np.where((g2s_wcs_averaged>=bin_l[ibin])&(g2s_wcs_averaged<bin_u[ibin])))]

	binned_av_g1[ibin] = np.mean(g1s_use)
	binned_av_g2[ibin] = np.mean(g2s_use)
	a = np.shape(g1s_use)
	b = np.shape(g2s_use)
	binned_N1[ibin]=a[1]
	binned_N2[ibin]=b[1]
	binned_stdev_g1[ibin] = np.sqrt(np.sum((g1s_use-binned_av_g1[ibin])**2)/binned_N1[ibin])
	binned_stdev_g2[ibin] = np.sqrt(np.sum((g2s_use-binned_av_g2[ibin])**2)/binned_N2[ibin])

binned_av_g1[np.where(binned_N1<=1)]=0 # values with fewer than 2 entries don't give good means/errors
binned_av_g2[np.where(binned_N2<=1)]=0
binned_stdev_g1[np.where(binned_N1<=1)]=0
binned_stdev_g2[np.where(binned_N2<=1)]=0



plt.figure()
plt.errorbar(bin_c,binned_av_g1,binned_stdev_g1,color='k',label='$g_1$',linestyle='')
plt.errorbar(bin_c,binned_av_g2,binned_stdev_g2,color='r',label='$g_2$',linestyle='')
plt.plot(np.arange(-1,2),np.zeros(np.shape(np.arange(-1,2))),'k--')
plt.xlabel('$<g^{wcs}>$')
plt.ylabel('$<g^{i}>$')
plt.legend(loc=0)
plt.title('Epoch data averaged over exposures, i-band')
plt.xlim([-0.01,0.01])
plt.show()





