import sys
import os 

import frc_utils as frc_util
import secondary_utils as su 
import argparse
import glob 
import numpy as np
import matplotlib.pyplot as plt

inside_square	= True
anaRing			= True
threshold ='all'

img1 = su.imageio_imread('chip_images/siemens/Siemens_star_60nm_16nmZP_8keV_det2sple_3400mm_10s_1.tif')
img2 = su.imageio_imread('chip_images/siemens/Siemens_star_60nm_16nmZP_8keV_det2sple_3400mm_10s_2.tif')

img1 = frc_util.apply_hanning_2d(su.normalize_data_ab(0, 1, img1))
img2 = frc_util.apply_hanning_2d(su.normalize_data_ab(0, 1, img2))
output_dir ='results/siemens/'
if not os.path.isdir(output_dir):
  print("===> creating the directory to store results from FRC calculations as: ")
  print("    ", output_dir)
  os.makedirs(output_dir)
xc, corr_avg, xt, thres_val = frc_util.FRC(img1, img2, thresholding=threshold, inscribed_rings=inside_square, analytical_arc_based=anaRing)

if threshold=='all':
	plt.plot(xc[:-1]/2, corr_avg[:-1], label = 'chip-FRC', color='black')
	plt.plot(xt[:-1]/2, (thres_val[0])[:-1], label='one-bit', color='green')
	plt.plot(xt[:-1]/2, (thres_val[1])[:-1], label='half-bit', color='red')
	plt.plot(xt[:-1]/2, (thres_val[2])[:-1], label='0.5 -Thres', color='brown')
	plt.plot(xt[:-1]/2, (thres_val[3])[:-1], label='EM', color='Orange')
else:
	plt.plot(xc[:-1]/2, corr_avg[:-1], label = 'FRC', color='black')
	plt.plot(xt[:-1]/2, thres_val[:-1], label=threshold, color='red')

plt.xlim(0.0, 0.5)
plt.ylim(0.0, 1)
plt.grid(linestyle='dotted', color='black', alpha=0.3) 
plt.xticks(np.arange(0.0, 0.5, step=0.03))
plt.yticks(np.arange(0, 1, step=0.1))
plt.legend(prop={'size':13})
plt.xlabel('Spatial Frequency (unit$^{-1}$)', {'size':13})
# plt.title ('Fourier Ring Correlation (FRC)', {'size':20})
plt.tick_params(axis='both', labelsize=7)
out_img_name = output_dir.split('/')[-1]
out_img_name = ('./'+ output_dir + '/FRCof_'+ out_img_name.split('.')[0]+ '4rm_2imgs.pdf')
plt.savefig(out_img_name)
#if io_plot is True: plt.show()
#plt.close()