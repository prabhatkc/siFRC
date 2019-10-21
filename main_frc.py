import sys
import os 

import frc_utils as frc_util
import secondary_utils as su 
import argparse
import glob 
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="Python FRC",
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input-dir', default='', type=str,
                    help='input folder name with experimental images')
parser.add_argument('--output-dir', default='', type=str,
                    help='output folder name that stores FRC plots')
parser.add_argument('--io-plot', default=False, type=bool,
                    help='whether to display splitted images and its FRC')
parser.add_argument('--thres', default='half-bit', type=str,
                    help='threshold type to be used in the FRC')
parser.add_argument('--crop', action='store_true',
					help='whether to pass input images as is or to crop before passing')
parser.add_argument('--frc-len', default=400, type=int,
					help='total length in x-y direction with center as frc_len/2')
parser.add_argument('--frc-center', default=0, type=int,
					help='new center of cropped image become (min(image.shape)/2) + frc-center')

args       = parser.parse_args()
io_plot    = args.io_plot
input_dir  = args.input_dir
output_dir = args.output_dir
threshold  = args.thres
crop 	   = args.crop

print('====> crop is:', crop)
inside_square	= True
anaRing			= True

if crop is True:
	#total length in x-y direction with center as (frc_len+frc_center)/2
	frc_len    = args.frc_len
	frc_center = args.frc_center

if not os.path.isdir(output_dir):
  print("===> creating the directory to store results from FRC calculations as: ")
  print("    ", output_dir)
  os.makedirs(output_dir)

#img_dir = os.path.join(os.getcwd(), input_dir)
img_names = sorted(glob.glob(os.path.join(input_dir, "*.*")))

for i in range(len(img_names)):
	
	img = su.imageio_imread(img_names[i])
	img = img.astype(np.float)
	img = su.normalize_data_ab(0, 1, img)
	if crop is True:
		frc_img = frc_util.get_frc_img(img, frc_len, frc_center)
	else:
		frc_img = img

	if io_plot is True: su.plot2dlayers(frc_img)

	sa1, sa2, sb1, sb2 = frc_util.diagonal_split(frc_img)
	all_splits = np.asanyarray([sa1, sa2, sb1, sb2])
	if io_plot is True: su.multi2dplots(2, 2, all_splits, 0)

	xc, corr1, xt, thres_val = frc_util.FRC(sa1, sa2, thresholding=threshold, inscribed_rings=inside_square, analytical_arc_based=anaRing)
	_, corr2, _ , _          = frc_util.FRC(sa1, sb1, thresholding=threshold, inscribed_rings=inside_square, analytical_arc_based=anaRing)
	_, corr3, _, _ 			 = frc_util.FRC(sa1, sb2, thresholding=threshold, inscribed_rings=inside_square, analytical_arc_based=anaRing)
	_, corr4, _ , _          = frc_util.FRC(sa2, sb1, thresholding=threshold, inscribed_rings=inside_square, analytical_arc_based=anaRing)
	#_, corr5, _, _ 	     = frc_util.FRC(sa2, sb2, thresholding=threshold, inscribed_rings=inside_square, analytical_arc_based=anaRing)
	#_, corr6, _ , _         = frc_util.FRC(sb1, sb2, thresholding=threshold, inscribed_rings=inside_square, analytical_arc_based=anaRing)
	corr_avg                 = (corr1+corr2+corr3+corr4)/4.0

	if threshold=='all':
		plt.plot(xc[:-1], corr_avg[:-1], label = 'chip-FRC', color='black')
		plt.plot(xt[:-1], (thres_val[0])[:-1], label='one-bit', color='green')
		plt.plot(xt[:-1], (thres_val[1])[:-1], label='half-bit', color='red')
		plt.plot(xt[:-1], (thres_val[2])[:-1], label='0.5 -Thres', color='brown')
		plt.plot(xt[:-1], (thres_val[3])[:-1], label='EM', color='Orange')
	else:
		plt.plot(xc[:-1], corr_avg[:-1], label = 'FRC', color='black')
		plt.plot(xt[:-1], thres_val[:-1], label=threshold, color='red')

	plt.xlim(-0.01, 1.0)
	plt.ylim(0.0, 1)
	plt.grid(linestyle='dotted', color='black', alpha=0.3) 
	plt.xticks(np.arange(0, 1, step=0.1))
	plt.yticks(np.arange(0, 1, step=0.1))
	plt.legend(prop={'size':15})
	plt.xlabel('Spatial Frequency/Nyquist', {'size':15})
	plt.title ('Fourier Ring Correlation (FRC)', {'size':20})
	plt.tick_params(axis='both', labelsize=14)
	out_img_name = img_names[i].split('/')[3]
	out_img_name = ('./'+ output_dir + '/FRCof_'+ out_img_name.split('.')[0]+ '.pdf')
	plt.savefig(out_img_name, dpi=300)
	if io_plot is True: plt.show()
	plt.close()