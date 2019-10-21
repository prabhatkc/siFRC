#####################################################################
# Understand Fourier Ring Correlation/Calculate Image resolution	#
#                           										#
#                                                               	#
# Copyright (C) Prabhat KC/Vincent									#
# All rights reserved. pkc@anl.gov                	        		#
#                                                               	#
#####################################################################


# (1) create demo images 
python lena_noise_creation.py

# (2) go through lenaFRC.ipynb

# (3) Resolution calculation
usage: main_frc.py [-h] [--input-dir INPUT_DIR] [--output-dir OUTPUT_DIR]
                   [--io-plot IO_PLOT] [--thres THRES] [--crop]
                   [--frc-len FRC_LEN] [--frc-center FRC_CENTER]

Python FRC

optional arguments:
  -h, --help            show this help message and exit
  --input-dir 			input folder name with experimental images 
  --output-dir 			output folder name that stores FRC plots (default: )
  --io-plot 		    whether to display splitted images and its FRC
                        (default: False)
  --thres 		        threshold type to be used in the FRC (default: half-
                        bit)
  --crop                whether to pass input images as is or to crop before
                        passing (default: False)
  --frc-len 		    total length in x-y direction with center as frc_len/2
                        (default: 400)
  --frc-center 			new center of cropped image become
                        (min(image.shape)/2) + frc-center (default: 0)

## Example usage 
python main_frc.py --input-dir './exp_images/std_data/' --output-dir 'sults/FRC_std_data/' --thres 'half-bit'