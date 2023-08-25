# siFRC 

## Highlights
(1) This implementation helps to understand Fourier Ring Correlation (FRC).
(2) It allows one to calculate FRC-based image resolution from a Single Image.

## Additional guide
(1) Create demo images: python lena_noise_creation.py.
(2) Go through lenaFRC.ipynb to see the relation between the FRC and the SNR.
(3) Calculate single image resolution using demo images.
(4) Compare FRC value obtained from single image (siFRC) against that obtained from two images using main_2imgs.py.

```
usage: main.py [-h] [--input-dir INPUT_DIR] [--output-dir OUTPUT_DIR]
                   [--io-plot IO_PLOT] [--thres THRES] [--crop]
                   [--frc-len FRC_LEN] [--frc-center FRC_CENTER]

Python FRC

optional arguments:
  -h, --help         show this help message and exit
  --input-dir        input folder name with experimental images 
  --output-dir       output folder name that stores FRC plots (default: )
  --io-plot          whether to display splitted images and its FRC
                     (default: False)
  --thres            threshold type to be used in the FRC (default: half-
                     bit)
  --crop             whether to pass input images as is or to crop before
                     passing (default: False)
  --frc-len          total length in x-y direction with center as frc_len/2
                     (default: 400)
  --frc-center       new center of cropped image become
                     (min(image.shape)/2) + frc-center (default: 0)
```

## Example usage 

`python main.py --input-dir './demo_images/noisy_lena_512/' --output-dir 'results/demo_images/noisy_lena_512' --thres 'half-bit'`

## Final Resolution 

*************************************************************************
```
==> if the intersection of the FRC curve and the threhold is (P) in x-axis
==> and 1 pixel = q unit (may be [nm] or [um] or [cm])
==> then the final resolution is (1/P)*q*sqrt(2) unit
```
**************************************************************************
## Package requirements

numpy, matplotlib, glob, imageio, itertools, scipy

## References

M. Van Heel and M. Schatz, “Fourier shell correlation threshold criteria,” Journal of structural biology, vol. 151, no. 3, pp. 250–262, 2005.

S. V. Koho, G. Tortarolo, M. Castello, T. Deguchi, A. Di- aspro, and G. Vicidomini, “Fourier ring correlation simplifies image restoration in fluorescence microscopy,” bioRxiv, p. 535583, 2019.


