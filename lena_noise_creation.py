import numpy as np
import frc_utils as fu
import secondary_utils as su

lena  = su.imageio_imread('./demo_images/512/lena512.jpg')
lena  = lena[:, :, 0]
lena  = lena.astype(np.float)
lena  = su.normalize_data_ab(0, 1, lena) #normalizing image to range [0, 1]

# average true signal in each pixel of lena 
h, w = lena.shape
s_bar = (np.linalg.norm(lena)**2)/(h*w)

# SNR of 0.4142 translates as 1/2 bit of information
noisy_lena1 = su.add_white_noise(lena, 0, np.sqrt(s_bar/.4142), 1.0, (h,w))
noisy_lena2 = su.add_white_noise(lena, 0, np.sqrt(s_bar/.4142), 1.0, (h,w))

noise1  = noisy_lena1 - lena
noise2 =  noisy_lena2 - lena

su.imsave(noisy_lena1.astype(np.float32), './demo_images/512/noisy_lena1.tif', type='orginal')
su.imsave(noisy_lena2.astype(np.float32), './demo_images/512/noisy_lena2.tif', type='orginal')
su.imsave(noise1.astype(np.float32), './demo_images/512/noise1.tif', type='orginal')
su.imsave(noise2.astype(np.float32), './demo_images/512/noise2.tif', type='orginal')

sa1, sa2, sb1, sb2   = fu.diagonal_split(noisy_lena1)
su.imsave(sa1.astype(np.float32), './demo_images/512/sa1.tif', type='orginal')
su.imsave(sa2.astype(np.float32), './demo_images/512/sa2.tif', type='orginal')
su.imsave(sb1.astype(np.float32), './demo_images/512/sb1.tif', type='orginal')
su.imsave(sb2.astype(np.float32), './demo_images/512/sb2.tif', type='orginal')

n1, n2, n3, n4 = fu.diagonal_split(noise1)
su.imsave(n1.astype(np.float32), './demo_images/512/n1.tif', type='orginal')
su.imsave(n2.astype(np.float32), './demo_images/512/n2.tif', type='orginal')
su.imsave(n3.astype(np.float32), './demo_images/512/n3.tif', type='orginal')
su.imsave(n4.astype(np.float32), './demo_images/512/n4.tif', type='orginal')