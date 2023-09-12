

##########################################################
# @author: pkc/Vincent 
# --------------------------------------------------------
# Based on the MATLAB code by Michael Wojcik
# modification of python code by sajid
#


import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt 
import itertools
import sys

def diagonal_split(img):
    '''
    This function takes an input image and splits it diagonally into four sub-regions.
    The input image must have dimensions that are divisible by 4.
    '''
    # Get the shape of the input image
    h, w = img.shape
    # Check that the image has dimensions divisible by 4
    if (h % 4 != 0) or (w % 4 != 0):
        raise ValueError('Input image must have dimensions divisible by 4')
    # Crop the image to make sure the dimensions are divisible by 4
    img = img[:h//4*4, :w//4*4]
    h, w = img.shape
    # Create indices for the rows and columns
    row_indices = np.arange(h)
    col_indices = np.arange(w)

    # Split the indices into two groups, one for each diagonal split
    row_split_u = row_indices[::2]
    row_split_d = row_indices[1::2]

    col_split_l = col_indices[::2]
    col_split_r = col_indices[1::2]

    # Split the image into four sub-regions using advanced indexing
    sub_a1 = img[np.ix_(row_split_u, col_split_l)]
    sub_a2 = img[np.ix_(row_split_d, col_split_r)]
    sub_b1 = img[np.ix_(row_split_d, col_split_l)]
    sub_b2 = img[np.ix_(row_split_u, col_split_r)]

    # Return the four sub-regions
    return sub_a1, sub_a2, sub_b1, sub_b2

def get_frc_img(img, frc_img_lx, center=None):
  ''' Returns a cropped image version of input image "img"
  img:        input image
  center:     cropping is performed with center a reference 
              point to calculate length in x and y direction.
              Unless otherwise stated center is basically center
              of input image "img"
  frc_img_lx: length of cropped image in x as well as y. Also 
              the cropped image is made to be square image for
              the FRC calculation
  '''

  h, w = img.shape
  cy = round(min(h, w)/2)
  if center is None:
    cy = cy
  else:
    cy = cy + center
  ep =  cy + round(frc_img_lx/2)
  sp = ep - frc_img_lx
  frc_img = img[sp:ep, sp:ep]
  return frc_img

def ring_indices(x, inscribed_rings=True, plot=False):
    print("ring plots is:", plot)
    
    #read the shape and dimensions of the input image
    shape = np.shape(x)     
    dim = np.size(shape)
    
    '''Depending on the dimension of the image 2D/3D, 
    create an array of integers  which increase with 
    distance from the center of the array
    '''
    if dim == 2 :
        nr,nc = shape
        nrdc = np.floor(nr/2)
        ncdc = np.floor(nc/2)
        r = np.arange(nr)-nrdc 
        c = np.arange(nc)-ncdc 
        [R,C] = np.meshgrid(r,c)
        index = np.round(np.sqrt(R**2+C**2))    
    
    elif dim == 3 :
        nr,nc,nz = shape
        nrdc = np.floor(nr/2)+1
        ncdc = np.floor(nc/2)+1
        nzdc = np.floor(nz/2)+1
        r = np.arange(nr)-nrdc + 1
        c = np.arange(nc)-ncdc + 1 
        z = np.arange(nc)-nzdc + 1 
        [R,C,Z] = np.meshgrid(r,c,z)
        index = np.round(np.sqrt(R**2+C**2+Z**2))+1    
    else :
        print('input is neither a 2d or 3d array')
   
    ''' if inscribed_rings is True then the outmost
    ring use to evaluate the FRC will be the circle
    inscribed in the square input image of size L. 
    (i.e. FRC_r <= L/2). Else the arcs of the rings 
    beyond the inscribed circle will also be
    considered while determining FRC 
    (i.e. FRC_r<=sqrt((L/2)^2 + (L/2)^2))
    '''
    if (inscribed_rings == True):
        maxindex = nr/2
    else:
        maxindex = np.max(index)
    #output = np.zeros(int(maxindex),dtype = complex)

    ''' In the next step the output is generated. The output is an array of length
    maxindex. The elements in this array corresponds to the sum of all the elements
    in the original array correponding to the integer position of the output array 
    divided by the number of elements in the index array with the same value as the
    integer position. 
    
    Depening on the size of the input array, use either the pixel or index method.
    By-pixel method for large arrays and by-index method for smaller ones.
    '''
    print('performed by index method')
    indices = []
    for i in np.arange(int(maxindex)):
        indices.append(np.where(index == i))

    if plot is True:
        img_plane = np.zeros((nr, nc))
        for i in range(int(maxindex)):
            if ((i%20)==0):
                img_plane[indices[i]]=1.0
            
        plt.imshow(img_plane, cmap='copper_r')
        if inscribed_rings is True:
            plt.title('   FRC rings with the max radius as that\
            \n of the inscribed circle in the image (spacing of 20 [px] between rings)')
        else:
            plt.title('   FRC rings extending beyond the radius of\
            \n the inscribed circle in the image (spacing of 20 [px] between rings)')
    return(indices)

def spinavej(x, inscribed_rings=True):
    ''' modification of code by sajid an
    Based on the MATLAB code by Michael Wojcik
    '''
    shape = np.shape(x)     
    dim = np.size(shape)
    ''' Depending on the dimension of the image 2D/3D, create an array of integers 
    which increase with distance from the center of the array
    '''
    if dim == 2 :
        nr,nc = shape
        nrdc = np.floor(nr/2)
        ncdc = np.floor(nc/2)
        r = np.arange(nr)-nrdc 
        c = np.arange(nc)-ncdc  
        [R,C] = np.meshgrid(r,c)
        index = np.round(np.sqrt(R**2+C**2))
        indexf = np.floor(np.sqrt(R**2+C**2))
        indexC = np.ceil(np.sqrt(R**2+C**2))
    
    elif dim == 3 :
        nr,nc,nz = shape
        nrdc = np.floor(nr/2)+1
        ncdc = np.floor(nc/2)+1
        nzdc = np.floor(nz/2)+1
        r = np.arange(nr)-nrdc + 1
        c = np.arange(nc)-ncdc + 1 
        z = np.arange(nz)-nzdc + 1 
        [R,C,Z] = np.meshgrid(r,c,z)
        index = np.round(np.sqrt(R**2+C**2+Z**2))
        indexf = np.floor(np.sqrt(R**2+C**2+Z**2))
        indexC = np.ceil(np.sqrt(R**2+C**2+Z**2))   
    else :
        print('input is neither a 2d or 3d array')
    '''
    The index array has integers from 1 to maxindex arranged according to distance
    from the center
    '''

    if (inscribed_rings == True):
        maxindex = nr/2
    else:
        maxindex = np.max(index)
    output = np.zeros(int(maxindex),dtype = complex)
    
    ''' In the next step output is generated. The output is an array of length
    maxindex. The elements in this array corresponds to the sum of all the elements
    in the original array correponding to the integer position of the output array 
    divided by the number of elements in the index array with the same value as the
    integer position. 
    
    Depending on the size of the input array, use either the pixel or index method.
    By-pixel method for large arrays and by-index method for smaller ones.
    '''
    print('performed by index method')
    # indices = []
    indicesf, indicesC = [], []
    for i in np.arange(int(maxindex)):
        #indices.append(np.where(index == i+1))
        indicesf.append(np.where(indexf == i))
        indicesC.append(np.where(indexC == i))

    for i in np.arange(int(maxindex)):
        #output[i] = sum(x[indices[i]])/len(indices[i][0])
        output[i] = (sum(x[indicesf[i]])+sum(x[indicesC[i]]))/2
    return output

def FRC( i1, i2, thresholding='half-bit', inscribed_rings=True, analytical_arc_based=True, info_split=True):
    
    ''' Check whether the dimensions of input image is 
    square or not
    '''
    if ( np.shape(i1) != np.shape(i2) ) :
        print('\n [!] input images must have the same dimensions\n')
        import sys
        sys.exit()
    if ( np.shape(i1)[0] != np.shape(i1)[1]) :
        print('\n [!] input images must be squares\n')
        import sys
        sys.exit()

    ''' Performing the fourier transform of input
    images to determine the FRC
    '''
    I1 = fft.fftshift(fft.fft2(i1))
    I2 = fft.fftshift(fft.fft2(i2))
    C  = spinavej(I1*np.conjugate(I2), inscribed_rings=inscribed_rings)
    C = np.real(C)
    C1 = spinavej(np.abs(I1)**2, inscribed_rings=inscribed_rings)
    C2 = spinavej(np.abs(I2)**2, inscribed_rings=inscribed_rings)
    C  = C.astype(np.float32)
    C1 = np.real(C1).astype(np.float32)
    C2 = np.real(C2).astype(np.float32)
    FSC    = abs(C)/np.sqrt(C1*C2)
    x_fsc  = np.arange(np.shape(C)[0])/(np.shape(i1)[0]/2)
   
    ring_plots=False
    if(inscribed_rings==True):
      ''' for rings with max radius 
      as L/2
      '''
      if (analytical_arc_based == True):
        ''' perimeter of circle based calculation to
        determine n in each ring
        '''
        r      = np.arange(np.shape(i1)[0]/2) # array (0:1:L/2-1)
        n      = 2*np.pi*r # perimeter of r's from above
        n[0]   = 1
        eps    = np.finfo(float).eps
        #t1 = np.divide(np.ones(np.shape(n)),n+eps)
        inv_sqrt_n = np.divide(np.ones(np.shape(n)),np.sqrt(n)) # 1/sqrt(n)
        x_T    = r/(np.shape(i1)[0]/2)
      else:
        ''' no. of pixels along the border of each circle 
        is used to determine n in each ring
        '''
        indices = ring_indices( i1, inscribed_rings=True, plot=ring_plots)
        N_ind = len(indices)  
        n = np.zeros(N_ind) 
        for i in range(N_ind):
          n[i] = len(indices[i][0])
        inv_sqrt_n = np.divide(np.ones(np.shape(n)),np.sqrt(n)) # 1/sqrt(n)
        x_T = np.arange(N_ind)/(np.shape(i1)[0]/2)

    else:
      ''' for rings with max radius as distance
      between origin and corner of image
      '''
      if (analytical_arc_based == True):
        r      = np.arange(len(C)) # array (0:1:sqrt(rx*rx + ry*ry)) where rx=ry=L/2
        n      = 2*np.pi*r # perimeter of r's from above
        n[0]   = 1
        eps    = np.finfo(float).eps
        #t1 = np.divide(np.ones(np.shape(n)),n+eps)
        inv_sqrt_n = np.divide(np.ones(np.shape(n)),np.sqrt(n)) # 1/sqrt(n)
        x_T    = r/(np.shape(i1)[0]/2)
      else:
        indices = ring_indices( i1, inscribed_rings=False, plot=ring_plots)
        N_ind = len(indices)  
        n = np.zeros(N_ind) 
        for i in range(N_ind):
          n[i] = len(indices[i][0])
        inv_sqrt_n = np.divide(np.ones(np.shape(n)),np.sqrt(n)) # 1/sqrt(n)
        x_T = np.arange(N_ind)/(np.shape(i1)[0]/2)


    if info_split:
      ''' Thresholding based on the fact that 
      SNR is split as the data is divided into
      two half datasets
      '''
      if (thresholding  == 'one-bit'):  T = (0.5+2.4142*inv_sqrt_n)/(1.5+1.4142*inv_sqrt_n) #information split
      elif(thresholding == 'half-bit'): T = (0.4142+2.287*inv_sqrt_n)/ (1.4142+1.287*inv_sqrt_n) # diagonal split 
      elif(thresholding == '0.5'):      T = 0.5*np.ones(np.shape(n))
      elif(thresholding =='em'):        T = (1/7)*np.ones(np.shape(n))
      else:
        t1 = (0.5+2.4142*inv_sqrt_n)/(1.5+1.4142*inv_sqrt_n)
        t2 = (0.2071+1.9102*inv_sqrt_n)/(1.2071+0.9102*inv_sqrt_n) # information split twice 
        t3 = 0.5*np.ones(np.shape(n))
        t4 = (1/7)*np.ones(np.shape(n))
        T = [t1, t2, t3, t4]
    else:  
      if (thresholding == 'one-bit'):  T = (1+3*inv_sqrt_n)/(2+2*inv_sqrt_n) # pixel split
      elif(thresholding == 'half-bit'):T = (0.4142+2.287*inv_sqrt_n)/ (1.4142+1.287*inv_sqrt_n) # diagonal split 
      elif(thresholding == '0.5'):     T = 0.5*np.ones(np.shape(n))
      elif(thresholding=='em'):        T = (1/7)*np.ones(np.shape(n))
      else:
          t1 = (1+3*inv_sqrt_n)/(2+2*inv_sqrt_n)
          t2 = (0.4142+2.287*inv_sqrt_n)/ (1.4142+1.287*inv_sqrt_n) 
          t3 = 0.5*np.ones(np.shape(n))
          t5 = (1/7)*np.ones(np.shape(n))
          T = [t1, t2, t3, t4]

    return (x_fsc, FSC, x_T, T)

def frc_4rm_snr(indices, signal, noise):
    snrFSC = []
    n = 1
    for i in range(len(indices)):
      n          = len(indices[i][0]) 
      SNR_n      = (np.linalg.norm(signal[indices[i]])**2)/n
      SNR_d      = (np.linalg.norm(noise[indices[i]])**2)/n
      SNR        = SNR_n/SNR_d
      FSC_n      = SNR + (2.0/np.sqrt(n))*np.sqrt(SNR) + (1.0/np.sqrt(n))
      FSC_d      = SNR + (2.0/np.sqrt(n))*np.sqrt(SNR) + 1.0
      each_snrFSC = FSC_n/FSC_d
      snrFSC.append(each_snrFSC)
    return(np.asarray(snrFSC))


def apply_hanning_2d(img):
  ''' used for experimental images to minimize 
  boundry effects
  '''
  hann_filt = np.hanning(img.shape[0])
  hann_filt = hann_filt.reshape(img.shape[0], 1)
  #hann_filt = np.power(hann_filt, 2)
  hann_img = img*hann_filt
  hann_img = hann_img*np.transpose(hann_img)
  return(hann_img)