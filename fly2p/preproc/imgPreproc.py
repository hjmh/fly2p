from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift

import numpy as np
    

## DATA CLASS FOR IMAGING DATA


    
## MOTION CORRECTION ##

def motionCorrection(stack, refImage, upsampleFactor, sigmaval):
    from scipy.ndimage.filters import gaussian_filter
    
    refImgFilt = gaussian_filter(refImage, sigma=sigmaval)
    
    shift = np.zeros((2, stack['volumes [s]'].size))
    error = np.zeros(stack['volumes [s]'].size)
    diffphase = np.zeros(stack['volumes [s]'].size)

    stackMC = stack.copy()#np.ones(stackMP.shape).astype('int16')

    for i in range(stack['volumes [s]'].size):
        shifImg = stack[i,:,:]

        shifImgFilt = gaussian_filter(shifImg, sigma=sigmaval)

        # subpixel precision
        shift[:,i], error[i], diffphase[i] = phase_cross_correlation(refImgFilt, shifImgFilt, 
                                                                     upsample_factor = upsampleFactor)

        offset_image = fourier_shift(np.fft.fftn(shifImg), shift[:,i])
        stackMC[i,:,:] = np.fft.ifftn(offset_image).real.astype('uint16')
    
    return stackMC, shift


def applyShiftTo4Dstack(stack4d, shift):
    #stack4d should be an xarray
    stack4dMC = stack4d.copy()
    
    for v in range(stack4d["volumes [s]"].size):
        #move one volume at a time
        tmpVol = stack4d[{"volumes [s]": v}]
        
        for p in range(tmpVol["planes [µm]"].size):
            offset_image = fourier_shift(np.fft.fftn(tmpVol[p,:,:]), shift[:,v])
            stack4dMC[v,p,:,:] = np.fft.ifftn(offset_image).real.astype('uint16')

    return stack4dMC