from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift
import xarray as xr

from scipy.signal import savgol_filter
from scipy.ndimage.filters import gaussian_filter

import numpy as np
from dataclasses import dataclass, asdict

## DATA CLASS FOR IMAGING DATA
@dataclass
class imagingTimeseries:
    # dataclass holding roi-time series data extracted from imaging experiment

    # metadata
    imgMetadata: dict
    tiffilename: str
    brainregion: str
    genotype: str
    flyid: str

    # reference images
    refImage: xr.DataArray # image used for motion correction (MC)
    refStackMC: xr.DataArray # image or stack, mean flourescense over time after MC
    dffStack: xr.DataArray # # image or stack, mean DFF over time after MC

    # roi data
    roitype: str #polygons ("poly") or correlation-based ("corr")?
    roiMask: np.ndarray
    roiDFF: np.ndarray
    time: np.ndarray

# ToDo: make dataclass for holding preprocessed, full imaging data (DFF volume,..)

## CONVERT TO XARRAY
def stack2xarray(stack, basicMetadat, data4D = True):
    volcoords = [i/basicMetadat['scanVolumeRate'] for i in range(stack.shape[0])]
    if data4D:
        slices = [i*basicMetadat['stackZStepSize'] for i in range(stack.shape[1])]
        xpx = np.linspace(0, basicMetadat['xrange_um'], stack.shape[2])
        ypx = np.linspace(0, basicMetadat['yrange_um'], stack.shape[3])
        imgStack = xr.DataArray(stack, coords = [volcoords, slices, xpx, ypx],
                            dims = ['volumes [s]', 'planes [µm]', 'xpix [µm]', 'ypix [µm]'])

    else:
        xpx = np.linspace(0, basicMetadat['xrange_um'], stack.shape[1])
        ypx = np.linspace(0, basicMetadat['yrange_um'], stack.shape[2])
        imgStack = xr.DataArray(stack, coords = [volcoords, xpx, ypx],
                            dims = ['volumes [s]', 'xpix [µm]', 'ypix [µm]'])

    minval = np.min(imgStack)
    if minval < 0: imgStack = imgStack - minval

    return imgStack


## DFF ##
# TODO: Make sure this works with 3D and 4D xarrays (i.e. with and without planes dimension)...
def computeDFF(stack3d, order = 3, window = 7, baseLinePercent = 10, offset = 0.0001):
    dffStack = np.zeros((stack3d.shape))
    stackF0 = np.zeros((stack3d["xpix [µm]"].size,stack3d["ypix [µm]"].size))

    filtStack = gaussian_filter(stack3d, sigma=[0,2,2])

    for x in range(stack3d["xpix [µm]"].size):
        for y in range(stack3d["ypix [µm]"].size):

            filtF = savgol_filter(filtStack[:,x,y], window, order)

            # Estimate baseline
            F0 = np.percentile(filtF, baseLinePercent)
            stackF0[x,y] = F0
            if F0 == 0: F0 += offset

            # Compute dF/F_0 = (F_raw - F_0)/F_0
            dFF = (filtF - F0) / F0

            dffStack[:,x,y] = dFF
    return dffStack, stackF0


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
