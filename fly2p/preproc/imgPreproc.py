from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift
import xarray as xr

from os.path import sep, exists
from os import mkdir, makedirs, getcwd

from scipy.signal import savgol_filter
from scipy.ndimage.filters import gaussian_filter

import numpy as np
import pandas as pd
import json
from dataclasses import dataclass, asdict

from matplotlib import pyplot as plt

# ToDo: make dataclass for holding preprocessed, full imaging data (DFF volume,..)


## DATA CLASS FOR IMAGING DATA
@dataclass
class imagingTimeseries:
    # dataclass holding roi-time series data extracted from imaging experiment

    # metadata
    imgMetadata: dict
    expMetadata: dict

    # reference images
    refImage: xr.DataArray # image used for motion correction (MC)
    refStackMC: xr.DataArray # image or stack, mean flourescense over time after MC
    dffMIP: xr.DataArray # image or stack, maximum intensity projection of DFF over time after MC 
    F0stack: xr.DataArray # image or stack of F0 (baseline flourescences)

    # roi data
    roitype: str #polygons ("poly") or correlation-based ("corr")?
    roiMask: np.ndarray
    roiDFF: pd.DataFrame

    def saveData(self, saveDir, saveName):
        savepath = sep.join([saveDir,saveName, 'img'])
        # make directory
        if not exists(savepath):
            makedirs(savepath)

        # save metadata
        with open(sep.join([savepath,'imgMetadata.json']), 'w') as outfile:
            json.dump(self.imgMetadata, outfile,indent=4)
        with open(sep.join([savepath,'expMetadata.json']), 'w') as outfile:
            json.dump(self.expMetadata, outfile,indent=4)

        # reference images
        self.refImage.to_netcdf(sep.join([savepath,'refImg.nc']))
        self.refStackMC.to_netcdf(sep.join([savepath,'refStackMC.nc']))
        self.dffMIP.to_netcdf(sep.join([savepath,'dffMIP.nc']))
        self.F0stack.to_netcdf(sep.join([savepath,'F0stack.nc']))

        # save roi data
        np.save(sep.join([savepath,'roiMask']),self.roiMask)
        self.roiDFF.to_csv(sep.join([savepath,'roiDFF.csv']))

        return savepath


# construct imaging timeseries object from saved data files
def loadImagingTimeseries(path2imgdat):
    
    dffMIP_load = xr.open_dataset(path2imgdat+sep+'dffMIP.nc')
    F0Xarray_load = xr.open_dataset(path2imgdat+sep+'F0stack.nc')
    refImg_load = xr.open_dataset(path2imgdat+sep+'refImg.nc')
    refStackMC_load = xr.open_dataset(path2imgdat+sep+'refImg.nc')

    with open(path2imgdat+sep+'imgMetadata.json') as f:
        basicMetadat_load = json.load(f)
    with open(path2imgdat+sep+'expMetadata.json') as f:
        expMetadat_load = json.load(f)

    roiDat_load = pd.read_csv(path2imgdat+sep+'roiDFF.csv')
    roimask_load = np.load(path2imgdat+sep+'roiMask.npy')

    imgTS = imagingTimeseries(
        imgMetadata = basicMetadat_load,
        expMetadata = expMetadat_load,
        refImage = refImg_load, 
        refStackMC = refStackMC_load, 
        dffMIP = dffMIP_load, 
        F0stack = F0Xarray_load,
        roitype = expMetadat_load['roitype'],
        roiMask = roimask_load,
        roiDFF = roiDat_load
    )
    
    return imgTS

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

## CONVERT TO XARRAY when no time dimension
def refStack2xarray(stack, basicMetadat, data4D = True):
    if data4D:
        slices = [i*basicMetadat['stackZStepSize'] for i in range(stack.shape[0])]
        xpx = np.linspace(0, basicMetadat['xrange_um'], stack.shape[1])
        ypx = np.linspace(0, basicMetadat['yrange_um'], stack.shape[2])
        imgStack = xr.DataArray(stack, coords = [slices, xpx, ypx], dims = ['planes [µm]', 'xpix [µm]', 'ypix [µm]'])

    else:
        xpx = np.linspace(0, basicMetadat['xrange_um'], stack.shape[0])
        ypx = np.linspace(0, basicMetadat['yrange_um'], stack.shape[1])
        imgStack = xr.DataArray(stack, coords = [xpx, ypx], dims = ['xpix [µm]', 'ypix [µm]'])
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

def computeMotionShift(stack, refImage, upsampleFactor, sigmaval = 2, doFilter = False, stdFactor = 2, showShiftFig = False):
    from scipy.ndimage.filters import gaussian_filter

    refImgFilt = gaussian_filter(refImage, sigma=sigmaval)

    shift = np.zeros((2, stack['volumes [s]'].size))
    error = np.zeros(stack['volumes [s]'].size)
    diffphase = np.zeros(stack['volumes [s]'].size)

    # compute shift
    for i in range(stack['volumes [s]'].size):
        shifImg = stack[i,:,:]

        shifImgFilt = gaussian_filter(shifImg, sigma=sigmaval)

        # subpixel precision
        shift[:,i], error[i], diffphase[i] = phase_cross_correlation(refImgFilt, shifImgFilt,
                                                                     upsample_factor = upsampleFactor)
    if showShiftFig:
        fig, ax = plt.subplots(1,1,figsize=(15,5))
        ax.plot(shift[0,:])
        ax.plot(shift[1,:])
        ax.set_xlabel('frames')
        ax.set_ylabel('image shift')
    
    if doFilter:
        shiftFilt_x = shift[0,:].copy()
        shiftFilt_y = shift[1,:].copy()
        shiftFilt_x[abs(shiftFilt_x) > stdFactor*np.std(shiftFilt_x)] = np.nan
        shiftFilt_y[abs(shiftFilt_y) > stdFactor*np.std(shiftFilt_y)] = np.nan

        allT = np.arange(len(shiftFilt_x))
        shiftFilt_x_interp = np.interp(allT, allT[~np.isnan(shiftFilt_x)], shiftFilt_x[~np.isnan(shiftFilt_x)])
        shiftFilt_y_interp = np.interp(allT, allT[~np.isnan(shiftFilt_y)], shiftFilt_y[~np.isnan(shiftFilt_y)])
    
        if showShiftFig:
            ax.plot(shiftFilt_x_interp,'b')
            ax.plot(shiftFilt_y_interp,'c')
        
        return np.vstack((shiftFilt_x_interp,shiftFilt_y_interp))
    else:
        return shift
    

def motionCorrection(stack, shift):
    stackMC = stack.copy()#np.ones(stackMP.shape).astype('int16')

    # shift stack according to shift
    for i in range(stack['volumes [s]'].size):
        shifImg = stack[i,:,:]
        offset_image = fourier_shift(np.fft.fftn(shifImg), shift[:,i])
        stackMC[i,:,:] = np.fft.ifftn(offset_image).real.astype('uint16')
        
    return stackMC


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
