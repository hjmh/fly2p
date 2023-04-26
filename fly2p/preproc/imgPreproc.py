import xarray as xr

from os.path import sep, exists
from os import mkdir, makedirs, getcwd

from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter

import numpy as np
import pandas as pd
import json
from dataclasses import dataclass, asdict

from matplotlib import pyplot as plt

## DATA CLASS FOR IMAGING DATA
@dataclass
class imagingTimeseries:
    # dataclass holding roi-time series data extracted from imaging experiment

    # metadata
    imgMetadata: dict
    expMetadata: dict

    # reference images
    refImage: xr.DataArray # image used for motion correction (MC)
    dffStack: xr.DataArray # image or stack, maximum intensity projection of DFF over time after MC
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
        self.refImage.to_netcdf(sep.join([savepath,'refImg.nc']), mode='w')
        self.dffStack.to_netcdf(sep.join([savepath,'dffStack.nc']), mode='w')
        self.F0stack.to_netcdf(sep.join([savepath,'F0stack.nc']), mode='w')

        # save roi data
        np.save(sep.join([savepath,'roiMask']),self.roiMask)
        self.roiDFF.to_csv(sep.join([savepath,'roiDFF.csv']))

        return savepath

# save subset of data (when doing motion correction separately)
def saveMetaData(saveDir, saveName, imgMetadata, rawTiff, genotype, flyID, trial, roitype, region):
    import json
    expMetadata = {
        'tiffilename': rawTiff,
        'genotype': genotype,
        'flyid': flyID,
        'trial':trial,
        'roitype': roitype,
        'brainregion': region
    }

    savepath = sep.join([saveDir,saveName, 'img'])

    # make directory
    if not exists(savepath):
        makedirs(savepath)

    # save metadata
    with open(sep.join([savepath,'imgMetadata.json']), 'w') as outfile:
        json.dump(imgMetadata, outfile,indent=4)
    with open(sep.join([savepath,'expMetadata.json']), 'w') as outfile:
        json.dump(expMetadata, outfile,indent=4)

    return savepath


def saveXArray(saveDir, saveName, fileName, myArray):
    savepath = sep.join([saveDir,saveName, 'img'])

    # make directory
    if not exists(savepath): makedirs(savepath)

    # save data
    myArray.to_netcdf(sep.join([savepath,fileName+'.nc']), mode='w')
    return savepath

def saveRoiData(roiMask, roiDFF, savepath, saveName = 'roiDFF.csv'):
    np.save(sep.join([savepath, 'img','roiMask']),roiMask)
    roiDFF.to_csv(sep.join([savepath,'img', saveName]))
    return savepath

# construct imaging timeseries object from saved data files
def loadImagingTimeseries(path2imgdat):

    dffStack_load = xr.open_dataarray(path2imgdat+sep+'dffStack.nc', decode_coords='coordinates')
    F0Xarray_load = xr.open_dataarray(path2imgdat+sep+'F0stack.nc', decode_coords='coordinates')
    refImg_load = xr.open_dataarray(path2imgdat+sep+'refImg.nc', decode_coords='coordinates')

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
        dffStack = dffStack_load,
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
def computeDFF(stack,
               order = 3,
               window = 7,
               baseLinePercent = 10,
               offset = 0.0001,
               subtract = False,
               background_mask = None,
               baselineLowestMean = False):
    #if subtract == True and a background_mask is provided, ROI based subtraction is assumed

    dffStack = np.zeros((stack.shape))

    if len(stack.shape) == 3:
        print('processing 3d stack')
        stackF0 = np.zeros((stack["xpix [µm]"].size,stack["ypix [µm]"].size))
        filtStack = gaussian_filter(stack, sigma=[0,2,2])

        #background estimation
        if subtract:
            if background_mask is not None:
                filtStack = xr.apply_ufunc(roi_subtract,filtStack,
                                           kwargs={"background_mask":background_mask})
            else:
                filtStack = xr.apply_ufunc(roll_ball,filtStack)

        filtF = savgol_filter(filtStack.astype('float'), window, order, axis=0)

        # Estimate baseline
        if baselineLowestMean:
            # TODO: replace with masked asarray
            # data = np.random.randint(0, 255, (100,100,100))
            # thr = np.percentile(data, 10)
            # masked = np.ma.masked_array(data, mask=data>thr)
            # result = masked.mean(axis=0)
            for x in range(stack["xpix [µm]"].size):
                for y in range(stack["ypix [µm]"].size):
                    stackF0[x,y] = filtF[filtF[:,x,y] < np.percentile(filtF[:,x,y], baseLinePercent, axis=0),x,y].mean()
        else:
            stackF0 = np.percentile(filtF, baseLinePercent, axis=0) + offset
        stackF0[np.where(stackF0 == 0)[0]] += offset

        # Compute dF/F_0 = (F_raw - F_0)/F_0
        dffStack = (filtF - stackF0) / stackF0

    else:
        print('processing 4d stack')
        stackF0 = np.zeros((stack["planes [µm]"].size, stack["xpix [µm]"].size,
                            stack["ypix [µm]"].size))

        for p in range(stack["planes [µm]"].size):
            filtStack = gaussian_filter(stack[{'planes [µm]': p}].squeeze(), sigma=[0,2,2])

            filtF = savgol_filter(filtStack.astype('float'), window, order, axis=0)

            # Estimate baseline
            if baselineLowestMean:
                for x in range(stack["xpix [µm]"].size):
                    for y in range(stack["ypix [µm]"].size):
                        stackF0[p,x,y] = filtF[filtF[:,x,y] < np.percentile(filtF[:,x,y], baseLinePercent, axis=0),x,y].mean()
            else:
                stackF0[p,:,:] = np.percentile(filtF, baseLinePercent, axis=0) + offset
            stackF0[p,np.where(stackF0[p,:,:] == 0)[0]] += offset

            # Compute dF/F_0 = (F_raw - F_0)/F_0
            dffStack[:,p,:,:] = (filtF - stackF0[p,:,:]) / stackF0[p,:,:]

    return np.float32(dffStack), np.float32(stackF0)


### functions for background subtraction
# Method 1: Background is manually drawn
def roi_subtract(stack, background_mask, order = 3,window = 7):
    T = stack.shape[0]

    #get mean value of fluorescence inside the background region
    back_F = [np.nanmean(np.where(background_mask, stack[t,:,:],
                                  float('nan'))) for t in range(T)]
    back_F = savgol_filter(np.asarray(back_F).astype('float'), window, order)
    #assumption: background is homogeneous
    #the best estimate of background fluorescence in the image is the mean of the fluorescence
    #in the background region

    #subtract
    filt = np.array([stack[t,:,:]-back_F[t] for t in range(T)])

    #range should be same as before subtraction and convert to integer values
    filt = np.round(((filt-filt.min())/
                     (filt.max()-filt.min())*(stack.max()-stack.min()))+stack.min())

    return filt

# Method 2: Rolling ball subtraction
def roll_ball(stack,radius=0.15):
    #slower than ROI based subtraction

    r = stack.shape[1]*radius
    R = r/np.max(stack) #normalised radius

    #larger radius fraction implies less artefacts but slower processing
    ker = restoration.ellipsoid_kernel((1, 2*r, 2*r), 2*R)

    #estimate background for each static image
    background = restoration.rolling_ball(stack,kernel=ker)

    #subtract
    filt = stack-background

    #range should be same as before subtraction and convert to integer values
    filt = np.round(((filt-filt.min())/
                     (filt.max()-filt.min())*(stack.max()-stack.min()))+stack.min())

    return filt

## MOTION CORRECTION ##

def computeMotionShift(stack, refImage, upsampleFactor, sigmaval = 2, doFilter = False, stdFactor = 2, showShiftFig = False):
    from skimage.registration import phase_cross_correlation

    if len(refImage.shape) == 3:
        print('perform motion correction on a volume')
        refImgFilt = refImage.copy()
        for p in range(stack['planes [µm]'].size):
            refImgFilt[p,:,:] = gaussian_filter(refImage[p,:,:], sigma=sigmaval)
        shift = np.zeros((2, stack['planes [µm]'].size,stack['volumes [s]'].size))
        error = np.zeros((stack['planes [µm]'].size,stack['volumes [s]'].size))
        diffphase = np.zeros((stack['planes [µm]'].size,stack['volumes [s]'].size))
    else:
        print('perform motion correction on a single plane/max projection')
        refImgFilt = gaussian_filter(refImage, sigma=sigmaval)

        shift = np.zeros((2, stack['volumes [s]'].size))
        error = np.zeros(stack['volumes [s]'].size)
        diffphase = np.zeros(stack['volumes [s]'].size)

    # compute shift
    for i in range(stack['volumes [s]'].size):
        if len(refImage.shape) == 3:
            for p in range(stack['planes [µm]'].size):
                shifImg = stack[i,p,:,:]
                shifImgFilt = gaussian_filter(shifImg, sigma=sigmaval)

                # compute shift
                shift[:,p,i], error[p,i], diffphase[p,i] = phase_cross_correlation(refImgFilt[p,:,:].data, shifImgFilt,
                                                                             upsample_factor = upsampleFactor, normalization=None)
        else:
            shifImg = stack[i,:,:]
            shifImgFilt = gaussian_filter(shifImg, sigma=sigmaval)

            # compute shift
            shift[:,i], error[i], diffphase[i] = phase_cross_correlation(refImgFilt, shifImgFilt,
                                                                         upsample_factor = upsampleFactor, normalization=None)
    if showShiftFig:
        if len(refImage.shape) == 3:
            fig, axs = plt.subplots(2,1,figsize=(15,6))
            axlab = ['x','y']
            for i, ax in enumerate(axs):
                ax.plot(shift[i,:].T)
                ax.set_xlabel('frames')
                ax.set_ylabel('image shift for {}'.format(axlab[i]))
        else:
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
    from scipy.ndimage import shift as spshift

    #check if shift was calculated for each plane in a volume separately, then check if stack to be aligned is 3d or 4d

    #stack should be an xarray
    stackMC = stack.copy()

    if len(shift.shape) == 3:
        # separate shifts for each plane in a volume
        if len(stack.shape) < 4:
            print("Imaging stack needs to be 4D.")
            return np.nan*stackMC
        for p in range(stack['planes [µm]'].size):
            for i in range(stack['volumes [s]'].size):
                shifImg = stack[i,p,:,:]
                stackMC[i,p,:,:] = spshift(shifImg, shift[:,p,i], order=1,mode='reflect')

    else:
        #one shift per volume per time point
        if len(stack.shape) < 4:
            # motion correction on single plane or max projection
            for i in range(stack['volumes [s]'].size):
                shifImg = stack[i,:,:]
                stackMC[i,:,:] = spshift(shifImg, shift[:,i], order=1,mode='reflect')
        else:
            #motion correction of 4D stack
            for v in range(stack["volumes [s]"].size):  #move one volume at a time
                tmpVol = stack[{"volumes [s]": v}]
                for p in range(tmpVol["planes [µm]"].size):
                    stackMC[v,p,:,:]  = spshift(tmpVol[p,:,:], shift[:,v], order=1,mode='reflect')

    return stackMC
