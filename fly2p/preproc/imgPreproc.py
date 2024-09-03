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

def saveRoiData(roiMask, roiDFF, savepath, saveName = 'roiDFF.csv',append=''):
    np.save(sep.join([savepath, 'img','roiMask'+append]),roiMask)
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
               savgol=False,
               order = 3,
               window = 7,
               baseline_percent = 10,
               offset = 0.0001,
               subtract = False,
               background_mask = None,
               gaussian_sigma = [0,2,2],
               showSubtractFig = False,
               baseline_lowest_mean = False, **showSubtractFigParams):
    #if subtract == True and a background_mask is provided, ROI based subtraction is assumed

    dffStack = np.zeros((stack.shape))

    if len(stack.shape) == 3:
        print('processing 3d stack')
        stackF0 = np.zeros((stack["xpix [µm]"].size,stack["ypix [µm]"].size))
        filtStack = gaussian_filter(stack, sigma=gaussian_sigma)

        #background subtraction
        if subtract:
            if background_mask is not None:
                filtStack = xr.apply_ufunc(roi_subtract,filtStack,
                                           kwargs={"background_mask":background_mask})
            else:
                print('please provide a background mask')
        
        if savgol:
            filtF = savgol_filter(filtStack.astype('float'), window, order, axis=0)
        else:
            filtF = filtStack.astype('float')

        # Estimate baseline
        if baseline_lowest_mean:
            # TODO: replace with masked asarray
            # data = np.random.randint(0, 255, (100,100,100))
            # thr = np.percentile(data, 10)
            # masked = np.ma.masked_array(data, mask=data>thr)
            # result = masked.mean(axis=0)
            for x in range(stack["xpix [µm]"].size):
                for y in range(stack["ypix [µm]"].size):
                    stackF0[x,y] = filtF[filtF[:,x,y] < np.percentile(filtF[:,x,y], baseline_percent, axis=0),x,y].mean()
        else:
            stackF0 = np.percentile(filtF, baseline_percent, axis=0) + offset
        stackF0[np.where(stackF0 == 0)[0]] += offset

        # Compute dF/F_0 = (F_raw - F_0)/F_0
        dffStack = (filtF - stackF0) / stackF0

    else:
        print('processing 4d stack')
        stackF0 = np.zeros((stack["planes [µm]"].size, stack["xpix [µm]"].size,
                            stack["ypix [µm]"].size))

        if subtract & showSubtractFig:
            fig, ax = plt.subplots(1,2,figsize = (10,1.5))
            paletteR = plt.cm.Reds(np.linspace(0, 1, stack["planes [µm]"].size))
            paletteB = plt.cm.Greys(np.linspace(0, 1, stack["planes [µm]"].size))

        for p in range(stack["planes [µm]"].size):
            filtStack = gaussian_filter(stack[{'planes [µm]': p}].squeeze(), sigma=[0,2,2])
            
            #background subtraction
            if subtract:
                if len(background_mask.shape) == 3:
                    filtStackSub = xr.apply_ufunc(roi_subtract,filtStack,
                                               kwargs={"background_mask":background_mask[p,:,:]})
                    if showSubtractFig:
                        fig, ax = subtract_fig(fig, ax, [filtStack, filtStackSub], colors=[paletteR[p], paletteB[p]], **showSubtractFigParams)
                else:
                    print('please provide a background mask')
            
            else: 
                filtStackSub = filtStack

            if savgol:
                filtF = savgol_filter(filtStackSub.astype('float'), window, order, axis=0)
            else:
                filtF = filtStackSub.astype('float')

            # Estimate baseline
            if baseline_lowest_mean:
                for x in range(stack["xpix [µm]"].size):
                    for y in range(stack["ypix [µm]"].size):
                        stackF0[p,x,y] = filtF[filtF[:,x,y] < np.percentile(filtF[:,x,y], baseline_percent, axis=0),x,y].mean()
            else:
                stackF0[p,:,:] = np.percentile(filtF, baseline_percent, axis=0) + offset
                
            stackF0[p,np.where(stackF0[p,:,:] == 0)[0]] += offset

            # Compute dF/F_0 = (F_raw - F_0)/F_0
            dffStack[:,p,:,:] = (filtF - stackF0[p,:,:]) / stackF0[p,:,:]

    return np.float32(dffStack), np.float32(stackF0)


### functions for background subtraction
# Background is manually drawn
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

def subtract_fig(fig, ax, stacks, colors=['r','b'], randomPoints = 1, ylims = [150, 250]):

    for i in range(2):
        if isinstance(stacks[i], xr.DataArray):
            stacks[i] = stacks[i].values

        flat_array = stacks[i].reshape(stacks[i].shape[0],-1)

        # Pick 10 random indices per time point
        random_indices = np.random.choice(flat_array.shape[1], size= randomPoints, replace=False)

        # Select the random values using advanced indexing
        random_values_per_time = flat_array[:, random_indices]
    
        ax[i].plot(random_values_per_time,'.',color=colors[i],alpha=0.5,markersize=0.5)
        ax[i].set_xlabel('volume #')
        if i==0:
            ax[i].set_ylabel('raw F')
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].set_ylim(ylims[0],ylims[1])
    return fig, ax

## MOTION CORRECTION ##

def genReference(imgStack, numRefImg, v1, v2, maxProject=False, rippleFilt=False, plane=-1, center_frac=1/100, ref_as_fraction=True): 
    # generate a 2D or 3D reference based on averages a subset of frames from the full time series and optional maxprojection
    
    if ref_as_fraction:
        t1 = round(imgStack['volumes [s]'].size/v1)
        t2 = round(imgStack['volumes [s]'].size/v2)
    else:
        t1= v1
        t2= v2
    
    if maxProject:
        stackMP = np.max(imgStack, axis=1) # max projection over volume
        reference = np.mean(stackMP[ t1 : t1+numRefImg,:,:],axis=0) + np.mean(stackMP[ t2 : t2+numRefImg,:,:],axis=0)
    else:
        reference = np.mean(imgStack[ t1 : t1+numRefImg,:,:,:],axis=0) + np.mean(imgStack[ t2 : t2+numRefImg,:,:,:],axis=0)
        if rippleFilt:
            reference = notchFilter2d(reference, plane=plane, center_frac=center_frac)

    return reference


def computeMotionShift(stack, refImage, upsampleFactor, sigmaval = 2, doFilter = False, stdFactor = 2, showShiftFig = False, inZ=False, mask=None, stdFactorZ=None):
    from skimage.registration import phase_cross_correlation
    
    if len(refImage.shape) == 3:
        if not inZ:
            print('perform motion correction on a volume plane-by-plane')
            refImgFilt = refImage.copy()
            for p in range(stack['planes [µm]'].size):
                refImgFilt[p,:,:] = gaussian_filter(refImage[p,:,:], sigma=sigmaval)
            shift = np.zeros((2, stack['planes [µm]'].size,stack['volumes [s]'].size))
            error = np.zeros((stack['planes [µm]'].size,stack['volumes [s]'].size))
            diffphase = np.zeros((stack['planes [µm]'].size,stack['volumes [s]'].size))
        else:
            print('perform motion correction on a volume including along z-axis')
            #check if sigma too high
            if sigmaval>=(refImage.shape[0]/2):
                print("sigmaval too high; sigmaval set to 2")
            refImgFilt = gaussian_filter(refImage, sigma=max([refImage.shape[0]/2,2])) 
            #planes above and below will be smoothed
            
            shift = np.zeros([3, stack['volumes [s]'].size])
            error = np.zeros(stack['volumes [s]'].size)
            diffphase = np.zeros(stack['volumes [s]'].size)
    else:
        print('perform motion correction on a single plane/max projection')
        refImgFilt = gaussian_filter(refImage, sigma=sigmaval)

        shift = np.zeros((2, stack['volumes [s]'].size))
        error = np.zeros(stack['volumes [s]'].size)
        diffphase = np.zeros(stack['volumes [s]'].size)

    # compute for various conditions
    for i in range(stack['volumes [s]'].size):
        
        #3d reference image
        if len(refImage.shape) == 3:
            
            #computing x-y shift for each plane seperately
            if not inZ:
                for p in range(stack['planes [µm]'].size):
                    shifImg = stack[i,p,:,:]
                    shifImgFilt = gaussian_filter(shifImg, sigma=sigmaval)

                    # compute shift
                    shift[:,p,i], error[p,i], diffphase[p,i] = phase_cross_correlation(refImgFilt[p,:,:].data, shifImgFilt,
                                                                                 upsample_factor = upsampleFactor, normalization=None, reference_mask=mask)
            
            #computing x-y-z shift for the entire volume
            else:
                shifImg = stack[i,:,:,:]
                shifImgFilt = gaussian_filter(shifImg, sigma=max([shifImg.shape[0]/2,2])) #planes above and below will be smoothed
                
                
                 #compute shift
                shift[:,i], error[i], diffphase[i] = phase_cross_correlation(refImgFilt, stack[i,:,:,:], upsample_factor = upsampleFactor, normalization=None, reference_mask=mask)
                
        else:
            shifImg = stack[i,:,:]
            shifImgFilt = gaussian_filter(shifImg, sigma=sigmaval)

            # compute shift
            shift[:,i], error[i], diffphase[i] = phase_cross_correlation(refImgFilt, shifImgFilt,
                                                                         upsample_factor = upsampleFactor, normalization=None, reference_mask=mask)
        
        #progress report
        if (i+1)%(int(stack['volumes [s]'].size/10)) == 0: 
            print(".", end = " ")
            
    if showShiftFig:
        if len(refImage.shape) == 3:
            if not inZ:
                fig, axs = plt.subplots(2,1,figsize=(15,6))
                axlab = ['x','y']
                for i, ax in enumerate(axs):
                    ax.plot(shift[i,:].T)
                    ax.set_xlabel('frames')
                    ax.set_ylabel('image shift for {}'.format(axlab[i]))
            else:
                fig, axs = plt.subplots(2,1,figsize=(15,6))
                axs[0].plot(shift[1,:])
                axs[0].plot(shift[2,:])
                axs[1].plot(shift[0,:], color = 'k')
                axs[0].set_xlabel('frames')
                axs[1].set_xlabel('frames')
                axs[0].set_ylabel('shift [px]')
                axs[1].set_ylabel('shift [planes]')
                
        else:
            fig, ax = plt.subplots(1,1,figsize=(15,5))
            axlab = ['x','y']
            for i in range(len(axlab)):
                ax.plot(shift[i,:])
            ax.set_xlabel('frames')
            ax.set_ylabel('image shift [px]')
            ax.legend(axlab)

    if doFilter:
        if len(refImage.shape) == 3:
            if not inZ:
                shiftFilt_x = shift[0,:,:].copy()
                shiftFilt_y = shift[1,:,:].copy()
                shiftFilt_x_interp = shiftFilt_x
                shiftFilt_y_interp = shiftFilt_y
                for p in range(stack['planes [µm]'].size):
                    shiftFilt_x[p,abs(shiftFilt_x[p,:]-np.mean(shiftFilt_x[p,:])) > stdFactor*np.std(shiftFilt_x.flatten())] = np.nan
                    shiftFilt_y[p,abs(shiftFilt_y[p,:]-np.mean(shiftFilt_y[p,:])) > stdFactor*np.std(shiftFilt_y.flatten())] = np.nan

                    allT = np.arange(len(shiftFilt_x[p,:]))
                    shiftFilt_x_interp[p,:] = np.interp(allT, allT[~np.isnan(shiftFilt_x[p,:])], shiftFilt_x[p,~np.isnan(shiftFilt_x[p,:])])
                    shiftFilt_y_interp[p,:] = np.interp(allT, allT[~np.isnan(shiftFilt_y[p,:])], shiftFilt_y[p,~np.isnan(shiftFilt_y[p,:])])

                    if showShiftFig & np.sum(abs(shiftFilt_x[p,:]) > stdFactor*np.std(shiftFilt_x.flatten()))>0:
                        axs[0].plot(shiftFilt_x_interp[p,:], linestyle='dashed')
                        axs[1].plot(shiftFilt_y_interp[p,:], linestyle='dashed')

                shift = np.zeros((2,shiftFilt_x_interp.shape[0],shiftFilt_x_interp.shape[1]))
                shift[0,:,:] = shiftFilt_x_interp
                shift[1,:,:] = shiftFilt_y_interp
                return shift
            else:
                #inZ is true
                if stdFactorZ is None: 
                    stdFactorZ = stdFactor
                shiftFilt_z = shift[0,:].copy()
                shiftFilt_x = shift[1,:].copy()
                shiftFilt_y = shift[2,:].copy()
                shiftFilt_z[abs(shiftFilt_z-np.mean(shiftFilt_z)) > stdFactorZ*np.std(shiftFilt_z)] = np.nan
                shiftFilt_x[abs(shiftFilt_x-np.mean(shiftFilt_x)) > stdFactor*np.std(shiftFilt_x)] = np.nan
                shiftFilt_y[abs(shiftFilt_y-np.mean(shiftFilt_y)) > stdFactor*np.std(shiftFilt_y)] = np.nan
                
                allT = np.arange(len(shiftFilt_x))
                shiftFilt_z_interp = np.interp(allT, allT[~np.isnan(shiftFilt_z)], shiftFilt_z[~np.isnan(shiftFilt_z)])
                shiftFilt_x_interp = np.interp(allT, allT[~np.isnan(shiftFilt_x)], shiftFilt_x[~np.isnan(shiftFilt_x)])
                shiftFilt_y_interp = np.interp(allT, allT[~np.isnan(shiftFilt_y)], shiftFilt_y[~np.isnan(shiftFilt_y)])

                if showShiftFig:
                    axs[0].plot(shiftFilt_x_interp,'b')
                    axs[0].plot(shiftFilt_y_interp,'c')
                    axs[1].plot(shiftFilt_z_interp,'r')

                return np.vstack((shiftFilt_z_interp,shiftFilt_x_interp,shiftFilt_y_interp))
        
        else:
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

    #check if shift was calculated for each plane in a volume separately, 
    #then check if stack to be aligned is 3d or 4d

    #stack should be an xarray
    stackMC = stack.copy()

    if len(shift.shape) == 3:
        # separate shifts for each plane in a volume
        assert len(stack.shape)==4, "Imaging stack needs to be 4D"
        
        for i in range(stack['volumes [s]'].size):
            for p in range(stack['planes [µm]'].size):
                shifImg = stack[i,p,:,:]
                stackMC[i,p,:,:] = spshift(shifImg, shift[:,p,i], order=1,mode='reflect')
                
            #progress report
            if (i+1)%(int(stack['volumes [s]'].size/10)) == 0: print(".", end = " ")
            
    else:
        #one shift per volume per time point
        
        if shift.shape[0] == 3:
            #shift has 3 axes
            assert len(stack.shape)==4, "Imaging stack needs to be 4D"
        
        # motion correction on max projection in 2 axes or entire volume in 3 axes
        for i in range(stack['volumes [s]'].size):
            shifImg = stack[i,:,:]
            stackMC[i,:,:] = spshift(shifImg, shift[:,i], order=1,mode='reflect')
            
            #progress report
            if (i+1)%(int(stack['volumes [s]'].size/10)) == 0: print(".", end = " ")
            
    return stackMC

def notchFilt1dRipple(img, rippleW0=0.1326, Q=0.75):
    
    import scipy as sp
    b, a = sp.signal.iirnotch(rippleW0, Q)
    
    filtImg = img.copy()
    
    for line in range(img.shape[1]):
        filtImg[:,line] = sp.signal.filtfilt(b, a, img[:,line])
        
    return filtImg

def refFilterRipple(refVol, **kwrds):
    filtRef = refVol.copy()
    for plane in range(refVol.shape[0]):
        filtRef[plane,:,:] = notchFilt1dRipple(refVol[plane,:,:], **kwrds)
    return filtRef

def computeNotchFilter2d(img,center_frac=1/100):
    from skimage.draw import disk
    
    img_shape = img.shape #get shape
    f = np.fft.fft2(img) #fourier transform
    fshift = np.fft.fftshift(f)
    
    phase_spectrumR = np.angle(fshift)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    
    center = np.zeros(img_shape, dtype=np.uint8)
    rr, cc = disk((int(img_shape[0]/2),int(img_shape[0]/2)), img_shape[0]*center_frac, shape=img_shape)
    center[rr, cc] = True
    NotchFilter = (magnitude_spectrum<=(np.median(magnitude_spectrum)+0.5*np.std(magnitude_spectrum))) | center
    return NotchFilter

def notchFilter2d(refVol, plane=-1, center_frac=1/100):
    NotchFilter = computeNotchFilter2d(refVol[plane,:,:], center_frac=center_frac)
    filtVol = refVol.copy()
    
    for p in range(refVol.shape[0]):
        f = np.fft.fftshift(np.fft.fft2(refVol[p,:,:])) 
        NotchRejectCenter = f * NotchFilter 
        NotchReject = np.fft.ifftshift(NotchRejectCenter)
        filtVol[p,:,:] = np.abs(np.fft.ifft2(NotchReject))  # Compute the inverse DFT of the result
    
    return filtVol