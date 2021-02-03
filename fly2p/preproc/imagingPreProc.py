import numpy as np
from scipy.signal import savgol_filter
from os.path import sep


### Pure python image processing pipeline

# extract meta data from scan image  tiff file header

#--> old and will be deleted soon <--
def metadatFromTiff(dataDir, tiffName, minLine = 10):
    lc = 0 #line count

    with open(dataDir +sep+ tiffName, 'rb') as fh:
        endofhead = 0 
        fpv = -1

        while(not endofhead):
            line = str(fh.readline()[:-1])

            lc += 1

            # extract version
            if 'VERSION_' in line: print(line)

            # get channel info
            if 'channelSave' in line:
                print(line)
                lineString = str(line)
                if not '[' in lineString:
                    nCh = 1
                else:
                    nCh = int(lineString.split('[')[-1][-2])

            # get number of planes per z-stack
            #if 'hStackManager' in line:
            #    if 'numFramesPerVolume' in line:
            #        print(line)
            #        fpv = int(line[-2:-1]) 

            if 'scanFrameRate' in line:
                print(line)
                lineString = str(line)
                fpsscan = float(lineString[lineString.find('=')+1:-1])

            if 'hFastZ' in line:
                if 'discardFlybackFrames' in line:
                    print(line)
                    lineString = str(line)
                    discardFBFrames = lineString[lineString.find('=')+1:-1]

                if 'numDiscardFlybackFrames' in line:
                    print(line)
                    lineString = str(line)
                    nDiscardFBFrames = int(lineString[lineString.find('=')+1:-1])

                if 'numFramesPerVolume' in line:
                    print(line)
                    lineString = str(line)
                    fpv = int(lineString[lineString.find('=')+1:-1])

                if 'numVolumes' in line:
                    print(line)
                    lineString = str(line)
                    nVols = int(lineString[lineString.find('=')+1:-1])

            if not 'SI' in line and lc > minLine:
                endofhead = 1
    print(' # channels: {}\n fly back? {}\n # discard frames: {}\n # frames/volume: {}\n # volumes: {}'.\
          format(nCh, discardFBFrames, nDiscardFBFrames, fpv, nVols))
    
    return nCh, discardFBFrames, nDiscardFBFrames, fpsscan, fpv, nVols


## Motion correction
#--> old and will be deleted soon <--
def motionCorrSinglePlane(stackMP, refImg, upsampleFactor, gaussianFiltRef = False):
    
    from skimage.feature import register_translation
    from scipy.ndimage import fourier_shift
    from scipy.ndimage.filters import gaussian_filter

    shift = np.zeros((2, stackMP.shape[0]))
    error = np.zeros(stackMP.shape[0])
    diffphase = np.zeros(stackMP.shape[0])

    stackMPMC = np.ones(stackMP.shape).astype('int16')
    
    if gaussianFiltRef:
        refImg = gaussian_filter(refImg, sigma=2)
        
    for i in range(stackMP.shape[0]):
        shifImg = stackMP[i,:,:]

        # subpixel precision
        shift[:,i], error[i], diffphase[i] = register_translation(refImg, shifImg, upsampleFactor)

        offset_image = fourier_shift(np.fft.fftn(shifImg), shift[:,i])
        stackMPMC[i,:,:] = np.fft.ifftn(offset_image).real.astype('uint16')
        
    return stackMPMC




### For mixed Matlab/python pipeline (based on Dan's image processing)

def interpFrames(rawDat, FG, fpv, t, framegrab, tVR):

    interpDat = np.zeros((FG[1]-FG[0], 1))
    
    for i in range(FG[0],FG[1]):
        
        iMatch = np.where( t >= ( t[0] + (framegrab[(i-1)*fpv+1] - tVR[0])/10000 ) )[0][0]
        interpDat[i - FG[0]] = rawDat[iMatch]
    
    return interpDat


# Parse preprocessed imaging data and align treadmill and imaging data
def parseMatPosdat(matdat, fpv):
    
    # Parse matlab data
    tmtime = np.ndarray.flatten(matdat[0,0][0])
    vrtime = np.ndarray.flatten(matdat[0,0][15][0])
    tFrameGrab = np.ndarray.flatten(matdat[0,0][14][0])

    vrOffsets = np.zeros((3, len(tmtime))) # offsetRot, offsetFwd, offsetLat
    for i in range(3):
        vrOffsets[i,:] = np.ndarray.flatten(matdat[0,0][i+1])
    
    tmdeltas = np.zeros((4, len(tmtime))) #dx0, dx1, dy0, dy1
    for i in range(4):
        tmdeltas[i,:] = np.ndarray.flatten(matdat[0,0][i+4])
    
    # Align treadmill and imaging data
    minFG = np.floor(np.where(tFrameGrab >= vrtime[0])[0][0]/fpv)
    maxFG = np.round(len(tFrameGrab)/fpv)
    FG = (int(minFG), int(maxFG))
    
    time = tFrameGrab[0::fpv]/10000
    time = time[FG[0]:FG[1]]

    tmtimeMatch = np.ndarray.flatten(interpFrames(tmtime,FG, fpv, tmtime, tFrameGrab, vrtime)) 
    vrOffsetsMatch = np.zeros((3, len(time))) 
    for i in range(3):
        vrOffsetsMatch[i,:] = np.ndarray.flatten(interpFrames(vrOffsets[i,:],FG, fpv, tmtime, tFrameGrab, vrtime))

    tmdeltasMatch = np.zeros((4, len(time))) 
    for i in range(4):
        tmdeltasMatch[i,:] = np.ndarray.flatten(interpFrames(tmdeltas[i,:],FG, fpv, tmtime, tFrameGrab, vrtime))

    # Compute velocities
    vRot = np.hstack((0,np.diff(vrOffsetsMatch[0,:])/np.diff(time)))*np.pi/180.
    vFwd = np.hstack((0,np.diff(vrOffsetsMatch[1,:])/np.diff(time)))
    vLat = np.hstack((0,np.diff(vrOffsetsMatch[2,:])/np.diff(time)))

    vTrans = np.hypot(vrOffsetsMatch[1,:], vrOffsetsMatch[2,:])*10
    vTrans = np.hstack((0,abs(np.diff(vTrans))/np.diff(time)))

    #package in dictionary
    posDat = {
        'time': time,
        'tmtime': tmtimeMatch,
        'heading': vrOffsetsMatch[0,:]*np.pi/180.,
        'xpos': vrOffsetsMatch[1,:],
        'ypos': vrOffsetsMatch[2,:],
        'vRot': vRot,
        'vFwd': vFwd,
        'vLat': vLat,
        'vTrans': vTrans
    }
    
    return posDat, tmdeltasMatch, FG


def filterVelos(vrDat, window, order):

    vTransFilt= savgol_filter(vrDat['vTrans'], window, order)
    vRotFilt= savgol_filter(vrDat['vRot'], window, order)
    vFwdFilt= savgol_filter(vrDat['vFwd'], window, order)
    vLatFilt= savgol_filter(vrDat['vLat'], window, order)
    #osRotMatch = np.hstack((osRotMatch[0], np.diff(savgol_filter(np.cumsum(osRotMatch), window, order))))

    return vTransFilt, vRotFilt, vFwdFilt, vLatFilt

#
def computePVA (locs, weights):
    """ Compute population vector average
    """
    nsteps = weights.shape[0]
    nvol = weights.shape[1]
    pva_x = np.cos(np.reshape(np.tile(locs,nvol),[nvol,nsteps])).T*weights
    pva_y = np.sin(np.reshape(np.tile(locs,nvol),[nvol,nsteps])).T*weights
    
    pva = np.vstack((sum(pva_x)/len(pva_x), sum(pva_y)/len(pva_x)))
    return pva