import json
import numpy as np


# Utility functions for working with ScanImage tiff files

## EXTRACT METADATA ##
def getSIbasicMetadata(metadat):
    for i, line in enumerate(metadat.split('\n')):
        
        #initilize dict
        metadict = {}
            
        if not 'SI.' in line: continue
        # extract version
        if 'VERSION_' in line: print(line)

        # get channel info
        if 'channelSave' in line:
            #print(line)
            if not '[' in line:
                nCh = 1
            else:
                nCh = int(line.split('=')[-1].strip())
                
        if 'scanFrameRate' in line:
            fpsscan = float(line.split('=')[-1].strip())

            
        #if 'hFastZ' in line:
        if 'discardFlybackFrames' in line:
            discardFBFrames = line.split('=')[-1].strip()

        if 'numDiscardFlybackFrames' in line:
            nDiscardFBFrames = int(line.split('=')[-1].strip())

        if 'numFramesPerVolume' in line:
            fpv = int(line.split('=')[-1].strip())


        if 'numVolumes' in line:
            nVols = int(line.split('=')[-1].strip())
            
        if 'hStackManager.stackZStepSize' in line:
            stackZStepSize = int(line.split('=')[-1].strip())
            
        if 'hRoiManager.scanVolumeRate' in line:
            scanVolumeRate = float(line.split('=')[-1].strip())
                    
        if 'SI.hRoiManager.imagingFovUm' in line:
            imagingFovUm = line.split('=')[-1].strip()
            p00 = np.fromstring(imagingFovUm[1:-1].split(';')[0], dtype=float, count=2, sep=' ')
            p10 = np.fromstring(imagingFovUm[1:-1].split(';')[1], dtype=float, count=2, sep=' ')
            p01 = np.fromstring(imagingFovUm[1:-1].split(';')[2], dtype=float, count=2, sep=' ')
            p11 = np.fromstring(imagingFovUm[1:-1].split(';')[3], dtype=float, count=2, sep=' ')

    metadict["nCh"] = nCh
    metadict["fpsscan"] = fpsscan
    metadict["discardFBFrames"] = discardFBFrames
    metadict["nDiscardFBFrames"] = nDiscardFBFrames
    metadict["fpv"] = fpv
    metadict["nVols"] = nVols
    metadict["stackZStepSize"] = stackZStepSize
    metadict["scanVolumeRate"] = scanVolumeRate
    metadict["fovCoords"] = [p00,p01,p10,p11]
    metadict["xrange_um"] = p01[0]-p00[0]
    metadict["yrange_um"] = p11[1]-p00[1]

    return metadict


def getSIMetadict(metadat):
    matches = [line for line in metadat.split('\n') if not 'SI.' in line]
    m = '\n'.join(matches[1:-1])
    SImetadict = json.loads(m)

    roiGroups = SImetadict['RoiGroups']
    #print(roiGroups.keys())
    #print(json.dumps(roiGroups['imagingRoiGroup']['rois']['UserData'],indent=4))
    return SImetadict



## LOAD AND RESHAPE IMAGE VOLUME ##

def loadvolume(mytiffreader, basicMetadat, selectCaChan):
    vol = mytiffreader.data()
    vol = vol.reshape((int(vol.shape[0]/(basicMetadat['fpv']*basicMetadat['nCh'])),
                       basicMetadat['fpv'],basicMetadat['nCh'],vol.shape[1], vol.shape[2]))
    # Full dimensional stack: volumes, planes, channels, xpix, ypix'
    
    if (selectCaChan):
        # Stack reduced to one color channel and flyback frames discrded
        vol = vol[:,0:basicMetadat['fpv']-basicMetadat['nDiscardFBFrames'],basicMetadat['CaCh'],:,:]

    return vol