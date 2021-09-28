import json
import numpy as np
from skimage import io

# Utility functions for working with image files from micromanager

## EXTRACT METADATA ##
def getBasicMetadata(path2meta):

    with open(path2meta) as f:
        metadat = json.load(f)
    metadat = metadat['Summary']

    #initilize dict
    metadict = {}

    metadict["nCh"] = metadat['Channels']
    metadict["fpsscan"] = 1000/metadat['Interval_ms']
    metadict["discardFBFrames"] = 0
    metadict["nDiscardFBFrames"] = 0
    metadict["fpv"] = metadat['IntendedDimensions']['z']
    metadict["nVols"] = metadat['IntendedDimensions']['time']
    metadict["stackZStepSize"] = metadat['z-step_um']
    metadict["scanVolumeRate"] = 1000/metadat['Interval_ms']
    metadict["fovCoords"] = -1
    metadict["xrange_um"] = metadat['Width']
    metadict["yrange_um"] = metadat['Height']

    return metadict

## LOAD AND RESHAPE IMAGE VOLUME ##
def loadvolume(path2tif, basicMetadat=None, selectCaChan=False):
    # read the image stack
    vol = io.imread(path2tif)
    if basicMetadat!=None:
        vol = vol.reshape((int(vol.shape[0]/(basicMetadat['fpv']*basicMetadat['nCh'])),
                           basicMetadat['fpv'],basicMetadat['nCh'],vol.shape[1], vol.shape[2]))
        # Full dimensional stack: volumes, planes, channels, xpix, ypix'

    if (selectCaChan):
        # Stack reduced to one color channel and flyback frames discrded
        vol = vol[:,0:basicMetadat['fpv']-basicMetadat['nDiscardFBFrames'],basicMetadat['CaCh'],:,:]

    return vol
