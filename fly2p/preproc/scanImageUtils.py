import json


# Extracting metadata from ScanImage tiff files

def getSIbasicMetadata(metadat):
    for i, line in enumerate(metadat.split('\n')):
        if not 'SI.' in line: continue
        # extract version
        if 'VERSION_' in line: print(line)

        # get channel info
        if 'channelSave' in line:
            print(line)
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
            
    metadict = {
        "nCh": nCh,
        "fpsscan": fpsscan,
        "nVols": nVols,
        "fpv": fpv,
        "discardFBFrames": discardFBFrames,
        "nDiscardFBFrames": nDiscardFBFrames
    }
            
    return metadict


def getSIMetadict(metadat):
    matches = [line for line in metadat.split('\n') if not 'SI.' in line]
    m = '\n'.join(matches[1:-1])
    SImetadict = json.loads(m)

    roiGroups = SImetadict['RoiGroups']
    #print(roiGroups.keys())
    #print(json.dumps(roiGroups['imagingRoiGroup']['rois']['UserData'],indent=4))
    return SImetadict