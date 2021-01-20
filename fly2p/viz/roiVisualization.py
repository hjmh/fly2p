"""Functions related to visulizing ROIs drawn in Ca-imaging data

Some functions assume that ROI data is saved in dictionary (as generated in 'AnalyzeImaging' notebook)

roiData = {
    'imgData': rawtiff,
    'img': refimg,
    'numframes': slct_numframes,
    'slctframes': slct_frames,
    'fpv': fpv
    'numRoi': len(rois)
    'Fraw': time series of raw average of points in roi
    'DFF': time series of delta F/F in roi
    'Pts: points defining outline of roi
}

install shapely (pip install shapely) and descartes (pip install descartes) 

"""

from matplotlib import pyplot as plt
import numpy as np

## General
# axis beautification
def myAxisTheme(myax):
    myax.get_xaxis().tick_bottom()
    myax.get_yaxis().tick_left()
    myax.spines['top'].set_visible(False)
    myax.spines['right'].set_visible(False)

def plotScaleBar(ax,xlen,pos,labeltext):
    ax.plot([pos[0],pos[0]+xlen],[pos[1],pos[1]],'k')
    ax.text(pos[0],pos[1],labeltext)
    
def minimalAxisTheme(myax, xlen,pos,labeltext):
    plotScaleBar(myax,xlen,pos,labeltext)
    myax.axis('off')
    myax.set_aspect('equal')

def pathPlotAxisTheme(myax, units):
    myax.spines['top'].set_visible(False)
    myax.spines['right'].set_visible(False)
    myax.spines['bottom'].set_visible(False)
    myax.spines['left'].set_visible(False)
    myax.get_xaxis().set_ticks([])
    myax.get_yaxis().set_ticks([])
    myax.set_aspect('equal')
    myax.set_xlabel('x [{}]'.format(units))
    myax.set_ylabel('y [{}]'.format(units))

    
## Plotting to be used with ROIdat in format described above
def illustrateRoiOutline(roidat, ax, roicmap, title):
    from shapely.geometry.polygon import LinearRing
    
    ax.imshow(roidat['img'].T,cmap='Greys_r', vmin=0, origin='upper')
    
    for i, roi in enumerate(roidat['Pts']):
        roiOutl = LinearRing(roi)
        xr, yr = roiOutl.coords.xy
        ax.plot(xr, yr, color=roicmap.to_rgba(i), lw=2)
        ax.text(xr[0], yr[0], 'roi'+str(i+1), color='w', fontsize=10)
    
    ax.set_title(title)
    return ax

def illustrateRoiArea(roidat, ax, roicmap, title):
    from shapely.geometry.polygon import Polygon
    from descartes import PolygonPatch
    
    ax.imshow(roidat['img'].T,cmap='Greys_r', vmin=0, origin='upper')
    
    for i, roi in enumerate(roidat['Pts']):
        roiArea = Polygon(roi)
        
        roipatch = PolygonPatch(roiArea,alpha=0.7, color=roicmap.to_rgba(i))
        ax.add_patch(roipatch)
        
        ax.text(roi[0,0], roi[0,1], 'roi'+str(i+1), color='w', fontsize=10)
    ax.set_title(title)
    return ax


def illustrateRoiTrace(roidat, ax, roicmap, framerange, xlab, ylab, title):
    
    for i in range(roidat['numRoi']):
        ax.plot(roidat['DFF'][framerange,i],'-', color=roicmap.to_rgba(i))
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    
    ax.set_title(title)
    return ax


## Plotting functions to be used with roi data generated with Dan's matlab scripts

def roiPlot(fig, ax, roidat, roiNames, time):
    cax = ax.pcolor(time,np.arange(0,len(roiNames)+1),
                     roidat,cmap='Blues', edgecolors='face')
    ax.set_xlabel('Time [s]')
    ax.set_yticks(np.arange(0.5,0.5+len(roiNames)))
    ax.set_yticklabels(roiNames)
    ax.set_ylabel('\nROIs (n = {})'.format(len(roiNames)))
    myAxisTheme(ax)
    
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax)
    cbar.set_label('$(F - F_0) / F_0$ (per ROI)')  # vertically oriented colorbar

# obsolte?
def plotDFFROIs (axs, numROIs, ROIdat, roitime):
    cax = axs.pcolor(roitime,np.arange(0,numROIs+1),ROIdat,cmap='Blues', edgecolors='face')
    axs.set_xlabel('Time [s]')
    axs.set_ylabel('\nROIs (n = {0})'.format(numROIs))
    myAxisTheme(axs)

    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax)
    cbar.set_label('$(F - F_0) / F_0$ (per ROI)')  # vertically oriented colorbar
    

def plotDFF_PVA_heading_fwdvelo(axs, vrDat, ROIdatMatch, pvaMatch, vTransFilt, numROIs):
    cax = axs[0].pcolor(vrDat['time'],np.arange(0,numROIs+1),ROIdatMatch[2:],cmap='Blues', edgecolors='face')
    axs[0].set_ylabel('\nROIs (n = {0})'.format(numROIs))
    axs[0].plot(vrDat['time'],pvaMatch['pvaROI'],'-', color='black', linewidth=1, alpha=0.5, label='PVA direction')
    axs[0].legend(loc=1)

    axs[1].plot(vrDat['time'],pvaMatch['pvaRad'],color='steelblue', label='PVA direction')
    axs[1].plot(vrDat['time'],vrDat['heading'],color='black', label='absolute heading')
    axs[1].set_ylabel('Radians')
    axs[1].legend(loc=1)

    NOsignal = np.sum(ROIdatMatch[:2], axis=0)
    axs[2].plot(vrDat['time'],NOsignal,color='steelblue', label='NO signal')
    axs[2].plot(vrDat['time'],vTransFilt,color='black', label='filt transl. velocity')
    #axs[2].set_ylim(0,1.5)
    axs[2].set_ylabel('Trans. velocity [cm/s] (black)\nNO signal (DFF)')
    axs[2].set_xlabel('Time [s]')
    
    for ax in axs:
        myAxisTheme(ax)
        ax.set_xlim(vrDat['time'][0],vrDat['time'][-1])
        
# OBSOLETE
def plotPVAerror(axs, vrDat, pvaMatch):
    # align PVA and heading  ## TODO: Replace with correlation analysis...
    pvaAligned = np.mod(np.unwrap(pvaMatch['pvaRad']) - pvaMatch['pvaRad'][0]+np.pi, 2*np.pi)-np.pi
    headingAligned = np.mod(np.unwrap(vrDat['heading']) - vrDat['heading'][0] + np.pi, 2*np.pi)-np.pi

    error = abs(headingAligned-pvaAligned)
    errorUW = abs((np.unwrap(vrDat['heading'])-vrDat['heading'][0]) - (np.unwrap(pvaMatch['pvaRad'])-pvaMatch['pvaRad'][0]))

    axs[0].plot(vrDat['time'],pvaAligned,'-', color='royalblue', label='PVA')
    axs[0].plot(vrDat['time'],headingAligned,'-', color='black', label='heading')

    axs[1].plot(vrDat['time'],np.unwrap(pvaAligned),'-', color='royalblue', label='PVA')
    axs[1].plot(vrDat['time'],np.unwrap(headingAligned),'-', color='black', label='heading')
    axs[1].legend(loc=0)

    axs[2].plot(vrDat['time'],error,'-', color='grey', label='error on wrapped')
    axs[2].plot(vrDat['time'],errorUW,'-', color='midnightblue', label='error on unwrapped')
    axs[2].legend(loc=0)

    for ax in axs:
        myAxisTheme(ax)
        ax.set_xlim(vrDat['time'][0], vrDat['time'][-1])