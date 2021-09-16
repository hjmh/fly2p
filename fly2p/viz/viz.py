from matplotlib import pyplot as plt
import numpy as np
import matplotlib.patches as ppatch

# General ......................................................................
## axis beautification
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

# Calcium traces vizualization .................................................
def plotDFFheatmap(time, roidat, ax, fig):
    cax = ax.pcolor(time,np.arange(0,roidat.shape[0]+1),roidat,cmap='Blues', edgecolors='face',shading='auto')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('\nROIs (n = {0})'.format(roidat.shape[0]))

    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax)
    cbar.set_label('$(F - F_0) / F_0$ (per ROI)')  # vertically oriented colorbar

    return ax

# ROI vizualization ............................................................

## plot shapely objects over reference image
def plotShapelyLine(ax, shapelyLine, labelStr, col):
    ax.plot(shapelyLine.coords.xy[0], shapelyLine.coords.xy[1], color=col)
    ax.text(shapelyLine.coords[0][0], shapelyLine.coords[0][1]+2, labelStr, color=col, fontsize=12)
    return ax

def plotShapelyPoly(ax, shapelyPoly, labelStr,col):
    ax.plot(shapelyPoly.coords.xy[0],EBroiRing.coords.xy[1], color=col)
    ax.text(shapelyPoly.coords[0][0], EBroiRing.coords[0][1], labelStr, color=col, fontsize=12)
    ax.plot(shapelyPoly.centroid.coords.xy[0],EBroiRing.centroid.coords.xy[1],'o')
    return ax

def plotEllipse(ax, ctr, longax, shortax, ellipseRot,col, alphaval):
    ellipsepatch = ppatch.Ellipse(ctr, longax, shortax, -ellipseRot, alpha=alphaval, color=col)
    ax.add_patch(ellipsepatch)
    return ax


## EB specific
### Show position of shapely ROIs
def plotEBshapelyROIs(refEBimg, ebcenter, EBaxisL, EBaxisS, ellipseRot, EBoutline, EBroiPts, EBroiPolys):
    from fly2p.viz.viz import myAxisTheme, plotShapelyLine, plotEllipse
    from shapely.geometry.polygon import Polygon

    fig, axs = plt.subplots(1,3, figsize=(15,5))

    for ax in axs:
        ax.imshow(refEBimg,cmap='Greys_r', origin='lower')
        ax.axis('off');

    axs[0].plot(ebcenter[0],ebcenter[1], 'ro')
    axs[0] = plotShapelyLine(axs[0], EBaxisL, 'Long axis', 'orange')
    axs[0] = plotShapelyLine(axs[0], EBaxisS, 'Short axis', 'tomato')

    axs[1] = plotEllipse(axs[1], ebcenter, EBaxisL.length, EBaxisS.length, ellipseRot
                         ,'turquoise', 0.4)
    axs[1] = plotShapelyLine(axs[1], EBaxisL, 'Axis 1', 'coral')
    axs[1] = plotShapelyLine(axs[1], EBaxisS, 'Axis 2', 'orangered')
    axs[1] = plotShapelyLine(axs[1], EBoutline, '', 'crimson')

    for s in range(len(EBroiPts)-1):
        axs[1].plot(EBroiPts[s+1][0],EBroiPts[s+1][1], 'wo')
        axs[1].text(EBroiPts[s+1][0]+2,EBroiPts[s+1][1]+1, str(s+1), color='w')
    axs[1].plot(ebcenter[0],ebcenter[1],'wo')

    axs[2].plot(EBoutline.coords.xy[0],EBoutline.coords.xy[1], color='crimson')
    for s in range(len(EBroiPts)-1):
        roiPatch = ppatch.Polygon(EBroiPolys[s],alpha=0.5, edgecolor='turquoise', facecolor='teal')
        axs[2].add_patch(roiPatch)
        labcoord = Polygon(EBroiPolys[s]).centroid.coords.xy
        axs[2].text(labcoord[0][0],labcoord[1][0], str(s+1), color='w')
        axs[2].plot(EBroiPts[s+1][0],EBroiPts[s+1][1], 'w.')
    axs[2].plot(EBroiPts[0][0],EBroiPts[0][1], 'w.')

    return fig
