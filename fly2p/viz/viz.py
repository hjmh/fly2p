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
def plotDFFheatmap(time, roidat, ax):
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
