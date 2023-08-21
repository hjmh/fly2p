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
    cax = ax.pcolor(time,np.arange(0,roidat.shape[0]),roidat,cmap='Blues', edgecolors='face',shading='auto')
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

def makeEllipse(centerpt, longax, shortax, rotation):
    import shapely as sp
    # 1st elem = center point (x,y) coordinates
    # 2nd elem = the two semi-axis values (along x, along y)
    # 3rd elem = angle in degrees between x-axis of the Cartesian base
    #            and the corresponding semi-axis
    ellipse = (centerpt,(longax/2., shortax/2.),rotation)

    # Let create a circle of radius 1 around center point:
    circ = sp.geometry.Point(ellipse[0]).buffer(1)

    # Let create the ellipse along x and y:
    ell  = sp.affinity.scale(circ, int(ellipse[1][0]), int(ellipse[1][1]))

    # Let rotate the ellipse (counterclockwise, x axis pointing right):
    ell = sp.affinity.rotate(ell,-ellipse[2])
    # According to the man, a positive value means a anti-clockwise angle,
    # and a negative one a clockwise angle.

    return ell.exterior


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

def generateEBellipse(longax,shortax,center,printResults = False):
    from shapely.geometry.polygon import LineString
    EBaxisL = LineString(longax)
    EBaxisS = LineString(shortax)

    if printResults:
        print('EB center coordinates (px): {0}'.format(center))
        print('EB axis lengths:  axis 1 = {0}, axis 2 = {1}'.format(round(EBaxisL.length/2.), round(EBaxisS.length/2.)))

    axvec = abs(np.asarray(EBaxisL.coords[0])-np.asarray(EBaxisL.coords[1]))
    ellipseRot = 90-np.arctan(axvec[0]/axvec[1])*180/np.pi

    if printResults: print('EB main axis rotation (deg): {0}'.format(round(ellipseRot)))

    EBoutline = makeEllipse(center, EBaxisL.length, EBaxisS.length, ellipseRot)

    return EBaxisL, EBaxisS, ellipseRot, center, EBoutline


def constructEBROIs(center, outline, nsteps=16, st=3):
    # Use st to shift ROI pts circularely to start at ventral part of EB
    EBroiPts = [center]

    for s in range(nsteps):
        [sx,sy] = outline.interpolate(s/nsteps, normalized=True).coords.xy
        EBroiPts.append((sx[0],sy[0]))

    EBroiPtsCopy = EBroiPts.copy()

    for s in range(1,nsteps+1):
        EBroiPts[s] = EBroiPtsCopy[(s+st)%nsteps+1]

    EBroiPolys = []
    for s in range(nsteps):
        if s+1==nsteps:EBroiPolys.append([EBroiPts[0],EBroiPts[s+1],EBroiPts[1]])
        else: EBroiPolys.append([EBroiPts[0],EBroiPts[s+1],EBroiPts[s+2]])

    return EBroiPts, EBroiPolys


def getDFFfromEllipseROI(EBroiPts,EBroiPolys, dffXarray):
    from shapely.geometry.polygon import Polygon
    nsteps = len(EBroiPts)-1

    # create a list of possible pixel coordinates
    imgrid = np.meshgrid(np.arange(0,dffXarray['xpix [µm]'].size), np.arange(0,dffXarray['ypix [µm]'].size))
    pxcoords = list(zip(*(c.flat for c in imgrid)))

    # Get all points in EB ellipse
    EBPatch = ppatch.Polygon(EBroiPts[1:])
    roiPts_x = [p[0] for p in pxcoords if EBPatch.contains_point(p, radius=0)]
    roiPts_y = [p[1] for p in pxcoords if EBPatch.contains_point(p, radius=0)]

    EBCoords = np.vstack((roiPts_x,roiPts_y))

    # List for all roi point lists
    EBroiCoords = []
    for s in range(nsteps):
        roiPatch = ppatch.Polygon(EBroiPolys[s])

        # create the list of valid coordi nates (from untransformed)
        roiPts_x = [p[0] for p in pxcoords if roiPatch.contains_point(p, radius=0)]
        roiPts_y = [p[1] for p in pxcoords if roiPatch.contains_point(p, radius=0)]
        EBroiCoords.append(np.vstack((np.asarray(roiPts_x),np.asarray(roiPts_y))))

    dffROI = np.zeros((nsteps,dffXarray.data.shape[0]))
    for s in range(nsteps):
        dffROI[s,:] = dffXarray.data[:,EBroiCoords[s][0],EBroiCoords[s][1]].mean(1)

    return dffROI
