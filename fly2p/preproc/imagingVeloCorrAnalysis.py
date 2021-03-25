import numpy as np
import scipy.stats as sts

def myxcorr(x,y,lags):
    cc = np.zeros((len(lags)))
    for i, lag in enumerate(lags):
        if lag < 0:
            cc[i] = np.correlate(x[:lag], y[-lag:])
        elif lag == 0:
            cc[i] = np.correlate(x, y)
        else:
            cc[i] = np.correlate(x[lag:], y[:-lag])
    return cc

def myxcorrCoeff(x,y,lags):
    cc = np.zeros((len(lags)))
    for i, lag in enumerate(lags):
        if lag < 0:
            cc[i] = sts.pearsonr(x[:lag], y[-lag:])[0]
        elif lag == 0:
            cc[i] = sts.pearsonr(x, y)[0]
        else:
            cc[i] = sts.pearsonr(x[lag:], y[:-lag])[0]
    return cc

def myxcorrCoeffPval(x,y,lags):
    cc = np.zeros((len(lags)))
    pvals = np.zeros((len(lags)))
    for i, lag in enumerate(lags):
        if lag < 0:
            cc[i] = sts.pearsonr(x[:lag], y[-lag:])[0]
            pvals[i] = sts.pearsonr(x[:lag], y[-lag:])[1]
        elif lag == 0:
            cc[i] = sts.pearsonr(x, y)[0]
            pvals[i] = sts.pearsonr(x, y)[1]
        else:
            cc[i] = sts.pearsonr(x[lag:], y[:-lag])[0]
            pvals[i] = sts.pearsonr(x[lag:], y[:-lag])[1]
    return cc, pvals