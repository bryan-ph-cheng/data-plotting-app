import numpy as np
from plotly import graph_objs as go


def normalizeCounts(countArray, num):
    reb_counts = np.squeeze(countArray)/np.mean(np.sort(np.squeeze(countArray))[-num:])
    return reb_counts

def plot_data(expt, files, normalize, plotcurrent=0, nPi = []):
    "Attribute - expt - can either be 'counting' or 'odmr' or 'pulsedodmr' or 'rabi' or 'ramsey' or 'spinecho' or 'doubleecho' or 'nmr'.\
    If value of plotcurrent is 1 then it will plot the current data. If value of normalize is 1 then it will normalize 'odmr' and 'doubleecho' signal."
    if plotcurrent == 1:
        filesize = np.size(files) + 1
    else:
        filesize = np.size(files)
    plotfun = go.Figure()
    for i in range(filesize):
        if plotcurrent == 1 and i == filesize-1:
            Data2 = exc.load_last_experiment()
            Data2 = Data2.last_data_set()
        else:
            Data2 = exc.load_by_id(files[i])
        if expt == 'spinecho':
            x2 = 2*np.squeeze(Data2.get_data('Time'))
            y2 = np.squeeze(Data2.get_data('Rebased_Counts'))
            plotfun.layout = go.Layout(xaxis=dict(title='Time (ns)'),yaxis=dict(title='Counts'),title='Spin Echo')
        elif expt == 'doubleecho':
            if nPi == [] or nPi[i] == 1:
                x2 = 2*np.squeeze(Data2.get_data('Time'))
            else:
                x2 = nPi[i]*np.squeeze(Data2.get_data('Time'))
            y2ms0 = np.squeeze(Data2.get_data('Act_Counts'))
            y2ms1 = np.squeeze(Data2.get_data('Ref_Counts'))
            y2 = y2ms0 - y2ms1
            plotfun.layout = go.Layout(xaxis=dict(title='Time (ns)'),yaxis=dict(title='Counts'),title='Spin Echo Double Measure')
            if normalize == 1:
                y2 = (y2 + max(y2))/(2*max(y2))
        elif expt == 'nmr':
            x2 = np.squeeze(Data2.get_data('Time'))
            y2ms0 = np.squeeze(Data2.get_data('Act_Counts'))
            y2ms1 = np.squeeze(Data2.get_data('Ref_Counts'))
            y2 = y2ms0 - y2ms1
            plotfun.layout = go.Layout(xaxis=dict(title='Time (ns)'),yaxis=dict(title='Counts'),title='NMR')
            if normalize == 1:
                y2 = (y2 + max(y2))/(2*max(y2))
        elif expt == 'odmr':
            x2 = np.squeeze(Data2.get_data('Frequency'))
            y2 = np.squeeze(Data2.get_data('Counts'))
            plotfun.layout = go.Layout(xaxis=dict(title='Frequency (Hz)'),yaxis=dict(title='Counts'),title='ODMR')
            if normalize == 1:
                y2 = normalizeCounts(Data2.get_data('Counts'),50)
                layout = go.Layout(xaxis=dict(title='Frequency (Hz)'),yaxis=dict(title='Normalized Counts'),title='ODMR')
        elif expt == 'pulsedodmr':
            x2 = np.squeeze(Data2.get_data('Frequency'))
            y2 = np.squeeze(Data2.get_data('Rebased_Counts'))
            plotfun.layout = go.Layout(xaxis=dict(title='Frequency (Hz)'),yaxis=dict(title='Counts'),title='Pulsed ODMR')
        elif expt == 'g2':
            x2 = np.squeeze(Data2.get_data('Time'))
            y2 = np.squeeze(Data2.get_data('Norm_Counts'))
            plotfun.layout = go.Layout(xaxis=dict(title='Time dif'), yaxis=dict(title='Normalised Counts'), title='g2 Dip')
        else:
            x2 = np.squeeze(Data2.get_data('Time'))
            y2 = np.squeeze(Data2.get_data('Rebased_Counts'))
            plotfun.layout = go.Layout(xaxis=dict(title='Time (ns)'),yaxis=dict(title='Counts'),title='Rabi')
        if plotcurrent == 1 and i == filesize-1:
            plotfun.add_scatter(x = x2, y = y2, name = 'Recent Data', mode='lines+markers')
        else:
            plotfun.add_scatter(x = x2, y = y2, name = all_runs[files[i]-1]['label'], mode='lines+markers')
    return plotfun


def lorentzian(x, amp, a0, x0, g):
    denom = (x - x0)**2 + (0.5*g)**2
    num = 0.5*g
    frac = a0 - (num/denom) * (amp)/np.pi
    return frac


def sinedamp(x,a,c,f,t):
    fun = a*np.cos(2*np.pi*f*x)*np.exp(-1*(x/t)) + c
    return fun


def stretchedexp(x,a,c,k,t):
    fun = a*np.exp(-1*(x/t)**k) + c
    return fun


def fitlorentzian(file, expt, plotfun, lb, ub, num, colour):
    "Fitting parameters are Amplitude, Baseline Offset, ODMR frequency in GHz, and Width in GHz. \
    Attribute - expt - can either be 'odmr' or 'pulsedodmr'"
    if np.size(file) != 0:
        Data = exc.load_by_id(file)
    else:
        lastExperiment = exc.load_last_experiment()
        Data = lastExperiment.last_data_set()
    xaxis = 'Frequency'
    x1 = np.squeeze(Data.get_data(xaxis))

    if expt == 'odmr':
        yaxis = 'Counts'
        y1 = normalizeCounts(Data.get_data(yaxis), 50)
    elif expt == 'pulsedodmr':
        y1 = np.squeeze(Data.get_data('Rebased_Counts'))
    '''    
    index = np.argmin(y1)
    freq = x1[index]/1e9

    if (lb and ub) == None:
        lb = [freq-0.01, 0.95, 0.007, 0.0005]
        ub = [freq+0.01, 1.1, 0.02, 0.005]
    '''
    popt, pcov = curve_fit(lorentzian, x1 / 1e9, y1, bounds=(lb, ub))
    fitname = 'Run ' + str(file) + ': Fit ' + str(num) + ', Resonance @ ' + str(round((popt[2]), 3)) + ' GHz'
    plotfun.add_scatter(x=x1, y=lorentzian(x1 / 1e9, *popt), name=fitname, marker=dict(color=alternate_colours[colour]))
    return plotfun, popt, pcov


def fitsinedamp(file, expt, plotfun, lb, ub, num, colour):
    "Fitting parameters are Amplitude, Baseline Offset, Oscillation Frequency in GHz, and Decay Time in ns.\
    Attribute - expt - can either be 'rabi' or 'spinecho' or 'doubleecho'"
    if np.size(file) != 0:
        Data = exc.load_by_id(file)
    else:
        lastExperiment = exc.load_last_experiment()
        Data = lastExperiment.last_data_set()
    xaxis = 'Time'

    if expt == 'rabi':
        x1 = np.squeeze(Data.get_data(xaxis))
        y1 = np.squeeze(Data.get_data('Rebased_Counts'))
    elif expt == 'spinecho':
        x1 = 2 * np.squeeze(Data.get_data(xaxis))
        y1 = np.squeeze(Data.get_data('Rebased_Counts'))
    elif expt == 'doubleecho':
        x1 = 2 * np.squeeze(Data.get_data(xaxis))
        y1ms0 = np.squeeze(Data.get_data('Act_Counts'))
        y1ms1 = np.squeeze(Data.get_data('Ref_Counts'))
        y1 = y1ms0 - y1ms1
    '''    
    ampMin = 0.8*(y1.max()-y1.min())/2
    ampMax = 1.2*(y1.max()-y1.min())/2
    boMin = 0.4*(y1.max()+y1.min())
    boMax = 0.6*(y1.max()+y1.min())
    freq = 1/(x1[y1.argmin()]*2)
    freqMin = freq*0.7
    freqMax = freq*1.3
    if (lb and ub) == None:
        lb = [ampMin, boMin, freqMin, 0]
        ub = [ampMax, boMax, freqMax, 1e5]
    '''
    popt, pcov = curve_fit(sinedamp, x1, y1, bounds=(lb, ub))
    yval = sinedamp(x1, *popt)
    fitname = 'Run ' + str(file) + ': Fit ' + str(num) + ', \u0394 = ' + str(
        round((popt[0] * 2 * 100), 1)) + ' %' + ', \u03C0 = ' + str(round((0.5 / popt[2]), 1)) + ' ns'
    if plotfun is not None:
        plotfun.add_scatter(x=x1, y=yval, name=fitname, line=dict(shape='spline'),
                            marker=dict(color=alternate_colours[colour]))
    return plotfun, popt, pcov


def find_T2(file, expt, plotfun, threshold, lb, ub, revivals, num, colour):
    "Fitting parameters are Amplitude, Baseline Offset, Power of Stretched Exponential, and Decay Time in ns.\
    Attribute - expt - can either be 'spinecho' or 'doubleecho'.\
    threshold is required for peak detection"

    if np.size(file) != 0:
        Data = exc.load_by_id(file)
    else:
        lastExperiment = exc.load_last_experiment()
        Data = lastExperiment.last_data_set()

    x1 = 2 * np.squeeze(Data.get_data('Time'))
    if expt == 'spinecho':
        y1 = np.squeeze(Data.get_data('Rebased_Counts'))
    elif expt == 'doubleecho':
        y1ms0 = np.squeeze(Data.get_data('Act_Counts'))
        y1ms1 = np.squeewze(Data.get_data('Ref_Counts'))
        y1 = y1ms0 - y1ms1
        y1 = (y1 + max(y1)) / (2 * max(y1))
    if revivals == 1:
        peaks, _ = find_peaks(y1, height=threshold, distance=6, width=3)
        x0 = np.array([0])
        y0 = np.array([y1[0]])
        xpeaks = np.concatenate([x0, x1[peaks]], axis=0)
        print(xpeaks)
        ypeaks = np.concatenate([y0, y1[peaks]], axis=0)
        print(ypeaks)
    else:
        xpeaks = x1
        ypeaks = y1

    popt, pcov = curve_fit(stretchedexp, xpeaks, ypeaks, bounds=(lb, ub))
    yval = stretchedexp(xpeaks, *popt)

    fitname = 'Run ' + str(file) + ': Fit ' + str(num) + ', T2 = ' + str(round((popt[3] / 1e3), 1)) + ' \u03BCs'

    # plotfun.add_scatter(x = xpeaks, y = ypeaks, mode='markers', name = 'Detected Peaks',marker=dict(color='red', size=10, opacity=0.5))
    plotfun.add_scatter(x=xpeaks, y=yval, name=fitname, line=dict(shape='spline'), mode='lines',
                        marker=dict(color=alternate_colours[colour]))
    return plotfun, popt, pcov
