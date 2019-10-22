#!/usr/bin/env python
# coding: utf-8

# In[1]:


import qcodes as qc
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import qcodes.dataset.experiment_container as exc
import dash
import dash_core_components as dcc
import dash_html_components as html
from qcodes.dataset.data_set import load_by_id
from qcodes.dataset.data_export import get_data_by_id
from qcodes import load_experiment
from dash.dependencies import Input, Output, State
from plotly import tools
import plotly.io as pio
from plotly import graph_objs as go
from qcodes.instrument_drivers.rohde_schwarz import SGS100A
from qcodes.dataset.measurements import Measurement
from scipy.optimize import curve_fit
import scipy.fftpack
from scipy import signal
from scipy.signal import find_peaks
from dash.exceptions import PreventUpdate
import dash_table as dt
import pandas as pd
import os


# In[2]:


configuration = qc.config
print(f'Using config file from {configuration.current_config_path}')


# In[3]:


#exc.experiments()


# In[4]:


def normalizeCounts(countArray, num):
    reb_counts = np.squeeze(countArray)/np.mean(np.sort(np.squeeze(countArray))[-num:])
    return reb_counts


# In[5]:


def plotdata(expt, files, normalize, name, plotcurrent=0, nPi = []): 
    "Attribute - expt - can either be 'counting' or 'odmr' or 'pulsedodmr' or 'rabi' or 'ramsey' or 'spinecho' or 'doubleecho' or 'nmr'.    If value of plotcurrent is 1 then it will plot the current data. If value of normalize is 1 then it will normalize 'odmr' and 'doubleecho' signal."
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
            plotfun.add_scatter(x = x2, y = y2, name = name[files[i]-1]['label'], mode='lines+markers')
    return plotfun


# In[6]:


def lorentzian(x, amp, a0, x0, g):
    denom = (x - x0)**2 + (0.5*g)**2
    num = 0.5*g
    frac = a0 - (num/denom) * (amp)/np.pi
    return frac


# In[7]:


def sinedamp(x,a,c,f,t):
    fun = a*np.cos(2*np.pi*f*x)*np.exp(-1*(x/t)) + c
    return fun


# In[8]:


def stretchedexp(x,a,c,k,t):
    fun = a*np.exp(-1*(x/t)**k) + c
    return fun


# In[9]:


def fitlorentzian(file, expt, plotfun,lb,ub, num,colour):
    "Fitting parameters are Amplitude, Baseline Offset, ODMR frequency in GHz, and Width in GHz.     Attribute - expt - can either be 'odmr' or 'pulsedodmr'"
    if np.size(file) != 0:
        Data = exc.load_by_id(file)
    xaxis = 'Frequency'
    x1 = np.squeeze(Data.get_data(xaxis))
    
    if expt == 'odmr':
        yaxis = 'Counts'
        y1 = normalizeCounts(Data.get_data(yaxis),50)
    elif expt == 'pulsedodmr':
        y1 = np.squeeze(Data.get_data('Rebased_Counts'))  
 
    popt, pcov = curve_fit(lorentzian, x1/1e9, y1, bounds=(lb,ub))
    fitname = 'Run '+ str(file) + ': Fit ' + str(num) + ', Resonance @ ' + str(round((popt[2]),3)) + ' GHz'
    plotfun.add_scatter(x = x1 , y = lorentzian(x1/1e9,*popt), name = fitname, marker=dict(color=alternate_colours[colour]))
    return plotfun,popt,pcov


# In[10]:


def fitsinedamp(file,expt,plotfun,lb,ub,num,colour):   
    "Fitting parameters are Amplitude, Baseline Offset, Oscillation Frequency in GHz, and Decay Time in ns.    Attribute - expt - can either be 'rabi' or 'spinecho' or 'doubleecho'"
    if np.size(file) != 0:
        Data = exc.load_by_id(file)
    xaxis = 'Time'
    if expt == 'rabi':
        x1 = np.squeeze(Data.get_data(xaxis))
        y1 = np.squeeze(Data.get_data('Rebased_Counts'))
    elif expt == 'spinecho':
        x1 = 2*np.squeeze(Data.get_data(xaxis))
        y1 = np.squeeze(Data.get_data('Rebased_Counts'))
    elif expt == 'doubleecho':
        x1 = 2*np.squeeze(Data.get_data(xaxis))
        y1ms0 = np.squeeze(Data.get_data('Act_Counts'))
        y1ms1 = np.squeeze(Data.get_data('Ref_Counts'))
        y1 = y1ms0 - y1ms1

    popt, pcov = curve_fit(sinedamp, x1, y1, bounds=(lb,ub))
    yval = sinedamp(x1,*popt)
    fitname = 'Run '+ str(file)+ ': Fit '+ str(num) + ', \u0394 = ' + str(round((popt[0]*2*100),1)) + ' %' + ', \u03C0 = ' +  str(round((0.5/popt[2]),1)) + ' ns'
    if plotfun is not None:
        plotfun.add_scatter(x = x1 , y = yval, name = fitname,line=dict(shape='spline'),marker=dict(color=alternate_colours[colour])) 
    return plotfun, popt, pcov


# In[11]:


def find_T2(file,expt,plotfun, threshold, lb, ub, revivals, num,colour):   
    "Fitting parameters are Amplitude, Baseline Offset, Power of Stretched Exponential, and Decay Time in ns.    Attribute - expt - can either be 'spinecho' or 'doubleecho'.    threshold is required for peak detection"
    
    if np.size(file) != 0:
        Data = exc.load_by_id(file)
    else:
        lastExperiment = exc.load_last_experiment()
        Data = lastExperiment.last_data_set()
        
    x1 = 2*np.squeeze(Data.get_data('Time'))
    if expt == 'spinecho':
        y1 = np.squeeze(Data.get_data('Rebased_Counts'))  
    elif expt == 'doubleecho':
        y1ms0 = np.squeeze(Data.get_data('Act_Counts'))
        y1ms1 = np.squeewze(Data.get_data('Ref_Counts'))
        y1 = y1ms0 - y1ms1 
        y1 = (y1 + max(y1))/(2*max(y1))
    if revivals == 1:
        peaks, _= find_peaks(y1,height=threshold,distance=6, width=3)
        x0 = np.array([0])
        y0 = np.array([y1[0]])
        xpeaks = np.concatenate([x0, x1[peaks]], axis=0)
        print(xpeaks)
        ypeaks = np.concatenate([y0, y1[peaks]], axis=0)
        print(ypeaks)
    else:
        xpeaks = x1
        ypeaks = y1
    
    popt, pcov = curve_fit(stretchedexp, xpeaks, ypeaks, bounds=(lb,ub))
    yval = stretchedexp(xpeaks,*popt)
    
    fitname = 'Run ' + str(file)+ ': Fit '+ str(num) + ', T2 = ' + str(round((popt[3]/1e3),1)) + ' \u03BCs' 
    
    #plotfun.add_scatter(x = xpeaks, y = ypeaks, mode='markers', name = 'Detected Peaks',marker=dict(color='red', size=10, opacity=0.5)) 
    plotfun.add_scatter(x = xpeaks , y = yval, name = fitname, line=dict(shape='spline'), mode='lines',marker=dict(color=alternate_colours[colour]))
    return plotfun, popt, pcov


# In[12]:


def fouriertransform(file,expt,plotfun):
    "Attribute - expt - can either be 'rabi' or 'ramsey' or 'spinecho' or 'doubleecho'"
    layout = go.Layout(xaxis=dict(title='Frequency (GHz)'),yaxis=dict(title='Amplitude'),title = 'Fourier Transform of a ' + expt.capitalize() + ' signal')
    freqPlot = go.Figure(layout=layout)
    if np.size(file) != 0:
        Data = exc.load_by_id(file)
    else:
        lastExperiment = exc.load_last_experiment()
        Data = lastExperiment.last_data_set()
        
    xaxis = 'Time'
    if expt == 'rabi' or expt == 'ramsey':
        x1 = np.squeeze(Data.get_data(xaxis))
        y1 = np.squeeze(Data.get_data('Rebased_Counts'))
    elif expt == 'spinecho':
        x1 = 2*np.squeeze(Data.get_data(xaxis))
        y1 = np.squeeze(Data.get_data('Rebased_Counts'))
    elif expt == 'doubleecho':
        x1 = 2*np.squeeze(Data.get_data(xaxis))
        y1ms0 = np.squeeze(Data.get_data('Act_Counts'))
        y1ms1 = np.squeeze(Data.get_data('Ref_Counts'))
        y1 = y1ms0 - y1ms1
    elif expt == 'nmr':
        x1 = np.squeeze(Data.get_data(xaxis))
        y1ms0 = np.squeeze(Data.get_data('Act_Counts'))
        y1ms1 = np.squeeze(Data.get_data('Ref_Counts'))
        y1 = y1ms1 - y1ms0
        
    step = x1[1] - x1[0]
        
    fourTrans = np.fft.fft(y1)
    freqs = np.fft.fftfreq(y1.shape[-1], step)
    fourTransReal = fourTrans.real
    if plotfun == None:
        freqPlot.add_scatter(x = freqs, y = fourTransReal,line=dict(shape='spline'), mode='lines', name = 'Run ' + str(file))
        return freqPlot
    else:
        plotfun.add_scatter(x = freqs, y = fourTransReal,line=dict(shape='spline'), mode='lines', name = 'Run ' + str(file))
        return plotfun


# In[13]:


types = [{'label':'Counting', 'value': 'counting'},
         {'label':'ODMR', 'value': 'odmr'},
         {'label':'Pulsed ODMR', 'value': 'pulsedodmr'},
         {'label':'Rabi', 'value': 'rabi'},
         {'label':'Ramsey', 'value': 'ramsey'},
         {'label':'Spin Echo', 'value': 'spinecho'},
         {'label':'Double Echo', 'value': 'doubleecho'},
         {'label':'NMR', 'value': 'nmr'}]
dataNeeded = {'odmr': ['Frequency','Counts'], 
              'pulsedodmr': ['Frequency','Rebased_Counts'], 
              'spinecho': ['Time','Rebased_Counts'],
              'doubleecho': ['Time','Act_Counts','Ref_Counts'],
              'nmr': ['Time','Act_Counts','Ref_Counts'],
              'counting': ['Time','Rebased_Counts'],
              'rabi': ['Time','Rebased_Counts'],
              'ramsey':['Time','Rebased_Counts']}
analysisTypes = {'counting': [{'label': 'None Available', 'value': 'none'}],
                 'odmr': [{'label': 'Lorentzian', 'value': 'lorentz'}], 
                 'pulsedodmr': [{'label': 'Lorentzian', 'value': 'lorentz'}],
                 'spinecho': [{'label': 'Stretched Exponential', 'value': 'exponent'},
                              {'label': 'Fourier Transform','value': 'fourier'}],
                 'doubleecho': [{'label': 'Stretched Exponential', 'value': 'exponent'},
                              {'label': 'Fourier Transform','value': 'fourier'}],
                 'nmr': [{'label': 'Fourier Transform','value': 'fourier'}],
                 'rabi': [{'label':'Damped Sine','value':'sine'}],
                 'ramsey': [{'label': 'Fourier Transform','value': 'fourier'}]}

bounds = {'lorentz': ['Amplitude','Baseline Offset', 'ODMR Frequency (GHz)', 'Width (GHz)'],
         'sine': ['Amplitude','Baseline Offset', 'Oscillation Frequency (GHz)', 'Decay Time (ns)'],
         'exponent': ['Amplitude','Baseline Offset','Power of Stretched Exponential','Decay Time (ns)']}
default_values = {'lorentz': {'ub': [0.005,1.1,2.9,0.02], 'lb': [0.0005,1,2.6,0.007]}, 
                  'sine': {'ub': [0.1,1.2,0.008,100000], 'lb': [0.01,0.7,0.001,0]}, 
                  'exponent': {'ub': [2,3,3,5e6], 'lb': [0,0,0,0],'threshold': 0}}
table_columns = {'fourier': [{'name': 'Analysis', 'id': 'analysis'}], 
                 'lorentz': [{'name': ['','Analysis'], 'id': 'analysis'}, 
                             {'name': ['Amplitude', 'Fit'], 'id': 'amplitude-opt'},
                             {'name': ['Amplitude', 'Error'], 'id': 'amplitude-cov'},
                             {'name': ['Baseline Offset', 'Fit'], 'id': 'baseline-opt'},
                             {'name': ['Baseline Offset', 'Error'], 'id': 'baseline-cov'},
                             {'name': ['ODMR Frequency (GHz)', 'Fit'], 'id': 'odmr-opt'},
                             {'name': ['ODMR Frequency (GHz)', 'Error'], 'id': 'odmr-cov'},
                             {'name': ['Width (GHz)', 'Fit'], 'id': 'width-opt'},
                             {'name': ['Width (GHz)', 'Error'], 'id': 'width-cov'}], 
                 'sine': [{'name': ['','Analysis'], 'id': 'analysis'}, 
                          {'name': ['Amplitude', 'Fit'], 'id': 'amplitude-opt'},
                          {'name': ['Amplitude', 'Error'], 'id': 'amplitude-cov'},
                          {'name': ['Baseline Offset', 'Fit'], 'id': 'baseline-opt'},
                          {'name': ['Baseline Offset', 'Error'], 'id': 'baseline-cov'},
                          {'name': ['Oscillation Frequency (GHz)', ' Fit'], 'id': 'freq-opt'},
                          {'name': ['Oscillation Frequency (GHz)', 'Error'], 'id': 'freq-cov'},
                          {'name': ['Decay Time (ns)', 'Fit'], 'id': 'decay-opt'},
                          {'name': ['Decay Time (ns)', 'Error'], 'id': 'decay-cov'}], 
                 'exponent': [{'name': ['','Analysis'], 'id': 'analysis'}, 
                             {'name': ['Amplitude', 'Fit'], 'id': 'amplitude-opt'},
                             {'name': ['Amplitude', 'Error'], 'id': 'amplitude-cov'},
                             {'name': ['Baseline Offset', 'Fit'], 'id': 'baseline-opt'},
                             {'name': ['Baseline Offset', 'Error'], 'id': 'baseline-cov'},
                             {'name': ['Power of Stretched Exponential', 'Fit'], 'id': 'power-opt'},
                             {'name': ['Power of Stretched Exponential', 'Error'], 'id': 'power-cov'},
                             {'name': ['Decay Time (ns)', 'Fit'], 'id': 'decay-opt'},
                             {'name': ['Decay Time (ns)', 'Error'], 'id': 'decay-cov'}],
                'none': []}

colours = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3' ,'#FF6692' ,'#B6E880', '#FF97FF', '#FECB52']
alternate_colours = ['#9ca2fc','#f7aea1','#99ffe4','#ca9cfc','#ffc599','#9eecfa','#ff99b6','#cdefa9','#ffe6ff','#fee09a']


# In[14]:


app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    dcc.Store(id='database-data', storage_type ='session',data = None),
    dcc.Store(id='experiment-data', storage_type='session', data = None),
    dcc.Store(id='graph-data', storage_type='session', data = None),
    dcc.Store(id='data-type-data', storage_type='session', data = None),
    dcc.Store(id='analysis-graph-data', storage_type='session', data = None),
    dcc.Store(id='live-data', storage_type='session', data = None),
    dcc.Store(id='live-plot-data', storage_type='session', data = None),
    dcc.Tabs(id="tabs", value='tab-0', children=[dcc.Tab(label='Database',value='tab-0'),
                                                 dcc.Tab(label='Graph', value='tab-1'),
                                                 dcc.Tab(label='Live', value= 'tab-2'),
                                                 dcc.Tab(label='Analyse', value='tab-3')]),
    html.Div(id='tabs-content'),
    dcc.Interval(id='interval-component',interval=2*1000, n_intervals=0)],
    style={'columnCount': 1})

@app.callback(Output('tabs-content', 'children'),
             [Input('database-data','data'), Input('tabs', 'value'), 
              Input('graph-data', 'data'), Input('analysis-graph-data','data')],
             [State('live-data','data'),State('experiment-data','data')])
def render_content(d_data, tab, data, a_data, l_data, e_data):
    if d_data == None:
        if tab == 'tab-0':
            return html.Div([
                             html.H2(children='Choose Database'),
                             html.P('Enter the file name of the database you would like to use:'),
                             dcc.Input(id= 'database-input', style = {'width': '300px'}),
                             html.Br(),
                             html.P(id='database-error',style ={'color': 'red'}),
                             html.Button(id= 'database-button', children = 'Use Database')
                            ])
        else:
            return html.Div([
                            html.P(children = 'No database chosen!', style={'color': 'red'})
                            ])
    else:
        if tab == 'tab-0':
            return html.Div([
                             html.H2(children='Choose Database'),
                             html.P('The database currently being used is: ' + d_data['file']),
                             html.P('Enter the file name of the database you would like to use:'),
                             dcc.Input(id= 'database-input', style = {'width': '300px'}),
                             html.Br(),
                             html.P(id='database-error',style ={'color': 'red'}),
                             html.Button(id= 'database-button', children = 'Use Database / Update Runs!')
                            ])
        if data == None:
            if tab == 'tab-1':
                return html.Div([
                    html.H2(children='Graph Data'),
                    html.Div([
                        html.Div([
                            html.Label('1) Experiment'), 
                            dcc.Dropdown(id='experiment', 
                                 options=[{'label': d_data['experiments'][o], 'value': o+1} 
                                          for o in range((len(d_data['experiments'])))]
                                        ),
                            html.Label('2) Runs'), 
                            dcc.Dropdown(id='runs', multi=True),
                            html.Label('3) Data Type'),
                            dcc.Dropdown(id='data-type', options = types),
                            html.P(id='type-err', style={'color': 'red'})],
                            style={'width': '48%', 'display': 'inline-block'}), 
                        html.Div([
                            html.Label('4) Extra Runs (Optional)'), 
                            dcc.Dropdown(id='extra-runs',options = d_data['runs'], multi=True),
                            html.P(id='run-err', style={'color': 'red'}),
                            html.Div([
                                html.Label('5) Normalise? (Needed For Analysis)'), 
                                dcc.RadioItems(id='normalise', options = [{'label': 'Yes', 'value': 1}, 
                                                                      {'label': 'No', 'value': 0}], value = 1),
                                html.Button(id='submit-button', children='Plot!')],
                                style={'width': '48%', 'display': 'inline-block'})],
                        style={'width': '48%', 'float': 'right', 'display': 'inline-block'})],
                    style ={'width': '100%','display': 'inline-block'}),
                html.Hr(),
                dcc.Loading(id="loading-1", children=
                            [dcc.Graph(style={'height': '700px'},id='graph', config=dict(showSendToCloud=True))], 
                            type="default"),
                ])
            elif tab == 'tab-2':
                l_type = None
                l_norm = 1
                if l_data != None:
                    l_type = l_data['type']
                    l_norm = l_data['normalise']
                return html.Div([
                    html.H2(children='Live Data'),
                    html.P(id='live-error',style={'color': 'red'}),
                    html.Div([
                        html.Div([
                            html.Label('1) Data Type'),
                            dcc.Dropdown(id='live-type', options = types, value = l_type)],
                        style={'width': '48%', 'display': 'inline-block'}),
                        html.Div([
                            html.Label('2) Normalise?'),
                            dcc.RadioItems(id='live-normalise', options = [{'label': 'Yes', 'value': 1}, 
                                                                           {'label': 'No', 'value': 0}], value = l_norm)],
                        style={'width': '48%', 'float':'right','display': 'inline-block'})
                    ],style = {'width': '100%', 'display': 'inline-block'}),
                    html.Hr(),
                    dcc.Graph(id='live-update-graph', style={'height': '700px'}, config=dict(showSendToCloud=True)),
                    dcc.RadioItems(id='live-add', options = [{'label': 'Yes', 'value': 1}, 
                                                             {'label': 'No', 'value': 0}], value = 0, style = {'display':'none'})
                ])
            elif tab == 'tab-3':
                return html.Div([
                    html.H2(children='Analyse Data'),
                    html.P(children = 'No data chosen in Graph tab!', style={'color': 'red'})
                ])

        else:
            if tab == 'tab-1':
                exper = None
                if e_data['file'] == d_data['file']:
                    exper = e_data['experiment']
                return html.Div([
                    html.H2(children='Graph Data'),
                    html.Div([
                        html.Div([
                            html.Label('1) Experiment'), 
                            dcc.Dropdown(id='experiment', 
                                 options=[{'label': d_data['experiments'][o], 'value': o+1} 
                                          for o in range((len(d_data['experiments'])))]
                                        , value = exper),
                            html.Label('2) Runs'), 
                            dcc.Dropdown(id='runs', multi=True, value = data['run']),
                            html.Label('3) Data Type'),
                            dcc.Dropdown(id='data-type', options = types, value = data['type']),
                            html.P(id='type-err', style={'color': 'red'})],
                            style={'width': '48%', 'display': 'inline-block'}), 
                        html.Div([
                            html.Label('4) Extra Runs (Optional)'), 
                            dcc.Dropdown(id='extra-runs',options = d_data['runs'], multi=True, value = data['extra-run']),
                            html.P(id='run-err', style={'color': 'red'}),
                            html.Div([
                                html.Label('5) Normalise? (Needed For Analysis)'), 
                                dcc.RadioItems(id='normalise', options = [{'label': 'Yes', 'value': 1}, 
                                                                      {'label': 'No', 'value': 0}], value = data['normalise']),
                                html.Br(),
                                html.Button(id='submit-button', children='Plot!')],
                                style={'width': '48%', 'display': 'inline-block'})],
                            style={'width': '48%', 'float': 'right', 'display': 'inline-block'})],
                    style = {'width': '100%','display': 'inline-block'}),
                html.Hr(),
                dcc.Loading(id="loading-1", children=
                            [dcc.Graph(style={'height': '700px'},id='graph', config=dict(showSendToCloud=True), 
                                       figure = data['graph'])], 
                            type="default"),

                ])
            elif tab == 'tab-2':
                l_type = None
                l_norm = 1
                l_live = 0
                if l_data != None:
                    l_type = l_data['type']
                    l_norm = l_data['normalise']
                    l_live = l_data['runs']
                return html.Div([
                    html.H2(children='Live Data'),
                    html.P(id='live-error',style={'color': 'red'}),
                    html.Div([
                        html.Div([
                            html.Label('1) Add Graphed Runs?'),
                            dcc.RadioItems(id='live-add', options = [{'label': 'Yes', 'value': 1}, 
                                                                      {'label': 'No', 'value': 0}], value = l_live),
                            html.P(id='adding-error',style={'color': 'red'})],
                        style={'width': '31%', 'float': 'left','display': 'inline-block'}),
                        html.Div([
                            html.Label('2) Data Type'),
                            dcc.Dropdown(id='live-type', options = types, value = l_type)],
                        style={'width': '31%','display': 'inline-block'}),
                        html.Div([
                            html.Label('3) Normalise?'),
                            dcc.RadioItems(id='live-normalise', options = [{'label': 'Yes', 'value': 1}, 
                                                                           {'label': 'No', 'value': 0}], value = l_norm)],
                        style={'width': '31%', 'float':'right','display': 'inline-block'})
                    ],style = {'width': '100%', 'display': 'inline-block'}),
                    html.Hr(),
                    dcc.Graph(id='live-update-graph', style={'height': '700px'},config=dict(showSendToCloud=True)),
                ])

            elif tab == 'tab-3':
                if data['total-run'] == []:
                    return html.Div([
                    html.H2(children='Analyse Data'),
                    html.P(children = 'No data chosen in Graph tab!', style={'color': 'red'})
                ])
                else:   
                    runs_for_analysis = ''
                    runs_for_table = []
                    for a in data['total-run']:
                        runs_for_table.append({'runs': d_data['runs'][a-1]['label']})
                    runs_options = [{'label': d_data['runs'][a-1]['label'], 'value': a} for a in data['total-run']]
                    runs_analysed = []
                    delete_options = []
                    if a_data != None:
                        if a_data['type'] == data['type']:
                            if a_data['a_type'] != 'fourier':
                                for b in a_data['analysis']:
                                    runs_analysed.append({'runs': b['name']})
                                delete_options = [{'label': d['name'], 'value': d['name']} for d in a_data['analysis']]
                            else: 
                                for b in a_data['fourier']['data']:
                                    runs_analysed.append({'runs': b['name']})
                                delete_options = [{'label': d['name'], 'value': d['name']} for d in a_data['fourier']['data']]
                    plot_data = data['graph']
                    fourier_graph = []
                    fourier_show = 'none'
                    if a_data != None:
                        if a_data['type'] == data['type']:
                            for u in a_data['analysis']:
                                plot_data['data'].append(u)
                        if a_data['a_type'] == 'fourier':
                            fourier_graph = a_data['fourier']
                            fourier_show = 'inline-block'
                    return html.Div([
                        html.Div([
                            html.H2(children='Analyse Data'),
                            html.Div([
                                dt.DataTable(id = 'run-table', columns = [{'name': 'Runs Available For Analysis', 'id': 'runs'}], 
                                             data = runs_for_table,style_as_list_view=True),
                                html.Br(),
                                dt.DataTable(id = 'run-table', columns = [{'name': 'Analyses', 'id': 'runs'}], 
                                             data = runs_analysed,style_as_list_view=True),
                                html.Br(),
                                html.Label('1) Run To Analyse'),
                                dcc.Dropdown(id='analysis-runs',options = runs_options, multi = True),
                                html.Label('2) Data Type'),
                                dcc.Dropdown(id='analysis-data-type', options = types, value = data['type']),
                                html.Label('3) Analysis Type'),
                                dcc.Dropdown(id='analysis-type')
                                ],
                                style={'width': '48%', 'display': 'inline-block'}),
                            html.Div([
                                html.Label('4) Upper & Lower Bounds'),
                                html.Div(id='bounds-table-wrap', children = [
                                    dt.DataTable(id = 'bounds-table',columns = [{'name': '', 'id': 'bounds'},
                                                                            {'name': 'Lower', 'id': 'lower'},
                                                                            {'name': 'Upper', 'id': 'upper'}],
                                         editable=True, data =[])],
                                style={'display': 'none'}),
                                html.Div(id='exponent-inputs',children = [
                                    html.Div([
                                        html.Label('Revivals?'),
                                        dcc.RadioItems(id='revivals', options = [{'label': 'Yes', 'value': 1}, 
                                                                          {'label': 'No', 'value': 0}],value=None)
                                    ], style={'width': '48%', 'display': 'inline-block'}),
                                    html.Div([
                                        html.Label('Threshold:'),
                                        dcc.Input(id='threshold', type='text', value =None)
                                    ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
                                ], style ={'display': 'none'}),
                                html.Br(),
                                html.Button(id = 'default_button', children = 'Reset To Default Bounds', style = {'display': 'none'}),
                                html.P(id='fourier',children = None),
                                html.Label('5) Delete Analyses'),
                                dcc.Dropdown(id='delete-runs', options = delete_options, multi = True),
                                html.Br(),
                                html.Button(id='analyse_button', children='Analyse/Delete!'),
                                html.P(id ='a-error',children=None, style={'color': 'red'})],
                                style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
                        ],style = {'width': '100%', 'display': 'inline-block'}),
                        html.Hr(),
                        dcc.Loading(id="loading-1", children=
                            [dcc.Graph(style={'height': '700px'},id='analysis-graph', figure = plot_data,
                                  config=dict(showSendToCloud=True)),
                            html.Div(id = 'data_table', style = {'width': '100%','display': 'inline-block'},children = [
                                html.Hr(),
                                html.H3('Data Table'),
                                dt.DataTable(id = 'output-table', columns = [{'name': 'Runs', 'id': 'runs'}], data =[], 
                                             export_format='csv', export_headers='display',merge_duplicate_headers=True)
                                ]),
                            html.Hr(),
                            dcc.Graph(style={'height': '700px','width': '100%','display': fourier_show},id='fourier-graph', 
                                      figure = fourier_graph,config=dict(showSendToCloud=True)),
                            ], type="default")
                    ])
                
                
@app.callback([Output('database-data','data'),Output('database-error','children')],
             [Input('database-button','n_clicks')],
             [State('database-input','value'), State('database-data', 'data')])
def update_database(clicks,filename,data):
    if clicks is None:
        raise PreventUpdate
    if filename is None:
        if data is None:
            raise PreventUpdate
        else:
            experimentsList = [exp.name for exp in exc.experiments()]
            all_runs = [{'label': str(load_by_id(a)).split('@')[0], 'value': a} for a in range(1, qc.load_last_experiment().last_data_set().run_id+1)]
            data['experiments'] = experimentsList
            data['runs'] = all_runs
            return data, None
    cwd = os.getcwd()
    result = []
    for root, dirs, files in os.walk(cwd):
        if filename in files:
            result.append(os.path.join(root, filename))
    if result == []:
        return dash.no_update, 'File Not Found!'
    data = data or {'file': None, 'experiments': None, 'runs': None}
    nfilename = './'+filename
    # configuration['core']['db_location'] = nfilename
    # qc.dataset.database.initialise_database()
    qc.initialise_or_create_database_at(nfilename)
    experimentsList = [exp.name for exp in exc.experiments()]
    all_runs = [{'label': str(load_by_id(a)).split('@')[0], 'value': a} for a in range(1, qc.load_last_experiment().last_data_set().run_id+1)]
    data['file'] = filename
    data['experiments'] = experimentsList
    data['runs'] = all_runs
    return data, None

@app.callback(
    [Output('runs', 'options'),Output('experiment-data','data')],
    [Input('experiment', 'value')],
    [State('experiment-data','data'),State('database-data','data')])
def update_run(selected_experiment, data,d_data):
    data = data or {'experiment': None, 'file': None}
    if selected_experiment is None:
        return [],data
    exper = load_experiment(selected_experiment)
    data['experiment'] = selected_experiment
    data['file'] = d_data['file']
    return [{'label': str(exper.data_set(a+1)).split('@')[0], 
             'value': exper.data_set(a+1).run_id} for a in range(exper.last_counter)], data

@app.callback(
    [Output('graph', 'figure'), Output('type-err','children'), 
     Output('run-err','children'), Output('graph-data','data')],
    [Input('submit-button','n_clicks')],
    [State('runs','value'), State('data-type','value'), 
     State('extra-runs','value'), State('normalise', 'value'), 
     State('experiment-data','data'),State('experiment','value'),
     State('database-data','data'),State('graph-data','data')])
def update_graph(clicks, selected_runs, selected_type, extra_runs, normalised, e_data, selected_experiment, d_data,data):
    if clicks is None:
        raise PreventUpdate
    if selected_experiment is None:
        if selected_type is None:
            return None, None, None, None
    data = data or {'experiment': None, 'run': None, 'type': None, 'extra-run': None, 
                    'normalise': 1, 'graph': None, 'total-run': None}
    data['experiment'] = e_data['experiment']
    if selected_runs == []:
        return dash.no_update, 'No Runs Selected!', dash.no_update, dash.no_update
    if selected_runs is None:
        return dash.no_update, 'No Runs Selected!', dash.no_update, dash.no_update
    if selected_type is None:
        return dash.no_update, 'No Data Type Selected!', dash.no_update, dash.no_update
    runcheck = [{'label': str(load_by_id(a)).split('@')[0], 'value': a} for a in range(1, qc.load_last_experiment().last_data_set().run_id+1)]
    if d_data['runs'] != runcheck:
        return dash.no_update, 'Please update runs in Database tab!', dash.no_update, dash.no_update 
    for a in selected_runs:
        params = str(load_by_id(a).parameters).split(',')
        for b in dataNeeded[selected_type]:
            if (b in params) is False:
                return dash.no_update, 'Wrong Data Type!', dash.no_update, dash.no_update
    total_runs = selected_runs
    if extra_runs != None:
        for c in extra_runs:
            params_2 = str(load_by_id(c).parameters).split(',')
            for d in dataNeeded[selected_type]:
                if (d in params_2) is False:
                    return dash.no_update, dash.no_update, 'Wrong Run!', dash.no_update
        total_runs = total_runs + extra_runs
    totaldata = plotdata(selected_type, total_runs, normalised,d_data['runs'])
    data['run'] = selected_runs
    data['type'] = selected_type
    data['extra-run'] = extra_runs
    data['normalise'] = normalised
    data['total-run'] = total_runs
    data['graph'] = totaldata
    return totaldata, None, None, data

@app.callback(
    [Output('analysis-type', 'options'), Output('analysis-type','value'), 
     Output('data-type-data','data')],
    [Input('analysis-data-type', 'value')],
    [State('data-type-data','data'), State('analysis-graph-data','data')])
def update_datatype(selected_data_type, data, g_data):
    data = data or {'data-type': None, 'analysis-type': None}
    if selected_data_type is None:
        raise PreventUpdate
    analysis = analysisTypes[selected_data_type]
    if data['analysis-type'] == None:
        analysisValue = analysis[0]['value']
    elif g_data is None:
        analysisValue = analysis[0]['value']
    elif selected_data_type != g_data['type']:
        analysisValue = analysis[0]['value']
    else:
        analysisValue = g_data['a_type']
    data['data-type'] = selected_data_type
    data['analysis-type'] = analysisValue
    return analysis, analysisValue, data

@app.callback(
    [Output('exponent-inputs','style'), Output('fourier','children'),Output('threshold','value'),
     Output('revivals','value'),Output('fourier-graph','style'),Output('default_button','style'),
     Output('bounds-table','data'), Output('bounds-table-wrap','style')],
    [Input('analysis-type','value'), Input('default_button','n_clicks')])
def update_bounds(a_type, clicks):
    if a_type is None:
        raise PreventUpdate
    if a_type == 'fourier':
        return ({'display': 'none'},'No Bounds Necessary.',None,None, 
                {'height': '700px','width': '100%','display': 'inline-block'},{'display': 'none'},[],{'display': 'none'})
    if a_type == 'exponent':
        return ({'display': 'inline'},'',default_values[a_type]['threshold'],0,
                {'height': '700px','display': 'none'},{'display': 'inline'},
                [{'bounds': 'Amplitude','lower': default_values[a_type]['lb'][0],'upper': default_values[a_type]['ub'][0]}, 
                  {'bounds': 'Baseline Offset','lower': default_values[a_type]['lb'][1],'upper': default_values[a_type]['ub'][1]},
                  {'bounds': 'Power Of Stretched Exponential','lower': default_values[a_type]['lb'][2],'upper': default_values[a_type]['ub'][2]},
                  {'bounds': 'Decay Time (ns)','lower': default_values[a_type]['lb'][3],'upper': default_values[a_type]['ub'][3]}],
               {'display': 'inline-block', 'width': '100%'})
    if a_type == 'lorentz':
        return ({'display': 'none'}, '',0,None,{'height': '700px','display': 'none'}, {'display': 'inline'},
                [{'bounds': 'Amplitude','lower': default_values[a_type]['lb'][0],'upper': default_values[a_type]['ub'][0]}, 
                 {'bounds': 'Baseline Offset','lower': default_values[a_type]['lb'][1],'upper': default_values[a_type]['ub'][1]},
                 {'bounds': 'ODMR Frequency (GHz)','lower': default_values[a_type]['lb'][2],'upper': default_values[a_type]['ub'][2]},
                 {'bounds': 'Width (GHz)','lower': default_values[a_type]['lb'][3],'upper': default_values[a_type]['ub'][3]}],
               {'display': 'inline-block', 'width': '100%'})
    if a_type == 'sine':
        return ({'display': 'none'},'',0,None,{'height': '700px','display': 'none'},{'display': 'inline'},
                 [{'bounds': 'Amplitude','lower': default_values[a_type]['lb'][0],'upper': default_values[a_type]['ub'][0]}, 
                  {'bounds': 'Baseline Offset','lower': default_values[a_type]['lb'][1],'upper': default_values[a_type]['ub'][1]},
                  {'bounds': 'Oscillation Frequency','lower': default_values[a_type]['lb'][2],'upper': default_values[a_type]['ub'][2]},
                  {'bounds': 'Decay Time (ns)','lower': default_values[a_type]['lb'][3],'upper': default_values[a_type]['ub'][3]}],
               {'display': 'inline-block', 'width': '100%'}) 
    if a_type == 'none':
        return ({'display': 'none'},'',None,None,{'height': '700px','display': 'none'},{'display': 'none'},[],
               {'display': 'none', 'width': '100%'})
    
@app.callback(
    [Output('analysis-graph', 'figure'), Output('analysis-graph-data','data'),
     Output('fourier-graph','figure'), Output('a-error','children')],
    [Input('analyse_button','n_clicks')],
    [State('analysis-runs','value'), State('analysis-data-type', 'value'),
     State('analysis-type','value'), State('analysis-graph','figure'),State('graph-data','data'),
     State('analysis-graph-data','data'),State('delete-runs','value'),
     State('bounds-table','data'),State('threshold','value'), State('revivals','value'),State('database-data','data')])
def update_analysis_graph(clicks, selected_run, data_type, analysis_type, graph, g_data,data,delete,bounds,threshold,revival,d_data):
    data = data or {'analysis':[], 'fourier':[], 'plot-num':[], 'type': '', 'a_type': '', 'graph': None, 'opt': {}, 'cov': {}}
    if clicks is None:
        raise PreventUpdate
    if selected_run is None and delete is None:
        return dash.no_update, dash.no_update, dash.no_update, 'No Run Chosen!'
    previous_runs = data['plot-num']
    data['plot-num'] = []
    for a in g_data['total-run']:
        data['plot-num'].append({'run': a, 'num': 0})
    for b in data['plot-num']:
        for c in previous_runs:
            if b['run'] == c['run']:
                b['num'] = c['num']
                
    if analysis_type == 'fourier':
        if data['type'] != data_type or '':
            data['fourier'] = []
        if selected_run != None or []:
            if data['fourier'] == []:
                plot_fourier = fouriertransform(selected_run[0],data_type,None)
                data['fourier'] = plot_fourier
                for i in range(1,len(selected_run)):
                    plot_fourier = fouriertransform(selected_run[i],data_type,go.Figure(data['fourier']))
                    data['fourier'] = plot_fourier
            else:
                for i in selected_run:
                    plot_fourier = fouriertransform(i,data_type,go.Figure(data['fourier']))
                    data['fourier'] = plot_fourier
        if delete != None:
            runs_to_delete=[]
            for e in delete:
                for f in data['fourier']['data']:
                    if f['name'] == e:
                        runs_to_delete.append(f)
            for g in runs_to_delete:
                data['fourier']['data'].remove(g)
        data['analysis'] = []
        data['type'] = data_type
        data['a_type'] = analysis_type
        data['graph'] = graph
        data['opt'] = {}
        data['cov'] = {}
        return graph, data, data['fourier'], None
    
    else:
        ub = []
        for a in bounds:
            ub.append(a['upper'])
        lb =[]
        for a in bounds:
            lb.append(a['lower'])
        totaldata = plotdata(g_data['type'], g_data['total-run'], g_data['normalise'],d_data['runs'])
        if data['type'] != data_type or '':
            data['analysis'] = []
            data['opt'] = {}
            data['cov'] = {}
        if selected_run != None or []:
            for a in selected_run:
                color_code = g_data['total-run'].index(a)%10
                for i in data['plot-num']:
                    if i['run'] == a:
                        i['num'] = i['num'] + 1
                        if analysis_type == 'lorentz':
                            totaldata = fitlorentzian(a, data_type, go.Figure(graph),
                                                      lb,ub, i['num'],color_code)
                        elif analysis_type == 'exponent':
                            totaldata = find_T2(a, data_type, go.Figure(graph),threshold,
                                            lb,ub, revival,i['num'],color_code)
                        elif analysis_type == 'sine':
                            totaldata = fitsinedamp(a, data_type, go.Figure(graph),
                                               lb,ub,i['num'],color_code)
                        data['analysis'].append(totaldata[0]['data'][-1])
                        data['graph'] = totaldata[0]
                        data['opt'][totaldata[0]['data'][-1]['name']] = totaldata[1]
                        data['cov'][totaldata[0]['data'][-1]['name']] = np.sqrt(np.diag(totaldata[2]))
        if delete != None:
            runs_to_delete=[]
            for e in delete:
                for f in data['analysis']:
                    if f['name'] == e:
                        runs_to_delete.append(f)
            for g in runs_to_delete:
                data['analysis'].remove(g)
                del data['opt'][g['name']]
                del data['cov'][g['name']]
        data['fourier'] = []
        data['type'] = data_type
        data['a_type'] = analysis_type
        return data['graph'], data, [], None

@app.callback(Output('output-table','columns'),
             [Input('analysis-type','value')],
             [State('output-table', 'data')])
def update_table_columns(a_type,data):
    if a_type is None:
        raise PreventUpdate
    else:
        return table_columns[a_type] 

@app.callback(Output('output-table', 'data'),
             [Input('analysis-graph-data','data'),
              Input('output-table','columns'),
              Input('analysis-data-type','value')])
def update_table_data(data,columns,a_type):
    if data is None:
        raise PreventUpdate
    table_data = []
    if data['type'] == a_type or '':
        for d in data['opt']:
            row = {}
            row['analysis'] = d.split(',')[0]
            row['amplitude-opt'] = round(data['opt'][d][0],5)
            row['amplitude-cov'] = round(data['cov'][d][0],5)
            row['baseline-opt'] = round(data['opt'][d][1],5)
            row['baseline-cov'] = round(data['cov'][d][1],5)
            if data['a_type'] == 'lorentz':
                row['odmr-opt'] = round(data['opt'][d][2],5)
                row['odmr-cov'] = round(data['cov'][d][2],5)
                row['width-opt'] = round(data['opt'][d][3],5)
                row['width-cov'] = round(data['cov'][d][3],5)
            elif data['a_type'] == 'sine':
                row['freq-opt'] = round(data['opt'][d][2],5)
                row['freq-cov'] = round(data['cov'][d][2],5)
                row['decay-opt'] = round(data['opt'][d][3],5)
                row['decay-cov'] = round(data['cov'][d][3],5)
            elif data['a_type'] == 'exponent':
                row['power-opt'] = round(data['opt'][d][2],5)
                row['power-cov'] = round(data['cov'][d][2],5)
                row['decay-opt'] = round(data['opt'][d][3],5)
                row['decay-cov'] = round(data['cov'][d][3],5)
            table_data.append(row)
    return table_data


@app.callback([Output('live-update-graph', 'figure'), Output('live-error','children')],
              [Input('interval-component', 'n_intervals')],
              [State('tabs','value'),State('live-plot-data','data'),
               State('live-type','value'),State('live-normalise','value'),
               State('live-data','data'),State('database-data','data')])
def update_graph_live(n,tab,data,a_type,normalise,l_data,d_data):
    if tab != 'tab-2':
        raise PreventUpdate
    if a_type is None:
        raise PreventUpdate
    if l_data is None:
        raise PreventUpdate
    live_run= exc.load_last_experiment().last_data_set().run_id
    if data != None:
        runs = data['runs']
        runs.append(live_run)
    else:
        runs = [live_run]
    runcheck = [{'label': str(load_by_id(a)).split('@')[0], 'value': a} for a in range(1, qc.load_last_experiment().last_data_set().run_id+1)]
    if d_data['runs'] != runcheck:
        return dash.no_update, 'Please update runs in Database tab!'
    if l_data != None:
        for a in runs:
            params = str(load_by_id(a).parameters).split(',')
            for b in dataNeeded[l_data['type']]:
                if (b in params) is False:
                    return dash.no_update, 'Wrong Data Type!'
        graph = plotdata(l_data['type'],runs, l_data['normalise'],d_data['runs'])
    else:
        for a in runs:
            params = str(load_by_id(a).parameters).split(',')
            for b in dataNeeded[a_type]:
                if (b in params) is False:
                    return dash.no_update, 'Wrong Data Type!'
        graph = plotdata(a_type,runs,normalise,d_data['runs'])
    return graph, None

@app.callback(Output('live-data','data'),
              [Input('live-type','value'), Input('live-normalise','value')],
              [State('live-data','data'), State('live-add','value')])
def update_live_data(l_type, norm, data, add):
    data = data or {'type': None, 'normalise': None, 'runs':  None}
    if l_type is None:
        data = {'type': None, 'normalise': 1, 'runs': 0}
    else:
        data['type'] = l_type
        data['normalise'] = norm
        data['runs'] = add
    return data

@app.callback([Output('live-type','value'), Output('live-normalise','value'),
               Output('live-plot-data','data')],
              [Input('live-add','value')],
              [State('graph-data','data'),State('live-plot-data','data'), State('live-data','data')])
def adhere_live_data(add,data,lp_data, l_data):
    if l_data is None:
        raise PreventUpdate
    l_type = l_data['type']
    l_norm = l_data['normalise']
    if add == 0:
        lp_data = None
    else:
        lp_data = lp_data or {'runs': None}
        lp_data['runs'] = data['total-run']
        l_type = data['type']
        l_norm = data['normalise']
    return l_type, l_norm, lp_data


# In[ ]:


if __name__ == '__main__':
    app.run_server(debug=True, port=8078, threaded=True)


# In[ ]:





# In[ ]:




