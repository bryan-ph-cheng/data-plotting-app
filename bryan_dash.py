external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    dcc.Store(id='experiment-data', storage_type='session', data=None),
    dcc.Store(id='graph-data', storage_type='session', data=None),
    dcc.Store(id='data-type-data', storage_type='session', data=None),
    dcc.Store(id='analysis-graph-data', storage_type='session', data=None),
    dcc.Tabs(id="tabs", value='tab-1', children=[dcc.Tab(label='Graph', value='tab-1'),
                                                 dcc.Tab(label='Analyse', value='tab-2')]),
    html.Div(id='tabs-content'),
    dcc.Interval(id='interval-component', interval=1 * 1000, n_intervals=0)],
    style={'columnCount': 1})


@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value'), Input('graph-data', 'data'),
               Input('analysis-graph-data', 'data')])
def render_content(tab, data, a_data):
    if data == None:
        if tab == 'tab-1':
            return html.Div([
                html.H2(children='Graph Data'),
                html.Div([
                    html.Label('1) Experiment'),
                    dcc.Dropdown(id='experiment',
                                 options=[{'label': experimentsList[o], 'value': o + 1}
                                          for o in range((len(experimentsList)))]
                                 ),
                    html.Label('2) Runs'),
                    dcc.Dropdown(id='runs', multi=True),
                    html.Label('3) Data Type'),
                    dcc.Dropdown(id='data-type', options=types),
                    html.P(id='type-err', style={'color': 'red'})],
                    style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    html.Label('4) Extra Runs (Optional)'),
                    dcc.Dropdown(id='extra-runs', options=all_runs, multi=True),
                    html.P(id='run-err', style={'color': 'red'}),
                    html.Div([
                        html.Label('5) Normalise? (Needed For Analysis)'),
                        dcc.RadioItems(id='normalise', options=[{'label': 'Yes', 'value': 1},
                                                                {'label': 'No', 'value': 0}], value=1),
                        html.Button(id='submit-button', children='Plot!')],
                        style={'width': '48%', 'display': 'inline-block'}),
                    html.Div([
                        html.Label('6) Live Plotting?'),
                        dcc.RadioItems(id='live', options=[{'label': 'Yes', 'value': 1},
                                                           {'label': 'No', 'value': 0}], value=0)
                    ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})],
                    style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
                html.Hr(),
                dcc.Loading(id="loading-1", children=
                [dcc.Graph(style={'height': '700px'}, id='graph', config=dict(showSendToCloud=True))],
                            type="default"),
            ])
        elif tab == 'tab-2':
            return html.Div([
                html.H2(children='Analyse Data'),
                html.P(children='No data chosen in Graph tab!', style={'color': 'red'})
            ])
    else:
        if tab == 'tab-1':
            return html.Div([
                html.H2(children='Graph Data'),
                html.Div([
                    html.Label('1) Experiment'),
                    dcc.Dropdown(id='experiment',
                                 options=[{'label': experimentsList[o], 'value': o + 1}
                                          for o in range((len(experimentsList)))]
                                 , value=data['experiment']),
                    html.Label('2) Runs'),
                    dcc.Dropdown(id='runs', multi=True, value=data['run']),
                    html.Label('3) Data Type'),
                    dcc.Dropdown(id='data-type', options=types, value=data['type']),
                    html.P(id='type-err', style={'color': 'red'})],
                    style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    html.Label('4) Extra Runs (Optional)'),
                    dcc.Dropdown(id='extra-runs', options=all_runs, multi=True, value=data['extra-run']),
                    html.P(id='run-err', style={'color': 'red'}),
                    html.Div([
                        html.Label('5) Normalise? (Needed For Analysis)'),
                        dcc.RadioItems(id='normalise', options=[{'label': 'Yes', 'value': 1},
                                                                {'label': 'No', 'value': 0}], value=data['normalise']),
                        html.Br(),
                        html.Button(id='submit-button', children='Plot!')],
                        style={'width': '48%', 'display': 'inline-block'}),
                    html.Div([
                        html.Label('6) Live Plotting?'),
                        dcc.RadioItems(id='live', options=[{'label': 'Yes', 'value': 1},
                                                           {'label': 'No', 'value': 0}], value=data['live'])
                    ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})],
                    style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
                html.Hr(),
                dcc.Loading(id="loading-1", children=
                [dcc.Graph(style={'height': '700px'}, id='graph', config=dict(showSendToCloud=True),
                           figure=data['graph'])],
                            type="default"),

            ])
        elif tab == 'tab-2':
            if data['total-run'] == []:
                return html.Div([
                    html.H2(children='Analyse Data'),
                    html.P(children='No data chosen in Graph tab!', style={'color': 'red'})
                ])
            else:
                runs_for_analysis = ''
                runs_for_table = []
                for a in data['total-run']:
                    runs_for_table.append({'runs': all_runs[a - 1]['label']})
                runs_options = [{'label': all_runs[a - 1]['label'], 'value': a} for a in data['total-run']]
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
                            delete_options = [{'label': d['name'], 'value': d['name']} for d in
                                              a_data['fourier']['data']]
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
                            dt.DataTable(id='run-table',
                                         columns=[{'name': 'Runs Available For Analysis', 'id': 'runs'}],
                                         data=runs_for_table, style_as_list_view=True),
                            html.Br(),
                            dt.DataTable(id='run-table', columns=[{'name': 'Analyses', 'id': 'runs'}],
                                         data=runs_analysed, style_as_list_view=True),
                            html.Br(),
                            html.Label('1) Run To Analyse'),
                            dcc.Dropdown(id='analysis-runs', options=runs_options),
                            html.Label('2) Data Type'),
                            dcc.Dropdown(id='analysis-data-type', options=types, value=data['type']),
                            html.Label('3) Analysis Type'),
                            dcc.Dropdown(id='analysis-type')
                        ],
                            style={'width': '48%', 'display': 'inline-block'}),
                        html.Div([
                            html.Label('4) Upper & Lower Bounds'),
                            html.Div(id='bounds-table-wrap', children=[
                                dt.DataTable(id='bounds-table', columns=[{'name': '', 'id': 'bounds'},
                                                                         {'name': 'Lower', 'id': 'lower'},
                                                                         {'name': 'Upper', 'id': 'upper'}],
                                             editable=True, data=[])],
                                     style={'display': 'none'}),
                            html.Div(id='exponent-inputs', children=[
                                html.Div([
                                    html.Label('Revivals?'),
                                    dcc.RadioItems(id='revivals', options=[{'label': 'Yes', 'value': 1},
                                                                           {'label': 'No', 'value': 0}], value=None)
                                ], style={'width': '48%', 'display': 'inline-block'}),
                                html.Div([
                                    html.Label('Threshold:'),
                                    dcc.Input(id='threshold', type='text', value=None)
                                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
                            ], style={'display': 'none'}),
                            html.Div([
                                html.Br(),
                                html.Button(id='default_button', children='Reset To Default Bounds',
                                            style={'display': 'none'})],
                                style={'display': 'inline-block'}),
                            html.P(id='fourier', children=None),
                            html.Label('5) Delete Analyses'),
                            dcc.Dropdown(id='delete-runs', options=delete_options, multi=True),
                            html.Br(),
                            html.Button(id='analyse_button', children='Analyse/Delete!'),
                            html.P(id='a-error', children=None, style={'color': 'red'})],
                            style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
                    ], style={'width': '100%', 'display': 'inline-block'}),
                    html.Hr(),
                    dcc.Loading(id="loading-1", children=
                    [dcc.Graph(style={'height': '700px'}, id='analysis-graph', figure=plot_data,
                               config=dict(showSendToCloud=True)),
                     html.Div(id='data_table', style={'width': '1270px', 'display': 'inline-block'}, children=[
                         html.Hr(),
                         html.H3('Data Table'),
                         dt.DataTable(id='output-table', columns=[{'name': 'Runs', 'id': 'runs'}], data=[],
                                      export_format='csv', export_headers='display', merge_duplicate_headers=True)
                     ]),
                     html.Hr(),
                     dcc.Graph(style={'height': '700px', 'width': '1270px', 'display': fourier_show},
                               id='fourier-graph',
                               figure=fourier_graph, config=dict(showSendToCloud=True)),
                     ], type="default")
                ])


@app.callback(
    [Output('runs', 'options'), Output('experiment-data', 'data')],
    [Input('experiment', 'value')],
    [State('experiment-data', 'data')])
def update_run(selected_experiment, data):
    if selected_experiment is None:
        raise PreventUpdate
    exper = load_experiment(selected_experiment)
    data = data or {'experiment': None, 'run': None, 'type': None, 'extra-run': None,
                    'normalise': 1, 'graph': None, 'total-run': None, 'live': 0}
    data['experiment'] = selected_experiment
    return [{'label': str(exper.data_set(a + 1)).split('@')[0],
             'value': exper.data_set(a + 1).run_id} for a in range(exper.last_counter)], data


@app.callback(
    [Output('graph', 'figure'), Output('type-err', 'children'),
     Output('run-err', 'children'), Output('graph-data', 'data')],
    [Input('submit-button', 'n_clicks')],
    [State('runs', 'value'), State('data-type', 'value'),
     State('extra-runs', 'value'), State('normalise', 'value'),
     State('experiment-data', 'data'), State('live', 'value')])
def update_graph(clicks, selected_runs, selected_type, extra_runs, normalised, data, live):
    if clicks is None:
        raise PreventUpdate
    if selected_runs is None:
        raise PreventUpdate
    if selected_type is None:
        return dash.no_update, 'No Data Type Selected!', dash.no_update, dash.no_update
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
    totaldata = plotdata(selected_type, total_runs, normalised)
    data['run'] = selected_runs
    data['type'] = selected_type
    data['extra-run'] = extra_runs
    data['normalise'] = normalised
    data['total-run'] = total_runs
    data['graph'] = totaldata
    data['live'] = live
    if live == 1:
        updatedata(selected_type, totaldata, normalised)
    return totaldata, None, None, data


@app.callback(
    [Output('analysis-type', 'options'), Output('analysis-type', 'value'),
     Output('data-type-data', 'data')],
    [Input('analysis-data-type', 'value')],
    [State('data-type-data', 'data'), State('analysis-graph-data', 'data')])
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
    [Output('exponent-inputs', 'style'), Output('fourier', 'children'), Output('threshold', 'value'),
     Output('revivals', 'value'), Output('fourier-graph', 'style'), Output('default_button', 'style'),
     Output('bounds-table', 'data'), Output('bounds-table-wrap', 'style')],
    [Input('analysis-type', 'value'), Input('default_button', 'n_clicks')])
def update_bounds(a_type, clicks):
    if a_type is None:
        raise PreventUpdate
    if a_type == 'fourier':
        return ({'display': 'none'}, 'No Bounds Necessary.', None, None,
                {'height': '700px', 'display': 'inline-block'}, {'display': 'none'}, [], {'display': 'none'})
    if a_type == 'exponent':
        return ({'display': 'inline'}, '', default_values[a_type]['threshold'], 0,
                {'height': '700px', 'display': 'none'}, {'display': 'inline', 'float': 'right'},
                [{'bounds': 'Amplitude', 'lower': default_values[a_type]['lb'][0],
                  'upper': default_values[a_type]['ub'][0]},
                 {'bounds': 'Baseline Offset', 'lower': default_values[a_type]['lb'][1],
                  'upper': default_values[a_type]['ub'][1]},
                 {'bounds': 'Power Of Stretched Exponential', 'lower': default_values[a_type]['lb'][2],
                  'upper': default_values[a_type]['ub'][2]},
                 {'bounds': 'Decay Time (ns)', 'lower': default_values[a_type]['lb'][3],
                  'upper': default_values[a_type]['ub'][3]}],
                {'display': 'inline-block', 'width': '600px'})
    if a_type == 'lorentz':
        return ({'display': 'none'}, '', 0, None, {'height': '700px', 'display': 'none'},
                {'display': 'inline', 'float': 'right'},
                [{'bounds': 'Amplitude', 'lower': default_values[a_type]['lb'][0],
                  'upper': default_values[a_type]['ub'][0]},
                 {'bounds': 'Baseline Offset', 'lower': default_values[a_type]['lb'][1],
                  'upper': default_values[a_type]['ub'][1]},
                 {'bounds': 'ODMR Frequency (GHz)', 'lower': default_values[a_type]['lb'][2],
                  'upper': default_values[a_type]['ub'][2]},
                 {'bounds': 'Width (GHz)', 'lower': default_values[a_type]['lb'][3],
                  'upper': default_values[a_type]['ub'][3]}],
                {'display': 'inline-block', 'width': '600px'})
    if a_type == 'sine':
        return ({'display': 'none'}, '', 0, None, {'height': '700px', 'display': 'none'},
                {'display': 'inline', 'float': 'right'},
                [{'bounds': 'Amplitude', 'lower': default_values[a_type]['lb'][0],
                  'upper': default_values[a_type]['ub'][0]},
                 {'bounds': 'Baseline Offset', 'lower': default_values[a_type]['lb'][1],
                  'upper': default_values[a_type]['ub'][1]},
                 {'bounds': 'Oscillation Frequency', 'lower': default_values[a_type]['lb'][2],
                  'upper': default_values[a_type]['ub'][2]},
                 {'bounds': 'Decay Time (ns)', 'lower': default_values[a_type]['lb'][3],
                  'upper': default_values[a_type]['ub'][3]}],
                {'display': 'inline-block', 'width': '600px'})
    if a_type == 'none':
        return ({'display': 'none'}, '', None, None, {'height': '700px', 'display': 'none'},
                {'display': 'none', 'float': 'right'}, [],
                {'display': 'none', 'width': '600px'})


@app.callback(
    [Output('analysis-graph', 'figure'), Output('analysis-graph-data', 'data'),
     Output('fourier-graph', 'figure'), Output('a-error', 'children')],
    [Input('analyse_button', 'n_clicks')],
    [State('analysis-runs', 'value'), State('analysis-data-type', 'value'),
     State('analysis-type', 'value'), State('analysis-graph', 'figure'), State('graph-data', 'data'),
     State('analysis-graph-data', 'data'), State('delete-runs', 'value'),
     State('bounds-table', 'data'), State('threshold', 'value'), State('revivals', 'value')])
def update_analysis_graph(clicks, selected_run, data_type, analysis_type, graph, g_data, data, delete, bounds,
                          threshold, revival):
    data = data or {'analysis': [], 'fourier': [], 'plot-num': [], 'type': '', 'a_type': '', 'graph': None, 'opt': {},
                    'cov': {}}
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
        if selected_run != None:
            if data['fourier'] == []:
                plot_fourier = fouriertransform(selected_run, data_type, None)
                data['fourier'] = plot_fourier
            else:
                plot_fourier = fouriertransform(selected_run, data_type, go.Figure(data['fourier']))
                data['fourier'] = plot_fourier
        if delete != None:
            runs_to_delete = []
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
        lb = []
        for a in bounds:
            lb.append(a['lower'])
        totaldata = plotdata(g_data['type'], g_data['total-run'], g_data['normalise'])
        if data['type'] != data_type or '':
            data['analysis'] = []
            data['opt'] = {}
            data['cov'] = {}
        if selected_run != None:
            color_code = g_data['total-run'].index(selected_run) % 10
            for i in data['plot-num']:
                if i['run'] == selected_run:
                    i['num'] = i['num'] + 1
                    if analysis_type == 'lorentz':
                        totaldata = fitlorentzian(selected_run, data_type, go.Figure(graph),
                                                  lb, ub, i['num'], color_code)
                    elif analysis_type == 'exponent':
                        totaldata = find_T2(selected_run, data_type, go.Figure(graph), threshold,
                                            lb, ub, revival, i['num'], color_code)
                    elif analysis_type == 'sine':
                        totaldata = fitsinedamp(selected_run, data_type, go.Figure(graph),
                                                lb, ub, i['num'], color_code)
                    data['analysis'].append(totaldata[0]['data'][-1])
                    data['graph'] = totaldata[0]
                    data['opt'][totaldata[0]['data'][-1]['name']] = totaldata[1]
                    data['cov'][totaldata[0]['data'][-1]['name']] = np.sqrt(np.diag(totaldata[2]))
        if delete != None:
            runs_to_delete = []
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


@app.callback(Output('output-table', 'columns'),
              [Input('analysis-type', 'value')],
              [State('output-table', 'data')])
def update_table_columns(a_type, data):
    if a_type is None:
        raise PreventUpdate
    else:
        return table_columns[a_type]


@app.callback(Output('output-table', 'data'),
              [Input('analysis-graph-data', 'data'),
               Input('output-table', 'columns'),
               Input('analysis-data-type', 'value')])
def update_table_data(data, columns, a_type):
    if data is None:
        raise PreventUpdate
    table_data = []
    if data['type'] == a_type or '':
        for d in data['opt']:
            row = {}
            row['analysis'] = d.split(',')[0]
            row['amplitude-opt'] = round(data['opt'][d][0], 5)
            row['amplitude-cov'] = round(data['cov'][d][0], 5)
            row['baseline-opt'] = round(data['opt'][d][1], 5)
            row['baseline-cov'] = round(data['cov'][d][1], 5)
            if data['a_type'] == 'lorentz':
                row['odmr-opt'] = round(data['opt'][d][2], 5)
                row['odmr-cov'] = round(data['cov'][d][2], 5)
                row['width-opt'] = round(data['opt'][d][3], 5)
                row['width-cov'] = round(data['cov'][d][3], 5)
            elif data['a_type'] == 'sine':
                row['freq-opt'] = round(data['opt'][d][2], 5)
                row['freq-cov'] = round(data['cov'][d][2], 5)
                row['decay-opt'] = round(data['opt'][d][3], 5)
                row['decay-cov'] = round(data['cov'][d][3], 5)
            elif data['a_type'] == 'exponent':
                row['power-opt'] = round(data['opt'][d][2], 5)
                row['power-cov'] = round(data['cov'][d][2], 5)
                row['decay-opt'] = round(data['opt'][d][3], 5)
                row['decay-cov'] = round(data['cov'][d][3], 5)
            table_data.append(row)
    return table_data