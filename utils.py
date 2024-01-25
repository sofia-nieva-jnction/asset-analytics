import plotly.express as px
import pandas as pd
import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import sqlalchemy
import streamlit as st
import numpy as np
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression

@st.cache_data
def get_l0(route):
    df = pd.read_csv('l0.csv')
    return df[df.route==route]

@st.cache_data
def get_l1(route):
    df = pd.read_csv('l1.csv')
    return df[df.route==route]

@st.cache_data
def get_l2(route, asset_class):
    df = pd.read_csv('l2.csv')
    df_filtered = df[(df.group_class==asset_class) & (df.route==route)]
    return df_filtered.drop(columns=['group_class'])

@st.cache_data
def get_list_assets(asset_class):
    if asset_class=='S&C (Signalling) - Point Operating Equipment':
        ret = [557646, 557668, 765375, 1653200, 548731]
    else:
        ret = [563000, 884639, 884689, 884673, 884735, 884652, 563463, 840591, 840548]
    return ret

@st.cache_data
def get_work_orders(asset_number):
    df = pd.read_csv(f'work_orders_{asset_number}.csv')
    return df

@st.cache_data
def get_worst_perfoming_table(route, asset_class):
    df = pd.read_csv(f'worst_performing_table_{route}_{asset_class}.csv')
    return df

@st.cache_data
def get_data_example(asset_number, fault_number):
    radar = pd.read_csv(f'radar_{asset_number}_{fault_number}_eg.csv')
    timeline = pd.read_csv(f'timeline_{asset_number}_{fault_number}_eg.csv')
    day1 = datetime.datetime.strptime(min(timeline.Time), '%Y-%m-%d %H:%M:%S').date()
    ellipse_details = pd.read_csv(f'ellipse_details_{asset_number}_{fault_number}_eg.csv')
    faults_list = pd.read_csv(f'faults_list_{asset_number}_{fault_number}_eg.csv')
    return radar, day1, timeline, ellipse_details, faults_list

@st.cache_data
def get_berth_steps(asset_number, fault_number):
    return pd.read_csv(f'berth_steps_{asset_number}_{fault_number}_eg.csv')

@st.cache_data
def plot_vertical_histograms(table, y_col, y_name):
    cat_order = table.sort_values('fms_failures_count_6m_2020', ascending=True)[y_col].to_list()

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, 
                        subplot_titles=['Number of Failures (Jan - Jun 2020)', 'Percentage of Assets with RADAR data'])

    fig.add_trace(go.Bar(y=table[y_col], x=table['fms_failures_count_6m_2020'],
                            hovertemplate="Total Failures: %{x}<br><extra></extra>", showlegend=False,
                            # text=table['fms_failures_count_6m_2020'], textposition='auto'
                            ), 
                row=1, col=1)
    fig.add_trace(go.Bar(y=table[y_col], x=table['count_service_affecting_faults_6m_2020'],
                            hovertemplate="Service Affecting<br>Failures: %{x}<br><extra></extra>", showlegend=False,
                            # text=table['count_service_affecting_faults_6m_2020'], textposition='auto'
                            ),
                row=1, col=1)
    fig.add_trace(go.Bar(y=table[y_col], x=table['has_radar_percentage']/100, marker_color='green', opacity=0.7,
                            hovertemplate="Assets with RADAR<br>data: %{x}<br><extra></extra>", showlegend=False,
                            # text=table['has_radar_percentage']/100, textposition='auto'
                            ), 
                row=1, col=2)
    
    fig.update_yaxes(title=y_name, row=1, col=1, showgrid=True, showline=True) 
    fig.update_xaxes(tickformat = '.0%', row=1, col=2, showgrid=True) 
    fig.update_yaxes(row=1, col=2, showgrid=True, showline=True) 
    fig.update_xaxes(row=1, col=1, showgrid=True) 
    fig.update_traces(orientation = 'h')
    fig.update_layout(barmode="overlay", bargap=0.1,
                      yaxis={'categoryorder': 'array', 'categoryarray':cat_order},
                      height=400, margin=dict(l=20,r=0,b=0,t=20))
    
    return fig

@st.cache_data
def highlight_threshold(cell, condition, threshold):
    attr = 'background-color: {}'.format('rgba(229, 40, 23, 0.2)')
    if condition=='Greater than':
        highlight = (cell>threshold)
    elif condition=='Less than':
        highlight = (cell<threshold)
    elif condition=='Equal to':
        highlight = (cell==threshold)
    elif condition=='Greater than or Equal to':
        highlight = (cell>=threshold)
    elif condition=='Less than or Equal to':
        highlight = (cell<=threshold) 
    return attr if highlight else ''
    
@st.cache_data
def get_trends_table(route, asset_class, attribute_line_chart):
    df = pd.read_csv(f'trends_table_{route}_{asset_class}_{attribute_line_chart}.csv')
    return df

@st.cache_data
def get_radar_data(asset_number):
    df = pd.read_csv(f'radar_{asset_number}.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

@st.cache_data
def get_radar_summary(asset_number):
    df = pd.read_csv(f'radar_summary_{asset_number}.csv')
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df

@st.cache_data
def get_ellipse_details(asset_number):
    df = pd.read_csv(f'ellipse_details_{asset_number}.csv')
    return df

@st.cache_data
def get_faults_list(asset_number):
    df = pd.read_csv(f'faults_list_{asset_number}.csv')
    return df

@st.cache_data
def get_all_faults_timeline(asset_number, faults_number):
    timeline = get_fault_timeline(asset_number, faults_number[0])
    for f in faults_number[1:]:
        timeline = pd.concat([timeline, get_fault_timeline(asset_number, f)])
    return timeline

@st.cache_data
def get_fault_timeline(asset_number, fault_number):
    timeline = pd.read_csv(f'timeline_{asset_number}_{fault_number}.csv')
    return timeline

@st.cache_data
def timeline_plot(timeline, fault_number, asset_number):
    df = timeline.copy()
    df['temp'] = 0
    df['Time'] = pd.to_datetime(df['Time'], format='%Y/%m/%d %H:%M:%S')

    fig = px.scatter(df, x='Time', y='temp',
                    title=f'Timeline of Failure {fault_number} (Asset {asset_number})',
                    template='plotly_white', custom_data=['Event']
                    )

    fig.update_xaxes(type='date', minor=dict(showgrid=True), title=None)
    fig.update_yaxes(showgrid=False, showticklabels=False, title=None)

    fig.update_layout(height=250, title_yanchor ='top', title_y=0.75)
    fig.update_traces(hovertemplate='Event: %{customdata[0]} <br>Time: %{x|%Y/%m/%d %H:%M:%S}')
    
    return fig

@st.cache_data
def radar_trace_plot(asset_class, radar, day_start, day_end, timeline, attribute, other):
    asset = radar.iloc[0]['asset']
    radar_day = radar[(radar.datetime >= str(day_start)) & (radar.datetime <= str(pd.to_datetime(day_end) + datetime.timedelta(days=1)))].copy().sort_values('datetime')

    if asset_class == 'S&C (Signalling) - Point Operating Equipment':
        attribute2 = attribute[:-2] + attribute[-2:][::-1]
        attributes = [attribute, attribute2]
        attributes.sort()
        if other is not None and other.split('_')[-1] in ['Average', 'Peak']:
            attributes = attributes + [other]
        df = radar_day[radar_day.attribute.isin(attributes)]
        df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M:%S.%f')
        fig = px.scatter(df, x='datetime', y='value', labels={'value': f"{attribute.replace('_', ' ')} Value", 'datetime':'Time'}, color='attribute',
                     template='plotly_white', color_discrete_map=dict(zip(attributes, ['blue', 'green', 'red'])))
        if other is not None and (other.split('_')[-1] == 'Length' or other=='Movement_Direction_Indicator'):
            subfig = make_subplots(specs=[[{"secondary_y": True}]])
            trace_name = f"{other.replace('_', ' ')}" if other=='Movement_Direction_Indicator' else f"{other.replace('_', ' ')} (seconds)"
            fig2 = px.scatter(radar_day[radar_day['attribute']==other], x='datetime', y='value', labels={'value': trace_name, 'datetime':'Time'}, color_discrete_sequence=['red'])
            fig2.update_traces(yaxis="y2")
            subfig.add_traces(fig.data + fig2.data)
            subfig.layout.yaxis2.title = trace_name
            fig = subfig
        
    else:
        df = radar_day[radar_day.attribute.isin([attribute])]
        df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M:%S.%f')
        fig = px.scatter(df, x='datetime', y='value', labels={'value': f"{attribute.replace('_', ' ')} Value", 'datetime':'Time'}, 
                     template='plotly_white')

    fig.update_xaxes(
        type='date',
        tick0=datetime.datetime.strptime('2020/01/01 00:00:00.000', '%Y/%m/%d %H:%M:%S.%f'),
    )

    fig.update_traces(marker_size=3,
                      mode='lines+markers',
                      line_width=1,
                     )

    day_timeline = timeline[(timeline['Time'] >= str(day_start)) & (timeline['Time'] <= str(pd.to_datetime(day_end) + datetime.timedelta(days=1)))]
    count_vlines = 0
    for i, row in day_timeline.iterrows():
        fig.add_vline(x=pd.to_datetime(row['Time']))
        fig.add_annotation(x=pd.to_datetime(row['Time']), xanchor='left', 
                        y=1-i*0.1, yref="paper", font_size=18, font_color='black',
                        text=f"<b>{row['Event']}</b>", showarrow=False)
        count_vlines = i
        
    fig.update_layout(height=450, width=1060,
                     title_yanchor ='top', #title_y=0.85,
                     font_size=14)
    title_attribute = attribute[:-3] if asset_class == 'S&C (Signalling) - Point Operating Equipment' else attribute
    fig.update_layout(title_text=f"{title_attribute.replace('_', ' ')} data from RADAR - Asset Reference {asset}", 
                      title_font_size=18)
    
    if other is None or (other.split('_')[-1] not in ['Average', 'Peak', 'Length'] and other!='Movement_Direction_Indicator'):
        for d in list(pd.to_datetime(radar_day[radar_day['attribute']==other]['datetime'])):
            fig.add_vline(x=d, line_color='darkmagenta', line_width=1)
        if other is not None:
            fig.add_annotation(x=d, xanchor='left', font_color='darkmagenta',
                            y=1-(count_vlines+1)*0.1, yref="paper", font_size=18,
                            text=f"<b>{other.replace('_', ' ')}</b>", showarrow=False)

    return fig


def headcodes_plot(radar, day, timeline, attribute, other, berth_steps, berth, td):
    asset = radar.iloc[0]['asset']
    radar_day = radar[(radar.datetime > str(day)) & (radar.datetime < str(day + datetime.timedelta(days=1)))].copy().sort_values('datetime')

    df = radar_day[radar_day.attribute.isin([attribute])]
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y/%m/%d %H:%M:%S.%f')
    
    
    fig = make_subplots(rows=2, cols=1, row_heights=[5,1.2], shared_xaxes=True, vertical_spacing=0.05)

    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['value'], name=f"{attribute.replace('_', '<br>')}", line_color='#636EFA',
                   hovertemplate=f"<b>{attribute.replace('_', ' ')}</b><br>" + "Value: %{y}<br>" + "Time: %{x|%H:%M:%S.%f}" + "<extra></extra>"
                  ),
        row=1, col=1
    )
    fig.update_xaxes(
        type='date',
        tick0=datetime.datetime.strptime('2020/01/01 00:00:00.000', '%Y/%m/%d %H:%M:%S.%f'),
        row=1, col=1
    )    
    fig.update_traces(marker_size=3,
                      mode='lines+markers',
                      line_width=1,
                      row=1, col=1
                     )
    fig.update_yaxes(title_text=f"{attribute.replace('_', ' ')} Value", row=1, col=1)


    berth_steps_day = berth_steps[(berth_steps.message_datetime > str(day)) 
                                  & (berth_steps.message_datetime < str(day + datetime.timedelta(days=1)))].copy().sort_values('message_datetime')
    berth_steps_day['message_datetime'] = pd.to_datetime(berth_steps_day['message_datetime']) #
    entry = berth_steps_day[berth_steps_day['to_berth']==berth]
    st.dataframe(entry)
    exit = berth_steps_day[berth_steps_day['from_berth']==berth]
    entry_exit = entry.merge(exit, on=['headcode', 'headcode_hour'])
    entry_exit['runtime'] = entry_exit['message_datetime_y'] - entry_exit['message_datetime_x'] 
    entry_exit['runtime_s'] = entry_exit['runtime'].apply(lambda x: str(int(x.total_seconds())))
    st.dataframe(entry_exit['message_datetime_x']) #

    fig.add_trace(
        go.Scatter(x=entry_exit['message_datetime_x'], y=[0]*len(entry_exit),
                   name='Train', line_color='darkgreen', text='<b>' + entry_exit['headcode'] + f'</b><br>Berth {td+berth}<br>Runtime: ' + entry_exit['runtime_s'],
                   hovertemplate="<b>Headcode</b> %{text}<br>" +
                                 "Entry Time: %{x|%H:%M:%S}" +
                                 "<extra></extra>",),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=entry_exit['message_datetime_y'], y=[0]*len(entry_exit),
                   name='Train', line_color='darkgreen', text='<b>' + entry_exit['headcode'] + f'</b><br>Berth {td+berth}<br>Runtime: ' + entry_exit['runtime_s'],
                   hovertemplate="<b>Headcode</b> %{text}<br>" +
                                 "Exit Time: %{x|%H:%M:%S}" +
                                 "<extra></extra>",),
        row=2, col=1
    )
    fig.update_traces(marker_size=7,#8,
                      mode='markers',
                      marker_symbol='diamond-tall',#'triangle-up',#'arrow',
                      row=2, col=1
                      )
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, title_text='Trains', row=2, col=1)
 
    
    for train in list(berth_steps_day.headcode_hour.unique()):
        temp = berth_steps_day[berth_steps_day.headcode_hour==train]
        fig.add_trace(
        go.Scatter(x=temp['message_datetime'], y=[0]*len(temp),
                   line_color='darkgreen', mode='lines', line_width=1, hoverinfo='skip'),
        row=2, col=1
    )
    
    day_timeline = timeline[(timeline['Time'] > str(day)) & (timeline['Time'] < str(day + datetime.timedelta(days=1)))]
    count_vlines = 0
    for i, row in day_timeline.iterrows():
        fig.add_vline(x=pd.to_datetime(row['Time']), line_color='grey', row=1, col=1)
        fig.add_annotation(x=pd.to_datetime(row['Time']), xanchor='left', 
                        y=1-i*0.1, yref="y domain", font_size=18, font_color='black',
                        text=f"<b>{row['Event']}</b>", showarrow=False, row=1, col=1)
        count_vlines = i
        
    fig.update_layout(height=600, width=1060,
                     title_yanchor ='top', #title_y=0.85,
                     font_size=14)
    fig.update_layout(title_text=f"{attribute.replace('_', ' ')} data from RADAR - Asset Reference {asset} - Berth {td+berth}", 
                      title_font_size=18, showlegend=False)
    
    ts = list(pd.to_datetime(radar_day[radar_day['attribute']==other]['datetime']))
    for t in ts:
        fig.add_vline(x=t, line_color='darkmagenta', line_width=1, row=1, col=1)
    if other is not None:
        fig.add_annotation(x=ts[-1], xanchor='left', font_color='darkmagenta',
                           y=1-(count_vlines+1)*0.1, yref="y domain", font_size=18,
                           text=f"<b>{other.replace('_', ' ')}</b>", showarrow=False, row=1, col=1)
    
    fig.update_xaxes(spikemode='across', spikecolor='black', spikethickness=-2)
    fig.update_traces(xaxis="x2")
    
    return fig

def alarms_near_failures(faults_list, radar, work_orders_asset, d=14):
    radar_filtered = radar.copy()
    radar_filtered['time'] = pd.to_datetime(radar_filtered.datetime).apply(lambda x: str(x.time()))
    radar_filtered = radar_filtered[radar_filtered.time>='06:00:00']
    asset_attributes = radar_filtered['attribute'].unique()
    alarms_types = [x for x in asset_attributes if (x not in ['Total_Occupations_Count', 'Circuit_Current'])]
    alarms = radar_filtered[radar_filtered.attribute.isin(alarms_types)].reset_index(drop=True).copy()
    alarms['attribute'] = alarms['attribute'].replace({'High_Occupied_Current_Count_Value': 'High_Occupied_Current_Count', 
                                                    'Clear_Occupied_Clear_Flick_Count': 'COC_Flick_Count',
                                                    'Occupied_Clear_Occupied_Flick_Count': 'OCO_Flick_Count',
                                                    'Poor_Shunt_Count_Value': 'Poor_Shunt_Count',
                                                    'Unstable_Clear_Current_Count_Value': 'Unstable_Clear_Current_Count'})
    alarms = alarms.drop(columns='value').drop_duplicates()
    alarms_types = pd.Series(alarms_types).replace({'High_Occupied_Current_Count_Value': 'High_Occupied_Current_Count', 
                                                    'Clear_Occupied_Clear_Flick_Count': 'COC_Flick_Count',
                                                    'Occupied_Clear_Occupied_Flick_Count': 'OCO_Flick_Count',
                                                    'Poor_Shunt_Count_Value': 'Poor_Shunt_Count',
                                                    'Unstable_Clear_Current_Count_Value': 'Unstable_Clear_Current_Count'}
                                                    ).drop_duplicates().tolist()
    alarms_dict = {'Date': [], 'Fault Number': [], 'Days since last Fault': [], 'Days since last Work Order': []}
    for alarm in alarms_types:
        column_name = alarm.replace('_', ' ').replace('Count', '')
        alarms_dict[column_name] = []
    faults_list['occurred_date'] = pd.to_datetime(faults_list['occurred_date']).dt.date
    for i, row in faults_list.iterrows():
        if i == 0:
            alarms_dict['Days since last Fault'].append('-')
        else:
            alarms_dict['Days since last Fault'].append(str((row['occurred_date']-faults_list.iloc[i-1]['occurred_date']).days) + ' days')
        wo = work_orders_asset[(pd.to_datetime(work_orders_asset['completed_date']).dt.date<row['occurred_date'])
                               | ((pd.to_datetime(work_orders_asset['completed_date']).dt.date==row['occurred_date']) & (~work_orders_asset['planned_start_date'].isna()))
                               ] ## Hopefully this would capture if there was a work order during the night previous to a failure but not the work order of the failure itself
        if len(wo) == 0:
            alarms_dict['Days since last Work Order'].append('-')
        else:
            alarms_dict['Days since last Work Order'].append(str((row['occurred_date']-pd.to_datetime(wo['completed_date']).dt.date.max()).days) + ' days') 
        alarms_dict['Date'].append(row['occurred_date'])
        alarms_dict['Fault Number'].append(row['fault_number'])
        row['failed_datetime'] = pd.to_datetime(row['failed_datetime'])
        alarms['datetime'] = pd.to_datetime(alarms['datetime'])
        for alarm in alarms_types:
            column_name = alarm.replace('_', ' ').replace('Count', '')
            alarms_dict[column_name].append(
                                    sum((alarms.attribute==alarm)
                                      & (alarms.datetime < row['failed_datetime'])
                                      & (alarms.datetime >= row['failed_datetime'] - datetime.timedelta(days=d))))
    return pd.DataFrame(alarms_dict)

@st.cache_data
def plot_max_smoothed_and_count_tc(date_start, date_end, attribute, count_attribute, radar, faults_list, work_orders,
                                   class_attributes_other, h=24, add_trend=False, trend_start=None, trend_end=None):
    date_start = pd.to_datetime(date_start)
    date_end = pd.to_datetime(date_end)
    radar_days = radar[(radar.datetime > str(date_start)) & (radar.datetime < str(date_end))].copy().sort_values('datetime')
    radar_days['time'] = pd.to_datetime(radar_days.datetime).apply(lambda x: str(x.time()))
    radar_days = radar_days[radar_days.time>='06:00:00']
    faults_list = faults_list[(faults_list.failed_datetime > str(date_start))
                              & (faults_list.failed_datetime < str(date_end))].copy().sort_values('failed_datetime')
    
    df = radar_days[radar_days.attribute.isin([attribute])].reset_index(drop=True).copy()
    df_count = radar_days[radar_days.attribute.isin([count_attribute])].reset_index(drop=True).copy()
    df_count['datetime'] = pd.to_datetime(df_count['datetime'])
    df_count['date'] = df_count['datetime'].apply(lambda x: x.date())#df_count['datetime'].apply(lambda x: str(x.date()) + ' ' + str(x.hour).zfill(2) + ':00:00.000')
    
    signal = df['value'].to_numpy()

    maxs, _ = find_peaks(signal, height=100)
    mins, _ = find_peaks(-signal, height=[-10, 30])

    df['is_max'] = False
    df['is_min'] = False
    df['is_extremum'] = None
    df.loc[maxs, 'is_max'] = True
    df.loc[mins, 'is_min'] = True
    df.loc[maxs, 'is_extremum'] = 'max'
    df.loc[mins, 'is_extremum'] = 'min'
    
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y/%m/%d %H:%M:%S.%f')
    df_smoothed_maxs = df[df.is_max][['datetime', 'value']].rolling(f'{h}H', center=True, on='datetime').mean()
    df_smoothed_mins = df[df.is_min][['datetime', 'value']].rolling(f'{h}H', center=True, on='datetime').mean()

    df_merged = df.merge(df_smoothed_maxs, on='datetime', how='left', suffixes=('', '_smoothed_max'))
    df_merged = df_merged.merge(df_smoothed_mins, on='datetime', how='left', suffixes=('', '_smoothed_min'))
    df_merged['value_smoothed'] = df_merged['value_smoothed_max'].combine_first(df_merged['value_smoothed_min'])

    q = 'Max' if attribute=='Circuit_Current' else 'Average'
    fig = make_subplots(rows=2, cols=1, row_heights=[5,2], shared_xaxes=True, vertical_spacing=0.1)

    hovertemplate="Value: %{y}<br><extra></extra>" #f"<b>{attribute.replace('_', ' ')}</b><br>" + "Value: %{y}<br>" + "<extra></extra>"
    fig.add_trace(go.Scatter(x=df_merged[df_merged.is_max]['datetime'],
                             y=df_merged[df_merged.is_max]['value_smoothed'],
                             line_color='red', name='Max', hovertemplate=hovertemplate, showlegend=False), 
                 row=1, col=1)
    
    fig.update_xaxes(
            type='date',
            tick0=datetime.datetime.strptime('2020/01/01 00:00:00.000', '%Y/%m/%d %H:%M:%S.%f'),
        )
    fig.update_yaxes(title='Current', row=1, col=1)  
    

    fig.update_traces(marker_size=3,
                          mode='lines',#+markers',
                          line_width=1,
                         )
    
    hovertemplate="%{x|%a %d %b}<br>Count: %{y}<br><extra></extra>" 
    fig.add_trace(go.Histogram(x=df_count['date'], 
                               xbins=dict(size=86400000, start=datetime.datetime.strptime('2020/01/01 00:00:00.000', '%Y/%m/%d %H:%M:%S.%f'),), autobinx=False,
                              # xbins=dict(size=3600000), autobinx=False,
                               showlegend=False, marker_color='darkgreen', opacity=0.5,
                               hovertemplate=hovertemplate
                              )
                  , row=2, col=1)  
    fig.update_yaxes(title='Count', row=2, col=1)  

    
    fig.update_layout(height=600, width=1060,
                      title_yanchor ='top', #title_y=0.85,
                      template='plotly_white',
                      title_text=f"{q} {attribute.replace('_', ' ')} - Average over {h} hour windows / {count_attribute.replace('_', ' ')} per day"
                    )

    faults_list['failed_datetime'] = pd.to_datetime(faults_list['failed_datetime'])                                  
    for i, row in faults_list.iterrows():
        if i<len(faults_list)-1:
            next_failure = pd.to_datetime(faults_list.loc[i+1, 'failed_datetime']).date()
            cond = (next_failure - row['failed_datetime'].date() > pd.to_timedelta('20D'))
        else: 
            cond = True
        fig.add_vline(x=pd.to_datetime(row['failed_datetime']), line_color='grey')
        fig.add_annotation(x=pd.to_datetime(row['failed_datetime']),
                           xanchor='left' if cond else 'right', 
                           align='left' if cond else 'right', 
                           y=1, yref="paper", font_size=14, font_color='black',
                           text=f"Fault<br><b>{row['fault_number']}</b><br><br>" 
                              + f"{row['failed_datetime'].strftime('%d %b')}<br>"
                              + f"{row['failed_datetime'].strftime('%H:%M')}<br><br>"
                              #+ ("<b>Service<br>Affecting</b>" if row['is_service_affecting'] else  "")
                              , showarrow=False)
        
    ###### Alarms ######
        
    m = df_merged[df_merged.is_max]['value_smoothed'].min()#df_smoothed_nr['value'].min()
    M = df_merged[df_merged.is_max]['value_smoothed'].max()#df_smoothed_nr['value'].max()
   
    asset_attributes = radar_days['attribute'].unique()
    alarms_types = [x for x in class_attributes_other if ((x not in [count_attribute]) and (x in asset_attributes))]
    alarms = radar_days[radar_days.attribute.isin(alarms_types)].reset_index(drop=True).copy()
    alarms['attribute'] = alarms['attribute'].replace({'High_Occupied_Current_Count_Value': 'High_Occupied_Current_Count', 
                                                       'Clear_Occupied_Clear_Flick_Count': 'COC_Flick_Count',
                                                       'Occupied_Clear_Occupied_Flick_Count': 'OCO_Flick_Count',
                                                       'Poor_Shunt_Count_Value': 'Poor_Shunt_Count',
                                                       'Unstable_Clear_Current_Count_Value': 'Unstable_Clear_Current_Count'})
    #alarms['attribute'] = alarms['attribute'].apply(lambda x: x.replace('_Value', '').replace('Clear_Occupied_Clear', 'COC').replace('Occupied_Clear_Occupied', 'OCO'))
    alarms_types = pd.Series(alarms_types).replace({'High_Occupied_Current_Count_Value': 'High_Occupied_Current_Count', 
                                                       'Clear_Occupied_Clear_Flick_Count': 'COC_Flick_Count',
                                                       'Occupied_Clear_Occupied_Flick_Count': 'OCO_Flick_Count',
                                                       'Poor_Shunt_Count_Value': 'Poor_Shunt_Count',
                                                       'Unstable_Clear_Current_Count_Value': 'Unstable_Clear_Current_Count'})
    color_seq_alarms = px.colors.qualitative.Prism
    # st.write(alarms_types.drop_duplicates())
    for i, type in enumerate(alarms_types.drop_duplicates()):
        alarm = alarms[alarms.attribute==type].copy()
        n = 20
        to_plot = [None]*(len(alarm)*2*n)
        for j in range(n):
            to_plot[j::(2*n)] = alarm.datetime.tolist()
        print(to_plot)
        fig.add_trace(go.Scatter(x=to_plot, 
                                 y=np.linspace(m, M, n).tolist()*(len(alarm)*2),
                                 mode="lines", 
                                 line_color=color_seq_alarms[i],
                                 name=type.replace('_', ' ').replace('Count', ''), 
                                 showlegend=True, 
                                 visible='legendonly',
                                 legendgroup="alarms", 
                                 legendgrouptitle_text="Alarms",
                                 connectgaps=False, 
                                 line_width=2,
                                 hovertemplate="%{x|%d %b}<br>%{x|%H:%M:%S}" ))
        
    ###### Trends ######
        
    color_seq_trends = px.colors.qualitative.Antique[3:]
    for i, d in enumerate([15, 30, 60, 90]):
        try:
            lr_start = pd.to_datetime(date_end) - datetime.timedelta(days=d)
            df_lr = df_smoothed_maxs[df_smoothed_maxs['datetime']>=lr_start].copy()
            df_lr['time_ms'] = (df_lr['datetime'] - lr_start) / datetime.timedelta(seconds=1)
            df_lr['x'] = df_lr['time_ms']/df_lr['time_ms'].max()
            
            x = df_lr['x'].tolist()
            y = df_lr['value']
            model = LinearRegression().fit(np.array(x).reshape(-1, 1), np.array(y))
            change = model.coef_[0]/(x[0]*model.coef_[0] + model.intercept_)
            sign = '+' if change > 0 else ''
            change_text = f"{sign}{change:.2%}"
    
            x_plot = [df_lr['datetime'].min(), df_lr['datetime'].max()]
            y_plot = [x[0]*model.coef_[0] + model.intercept_, x[-1]*model.coef_[0] + model.intercept_]
    
            fig.add_trace(go.Scatter(x=x_plot,
                                     y=y_plot,
                                     mode='lines',
                                     line_color=color_seq_trends[i],
                                     name=f'Last {d} Days: Change {change_text}',
                                     showlegend=True,
                                     visible='legendonly',
                                     legendgroup="trends", 
                                     legendgrouptitle_text="Recent Trends",
                                     line_width=2,
                                     line_dash='dash',
                                     hovertemplate="%{x|%d %b}<br>%{y:.3f}<br><extra></extra>"), 
                            row=1, col=1)
        except:
            pass
        
    ###### Work Orders ######
        
    m = df_merged[df_merged.is_max]['value_smoothed'].min()#df_smoothed_nr['value'].min()
    M = df_merged[df_merged.is_max]['value_smoothed'].max()#df_smoothed_nr['value'].max()
   
    color_seq_alarms = px.colors.qualitative.Vivid[7]
    to_plot = []
    text_hover =[]
    for i, day in enumerate(work_orders.completed_date.unique().tolist()):  # I should check that there is no more than one record per work order number 
        number_day = work_orders[work_orders.completed_date==day]['work_order_number'].astype(str).tolist()
        number_text = ('Work Order<br>' if len(number_day)==1 else 'Work Orders<br>') + ('<br>').join(number_day)
        n = 20
        to_plot.append([None]*(2*n))
        text_hover.append([None]*(2*n))
        for j in range(n):
            to_plot[i][j::(2*n)] = [day]
            text_hover[i][j::(2*n)] = [number_text]
    to_plot = np.array(to_plot).flatten().tolist()
    text_hover = np.array(text_hover).flatten().tolist()
    fig.add_trace(go.Scatter(x=to_plot, 
                             y=np.linspace(m, M, n).tolist()*(len(work_orders.completed_date.unique())*2),
                             mode="lines", 
                             line_color=color_seq_alarms,
                             name='Completed Date', 
                             showlegend=True, 
                             #visible='legendonly',
                             legendrank=1,
                             legendgroup="work_orders", 
                             legendgrouptitle_text="Work Orders",
                             connectgaps=False, 
                             line_width=2,
                             line_dash='dash',
                             text=text_hover, 
                             hovertemplate="%{x|%d %b}<br><br>%{text}" ))
    
    if add_trend:
        fig, custom_trend_change_text = add_trend_to_plot(fig, df_smoothed_maxs, trend_start, trend_end)
    else:
        custom_trend_change_text = None


    fig.update_layout(legend_groupclick="toggleitem")#legend_title_text='Alarms')
    
    return fig, custom_trend_change_text

@st.cache_data
def add_trend_to_plot(fig, df_smoothed_maxs, trend_start, trend_end):
    color_seq_trends = px.colors.qualitative.Antique[0]
    lr_start = pd.to_datetime(trend_start) 
    lr_end = pd.to_datetime(trend_end) 
    df_lr = df_smoothed_maxs[(df_smoothed_maxs['datetime']>=lr_start) & (df_smoothed_maxs['datetime']<=lr_end + datetime.timedelta(days=1))].copy()
    df_lr['time_ms'] = (df_lr['datetime'] - lr_start) / datetime.timedelta(seconds=1)
    df_lr['x'] = df_lr['time_ms']/df_lr['time_ms'].max()
    
    x = df_lr['x'].tolist()
    y = df_lr['value']
    model = LinearRegression().fit(np.array(x).reshape(-1, 1), np.array(y))
    change = model.coef_[0]/(x[0]*model.coef_[0] + model.intercept_)
    sign = '+' if change > 0 else ''
    change_text = f"{sign}{change:.2%}"

    x_plot = [df_lr['datetime'].min(), df_lr['datetime'].max()]
    y_plot = [x[0]*model.coef_[0] + model.intercept_, x[-1]*model.coef_[0] + model.intercept_]

    fig.add_trace(go.Scatter(x=x_plot,
                             y=y_plot,
                             mode='lines',
                             line_color=color_seq_trends,
                             name=f'{trend_start} to {trend_end}:<br>Change {change_text}',
                             showlegend=True,
                             #visible='legendonly',
                             legendgroup="other_trends", 
                             legendgrouptitle_text="Custom Trend Period",
                             line_width=3,
                             line_dash='dashdot',
                             hovertemplate="%{x|%d %b}<br>%{y:.3f}<br><extra></extra>"), 
                    row=1, col=1)

    return fig, change_text

@st.cache_data
def plot_max_smoothed_and_count_points(date_start, date_end, attribute, count_attribute, radar, faults_list, work_orders,
                                       class_attributes_other, h=24, add_trend=False, trend_start=None, trend_end=None):
    date_start = pd.to_datetime(date_start)
    date_end = pd.to_datetime(date_end)
    radar_days = radar[(radar.datetime > str(date_start)) & (radar.datetime < str(date_end))].copy().sort_values('datetime')
    radar_days['time'] = pd.to_datetime(radar_days.datetime).apply(lambda x: str(x.time()))
    radar_days = radar_days[radar_days.time>='06:00:00']
    faults_list = faults_list[(faults_list.failed_datetime > str(date_start))
                              & (faults_list.failed_datetime < str(date_end))].copy().sort_values('failed_datetime')
    
    df = radar_days[radar_days.attribute.isin([attribute])].reset_index(drop=True).copy()
    df_count = radar_days[radar_days.attribute.isin([count_attribute])].reset_index(drop=True).copy()
    df_count['datetime'] = pd.to_datetime(df_count['datetime'])                                   
    df_count['date'] = df_count['datetime'].apply(lambda x: x.date())#df_count['datetime'].apply(lambda x: str(x.date()) + ' ' + str(x.hour).zfill(2) + ':00:00.000')
    # alarms_types = [x for x in class_attributes_other if x not in [count_attribute]]
    # alarms = radar_days[radar_days.attribute.isin(alarms_types)].reset_index(drop=True).copy()

    h = h
    df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M:%S.%f')
    df_smoothed_nr = df[df.attribute==attribute][['datetime', 'value']].rolling(f'{h}H', center=True, on='datetime').mean()
    # df_smoothed_rn = df[df.attribute=='Current_Waveform_RN_Average'][['datetime', 'value']].rolling(f'{h}H', center=True, on='datetime').mean()

    q = 'Average' #Max' if attribute=='Circuit_Current' else 'Average'
    fig = make_subplots(rows=2, cols=1, row_heights=[5,2], shared_xaxes=True, vertical_spacing=0.1)

    hovertemplate="Value: %{y}<br><extra></extra>" #f"<b>{attribute.replace('_', ' ')}</b><br>" + "Value: %{y}<br>" + "<extra></extra>"
    fig.add_trace(go.Scatter(x=df_smoothed_nr['datetime'],
                             y=df_smoothed_nr['value'],
                             #labels={'value_smoothed': f"Value", 'datetime':'Time'}, 
                             line_color='red', name=q, hovertemplate=hovertemplate, showlegend=False), 
                 row=1, col=1)
    
    fig.update_xaxes(
            type='date',
            tick0=datetime.datetime.strptime('2020/01/01 00:00:00.000', '%Y/%m/%d %H:%M:%S.%f'),
        )
    
    y_name = attribute.split('_')[-1] if attribute.split('_')[-1]=='Average' else 'Avg ' + attribute.split('_')[-1]
    fig.update_yaxes(title=f'{y_name} Current Waveform', row=1, col=1)  
    
    fig.update_traces(marker_size=3,
                          mode='lines',#+markers',
                          line_width=1,
                         )
    
    hovertemplate="%{x|%a %d %b}<br>Count: %{y}<br><extra></extra>" 
    fig.add_trace(go.Histogram(x=df_count['date'], 
                               xbins=dict(size=86400000), autobinx=False,
                              # xbins=dict(size=3600000), autobinx=False,
                               showlegend=False, marker_color='darkgreen', opacity=0.5,
                               hovertemplate=hovertemplate
                              )
                  , row=2, col=1)  
    fig.update_yaxes(title='Count', row=2, col=1)  

    
    fig.update_layout(height=500, width=1060,
                      title_yanchor ='top', #title_y=0.85,
                      template='plotly_white',
                      title_text=f"{attribute.replace('_', ' ')} - Average over {h} hour windows / {count_attribute.replace('_', ' ')} per day"
                    )

    faults_list['failed_datetime'] = pd.to_datetime(faults_list['failed_datetime'])    
    for i, row in faults_list.iterrows():
        if i<len(faults_list)-1:
            next_failure = pd.to_datetime(faults_list.loc[i+1, 'failed_datetime']).date()
            cond = (next_failure - row['failed_datetime'].date() > pd.to_timedelta('20D'))
        else: 
            cond = True
        fig.add_vline(x=pd.to_datetime(row['failed_datetime']), line_color='grey')
        fig.add_annotation(x=pd.to_datetime(row['failed_datetime']),
                           xanchor='left' if cond else 'right', 
                           align='left' if cond else 'right', 
                           y=1, yref="paper", font_size=14, font_color='black',
                           text=f"Fault<br><b>{row['fault_number']}</b><br><br>" 
                              + f"{row['failed_datetime'].strftime('%d %b')}<br>"
                              + f"{row['failed_datetime'].strftime('%H:%M')}<br><br>"
                              #+ ("<b>Service<br>Affecting</b>" if row['is_service_affecting'] else  "")
                              , showarrow=False)
        
    ###### Trends ######
        
    color_seq_trends = px.colors.qualitative.Antique[3:]
    for i, d in enumerate([15, 30, 60, 90]):
        try:
            lr_start = pd.to_datetime(date_end) - datetime.timedelta(days=d)
            df_lr = df_smoothed_nr[df_smoothed_nr['datetime']>=lr_start].copy()
            df_lr['time_ms'] = (df_lr['datetime'] - lr_start) / datetime.timedelta(seconds=1)
            df_lr['x'] = df_lr['time_ms']/df_lr['time_ms'].max()
            
            x = df_lr['x'].tolist()
            y = df_lr['value']
            model = LinearRegression().fit(np.array(x).reshape(-1, 1), np.array(y))
            change = model.coef_[0]/(x[0]*model.coef_[0] + model.intercept_)
            sign = '+' if change > 0 else ''
            change_text = f"{sign}{change:.2%}"
    
            x_plot = [df_lr['datetime'].min(), df_lr['datetime'].max()]
            y_plot = [x[0]*model.coef_[0] + model.intercept_, x[-1]*model.coef_[0] + model.intercept_]
    
            fig.add_trace(go.Scatter(x=x_plot,
                                     y=y_plot,
                                     mode='lines',
                                     line_color=color_seq_trends[i],
                                     name=f'Last {d} Days: Change {change_text}',
                                     showlegend=True,
                                     visible='legendonly',
                                     legendgroup="trends", 
                                     legendgrouptitle_text="Recent Trends",
                                     line_width=2,
                                     line_dash='dash',
                                     hovertemplate="%{x|%d %b}<br>%{y:.3f}<br><extra></extra>"), 
                            row=1, col=1)
        except:
            pass
        
    ###### Work Orders ######
        
    m = df_smoothed_nr['value'].min()#df_smoothed_nr['value'].min()
    M = df_smoothed_nr['value'].max()#df_smoothed_nr['value'].max()
   
    color_seq_alarms = px.colors.qualitative.Vivid[7]
    to_plot = []
    text_hover =[]
    for i, day in enumerate(work_orders.completed_date.unique().tolist()):  # I should check that there is no more than one record per work order number 
        number_day = work_orders[work_orders.completed_date==day]['work_order_number'].astype(str).tolist()
        number_text = ('Work Order<br>' if len(number_day)==1 else 'Work Orders<br>') + ('<br>').join(number_day)
        n = 20
        to_plot.append([None]*(2*n))
        text_hover.append([None]*(2*n))
        for j in range(n):
            to_plot[i][j::(2*n)] = [day]
            text_hover[i][j::(2*n)] = [number_text]
    to_plot = np.array(to_plot).flatten().tolist()
    text_hover = np.array(text_hover).flatten().tolist()
    fig.add_trace(go.Scatter(x=to_plot, 
                             y=np.linspace(m, M, n).tolist()*(len(work_orders.completed_date.unique())*2),
                             mode="lines", 
                             line_color=color_seq_alarms,
                             name='Completed Date', 
                             showlegend=True, 
                             #visible='legendonly',
                             legendrank=1,
                             legendgroup="work_orders", 
                             legendgrouptitle_text="Work Orders",
                             connectgaps=False, 
                             line_width=2,
                             line_dash='dash',
                             text=text_hover, 
                             hovertemplate="%{x|%d %b}<br><br>%{text}" ))
    
        
    if add_trend:
        fig, custom_trend_change_text = add_trend_to_plot(fig, df_smoothed_nr, trend_start, trend_end)
    else:
        custom_trend_change_text = None

    fig.update_layout(legend_groupclick="toggleitem")#legend_title_text='Alarms')
        
    return fig, custom_trend_change_text#, df_smoothed_nr['datetime'], df_smoothed_nr['value']

