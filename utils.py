import plotly.express as px
import pandas as pd
import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import sqlalchemy
import streamlit as st
import numpy as np


@st.cache_data
def get_l1(route):
    df = pd.read_csv('l1.csv')
    return df[df.route==route]

@st.cache_data
def get_l2(route, asset_class):
    df = pd.read_csv('l2.csv')
    df_filtered = df[df.group_class==asset_class]
    return df_filtered.drop(columns=['group_class'])


@st.cache_data
def get_radar_data(asset_number):
    df = pd.read_csv(f'radar_{asset_number}.csv')
    return df


def get_fault_timeline(asset_number, fault_number):
    timeline = pd.read_csv(f'timeline_{asset_number}_{fault_number}.csv')
    return timeline

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
def radar_trace_plot(asset_class, radar, day, timeline, attribute, other):
    asset = radar.iloc[0]['asset']
    radar_day = radar[(radar.datetime > str(day)) & (radar.datetime < str(day + datetime.timedelta(days=1)))].copy().sort_values('datetime')

    if asset_class == 'S&C (Signalling) - Point Operating Equipment':
        attribute2 = attribute[:-2] + attribute[-2:][::-1]
        attributes = [attribute, attribute2]
        attributes.sort()
        if other is not None and other.split('_')[-1] in ['Average', 'Peak']:
            attributes = attributes + [other]
        df = radar_day[radar_day.attribute.isin(attributes)]
        df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M:%S.%f')
        fig = px.scatter(df, x='datetime', y='value', labels={'value': f"{attribute.replace('_', ' ')} Value", 'datetime':''}, color='attribute',
                     template='plotly_white', color_discrete_map=dict(zip(attributes, ['blue', 'green', 'red'])))
        if other is not None and (other.split('_')[-1] == 'Length' or other=='Movement_Direction_Indicator'):
            subfig = make_subplots(specs=[[{"secondary_y": True}]])
            trace_name = f"{other.replace('_', ' ')}" if other=='Movement_Direction_Indicator' else f"{other.replace('_', ' ')} (seconds)"
            fig2 = px.scatter(radar_day[radar_day['attribute']==other], x='datetime', y='value', labels={'value': trace_name, 'datetime':''}, color_discrete_sequence=['red'])
            fig2.update_traces(yaxis="y2")
            subfig.add_traces(fig.data + fig2.data)
            subfig.layout.yaxis2.title = trace_name
            fig = subfig
        
    else:
        df = radar_day[radar_day.attribute.isin([attribute])]
        df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M:%S.%f')
        fig = px.scatter(df, x='datetime', y='value', labels={'value': f"{attribute.replace('_', ' ')} Value", 'datetime':''}, 
                     template='plotly_white')

    fig.update_xaxes(
        type='date',
        tick0=datetime.datetime.strptime('2020/01/01 00:00:00.000', '%Y/%m/%d %H:%M:%S.%f'),
    )

    fig.update_traces(marker_size=3,
                      mode='lines+markers',
                      line_width=1,
                     )

    day_timeline = timeline[(timeline['Time'] > str(day)) & (timeline['Time'] < str(day + datetime.timedelta(days=1)))]
    for i, row in day_timeline.iterrows():
        fig.add_vline(x=pd.to_datetime(row['Time']))

        fig.add_annotation(x=pd.to_datetime(row['Time']), xanchor='left', 
                        y=1-i*0.1, yref="paper", font_size=18, font_color='black',
                        text=f"<b>{row['Event']}</b>", showarrow=False)
        
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
                            y=1-(i+1)*0.1, yref="paper", font_size=18,
                            text=f"<b>{other.replace('_', ' ')}</b>", showarrow=False)

    return fig
