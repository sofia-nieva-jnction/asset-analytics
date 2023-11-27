import plotly.express as px
import pandas as pd
import datetime
import sqlalchemy
import streamlit as st
import numpy as np
from utils import get_l1, get_l2, get_radar_data, get_fault_timeline, timeline_plot, radar_trace_plot

st.set_page_config(layout="wide", page_icon="", page_title="Asset Analysis")


st.title(f'Overview of Asset Classes​')

route = st.selectbox('Select Route',
                     ['LNW South', 'LNW North'],
                     key = 'route')

l1 = get_l1(route)
st.dataframe(l1, use_container_width=True)


st.title(f'Overview of Individual Assets in a Class​')

asset_class = st.selectbox('Select Class',
                     #classes_in_radar,
                     ['Signalling - TC - DC', 'S&C (Signalling) - Point Operating Equipment'], 
                     key = 'class')
l2 = get_l2(route, asset_class)
st.dataframe(l2, use_container_width=True)


st.title(f'Asset Details​')

asset_number = st.selectbox('Select Asset Number',
                             [563000, 884652, 7439162] if asset_class=='Signalling - TC - DC' else [866066, 765267],
                             key='assets')

st.subheader(f'Ellipse Details')
ellipse_details = pd.read_csv(f'ellipse_{asset_number}.csv')
st.dataframe(ellipse_details.dropna(axis=1), use_container_width=True)

st.markdown('West Coast South covers Euston, Bletchley and Stafford DUs')

st.subheader(f'Faults List (Jan-July 2020)')
faults_list = pd.read_csv(f'faults_{asset_number}.csv')

st.dataframe(faults_list.dropna(axis=1), use_container_width=True)


st.header(f'Fault Details')

fault_number = st.selectbox('Select Fault',
                             faults_list['fault_number'].tolist(),
                             key='faults')

st.subheader(f'Fault Timeline')

col1, space1, col2 = st.columns([0.35, 0.03, 0.65])

with col1:
    timeline = get_fault_timeline(asset_number, fault_number)
    st.dataframe(timeline, use_container_width=True)#, height=100*(len(timeline)//2-1))
    
with space1:
    st.write('')

with col2:
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    timeline_fig = timeline_plot(timeline, fault_number, asset_number)
    st.plotly_chart(timeline_fig, use_container_width=True)


st.subheader(f'RADAR traces')
if asset_number not in ['563000', '884652', '7439162', '866066', '765267']:
    st.text('No RADAR Traces for this Asset')
else:
    radar = get_radar_data(asset_number)
    class_attributes_traces = pd.read_csv(f'class_attributes_traces_{asset_class}.csv').values.flatten().tolist()
    class_attributes_other =  pd.read_csv(f'class_attributes_other_{asset_class}.csv').values.flatten().tolist()
    radar_summary = pd.read_csv(f'radar_summary_{asset_number}.csv')
    #radar['datetime'] = pd.to_datetime(radar['datetime'])
    radar_summary['date'] = pd.to_datetime(radar_summary['date'])

    st.markdown('Select days to analyse or compare:')

    col3, space2, col4, space3, col5 = st.columns([0.4, 0.03, 0.5, 0.03, 0.1])

    with col3:   
        default_day1 = datetime.datetime.strptime(min(timeline.Time), '%Y-%m-%d %H:%M:%S').date()
        day1 = st.date_input("Select Day", default_day1, key='day1')

        day1_attributes = radar[(radar.datetime > str(day1)) & (radar.datetime < str(day1 + datetime.timedelta(days=1)))]['attribute'].unique()

        attribute_trace1 = st.selectbox('Select Attribute to plot',
                                        [x for x in class_attributes_traces if x in day1_attributes],
                                        key = 'att1')
        attribute_other1 = st.selectbox('Select other Attribute to plot',
                                        [None] + [x for x in class_attributes_other if x in day1_attributes],
                                        key = 'other1')
        
    with space2:
        st.write('')    

    with col4:
        st.markdown(f"RADAR summary for {day1.strftime('%A')} {str(day1)}")
        radar_summary_day1 = radar_summary[radar_summary['date']==day1].reset_index(drop=True)
        st.dataframe(radar_summary_day1, use_container_width=True)

    with space3:
        st.write('') 

    with col5:
        st.markdown(f'Available Dates')
        st.dataframe(pd.to_datetime(radar['datetime']).apply(lambda x: x.date()).drop_duplicates().sort_values(),
                     hide_index=True, use_container_width=True, height=250)

    if attribute_trace1 in radar_summary_day1.attribute.tolist():      
        fig1 = radar_trace_plot(asset_class, radar, day1, timeline, attribute_trace1, attribute_other1)
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.text('No RADAR Traces for this Day')
        
    col6, space4, col7, space5, col8 = st.columns([0.4, 0.03, 0.5, 0.03, 0.1])

    with col6:
        default_day2 = datetime.datetime.strptime(timeline[timeline['Event']=='In Order'].Time.values[0], '%Y-%m-%d %H:%M:%S').date()
        default_day2 = default_day2 if default_day2!=default_day1 else default_day2 + datetime.timedelta(days=1)
        day2 = st.date_input("Select Day", default_day2, key='day2')
        
        day2_attributes = radar[(radar.datetime > str(day2)) & (radar.datetime < str(day2 + datetime.timedelta(days=1)))]['attribute'].unique()

        attribute_trace2 = st.selectbox('Select Attribute to plot',
                                        [x for x in class_attributes_traces if x in day2_attributes],
                                        key = 'att2')
        attribute_other2 = st.selectbox('Select other Attribute to plot',
                                        [None] + [x for x in class_attributes_other if x in day2_attributes],
                                        key = 'other2')
        
    with space4:
        st.write('')

    with col7:
        st.markdown(f"RADAR summary for {day2.strftime('%A')} {str(day2)}")
        radar_summary_day2 = radar_summary[radar_summary['date']==day2].reset_index(drop=True)
        st.dataframe(radar_summary_day2, use_container_width=True)

    with space5:
        st.write('') 

    with col8:
        st.markdown(f'Available Dates')
        st.dataframe(pd.to_datetime(radar['datetime']).apply(lambda x: x.date()).drop_duplicates().sort_values(), 
                     hide_index=True, use_container_width=True, height=250)

    if attribute_trace2 in radar_summary_day2.attribute.tolist():      
        fig2 = radar_trace_plot(asset_class, radar, day2, timeline, attribute_trace2, attribute_other2)     
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.text('No RADAR Traces for this Day')
