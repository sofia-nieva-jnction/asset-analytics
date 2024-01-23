import plotly.express as px
import pandas as pd
import datetime
import sqlalchemy
import streamlit as st
import numpy as np
from utils import get_l0, get_l1, get_worst_perfoming_table, plot_vertical_histograms, get_l2, highlight_threshold, get_radar_data, get_fault_timeline, get_all_faults_timeline, timeline_plot, get_work_orders, get_radar_summary, get_list_assets
from utils import radar_trace_plot, headcodes_plot, get_data_example, get_berth_steps, plot_max_smoothed_and_count_tc, plot_max_smoothed_and_count_points, alarms_near_failures, get_trends_table, get_ellipse_details, get_faults_list

st.set_page_config(layout="wide", page_icon="", page_title="Asset Analysis")

tab1, tab2, = st.tabs(["Ellipse-FMS-RADAR merge", "Berth Data Merge Examples"])

with tab1:

    route = st.selectbox('Select Route',
                        ['LNW South'],
                        key = 'route')
    relevant_classes = ['Signalling - TC - DC', 'S&C (Signalling) - Point Operating Equipment'] #'Signalling - TC - High Freq',
    
    asset_class = st.selectbox('Select Group - Class',
                                relevant_classes, 
                                key = 'class')

    tab11, tab12, tab13 = st.tabs(["Route Overview", "Worst Performing", "Individual Assets"])

    with tab11:

        st.header(f'Failures and RADAR data by Asset Class Group​')
        col1, col2 = st.columns([0.5, 0.47])
        with col1:
            l0 = get_l0(route)
            st.dataframe(l0, column_config={'route': 'Route', 
                                            'ellipse_asset_class_group_desc': 'Class Group (Ellipse)',
                                            'ellipse_asset_count': 'Asset Count (Ellipse)',
                                            'fms_failures_count_6m_2020': 'Total Failures (FMS)',
                                            'count_service_affecting_faults_6m_2020': 'Service Affecting Failures (FMS)', 
                                            'has_radar_percentage': '% in RADAR'},
                        hide_index=True, use_container_width=True)
        with col2:
            fig = plot_vertical_histograms(l0, y_col='ellipse_asset_class_group_desc', y_name='Asset Class Group')
            st.plotly_chart(fig, use_container_width=True)


        st.header(f'Breakdown by Asset Class​')
        col1, col2 = st.columns([0.5, 0.47])
        with col1:
            l1 = get_l1(route).drop(columns=['route'])
            st.dataframe(l1, column_config={'ellipse_asset_class_group_desc': 'Class Group (Ellipse)',
                                            'ellipse_asset_class_desc': 'Class (Ellipse)',
                                            'ellipse_asset_count': 'Asset Count (Ellipse)',
                                            'fms_failures_count_6m_2020': 'Total Failures (FMS)',
                                            'count_service_affecting_faults_6m_2020': 'Service Affecting (FMS)', 
                                            'has_radar_percentage': '% in RADAR'},
                        hide_index=True, use_container_width=True)
        with col2:
            fig = plot_vertical_histograms(l1[l1.has_radar_percentage>0], y_col='ellipse_asset_class_desc', y_name='Asset Class')
            st.plotly_chart(fig, use_container_width=True)

    with tab12:
        st.header('Failures, Work Orders and Alarms (for the period Jan-July 2020)')
        st.write('Default order by number of Service Affecting Failures. Click on the name of another column to reorder based on that column.')
        worst_perfoming_table = get_worst_perfoming_table(route, asset_class)
        st.dataframe(worst_perfoming_table,
                    column_config={'route': 'Route', 
                                'ellipse_asset_class_group_desc': 'Class Group',
                                'ellipse_asset_class_desc': 'Class',
                                'ellipse_asset_number': 'Asset Number',
                                'fms_failures_count_6m_2020': 'Total Failures',
                                'count_service_affecting_faults_6m_2020': 'Service Affecting Failures', 
                                'is_in_radar': 'In RADAR',
                                'total_work_orders': 'Total Work Orders', 
                                'total alarms': 'Total Alarms'},
                    use_container_width=True)

        # st.header('Latest Changes in Trend (valid as of 2020/07/06)')
                
        # c1, c2, c3 = st.columns([0.06, 0.2, 0.2])
        # with c1:
        #     st.write('')
        #     st.write('')
        #     st.write('Choose values to highlight')
        # with c2:
        #     condition = st.selectbox('Select Condition', ['Greater than', 'Less than', 'Equal to', 'Greater than or Equal to', 'Less than or Equal to'])
        # with c3:
        #     threshold = st.number_input("Input percentage", step=0.1)

        # if asset_class == 'S&C (Signalling) - Point Operating Equipment':
        #     attribute_line_chart = st.selectbox('Select Attribute to plot', 
        #                                             ['Current_Waveform_NR_Average', 'Current_Waveform_RN_Average',
        #                                              'Current_Waveform_NR_Peak', 'Current_Waveform_RN_Peak',
        #                                              'Current_Waveform_NR_Length', 'Current_Waveform_RN_Length'])
        # else:
        #     attribute_line_chart = 'Circuit_Current'

        # asset_number_search = st.text_input('Search Asset', placeholder = 'Input Asset Number')        

        # trends_df = get_trends_table(route, asset_class, attribute_line_chart)
        # trends_df['highlight_15d'] = trends_df['change_15d'] > threshold
        # trends_df['highlight_1m'] = trends_df['change_1m'] > threshold
        # trends_df['highlight_2m'] = trends_df['change_2m'] > threshold
        # trends_df['highlight_3m'] = trends_df['change_3m'] > threshold
        # trends_df.sort_values(['highlight_15d', 'highlight_1m', 'highlight_2m', 'highlight_3m'], ascending=[False, False, False, False], inplace=True)
        # trends_df = trends_df[['asset_number', 'days_since_last_fault', 'change_15d', 'change_1m', 'change_2m', 'change_3m'] + [attribute_line_chart]]
        # trends_df = trends_df.style.applymap(highlight_threshold, condition=condition, threshold=threshold, subset=['change_15d', 'change_1m', 'change_2m', 'change_3m'])

        # trends_df = trends_df[trends_df.asset_number==asset_number_search] if asset_number_search is not None else trends_df

        # if len(trends_df) > 0:
        #     st.dataframe(
        #         trends_df,
        #         column_config={
        #             'asset_number': 'Asset Number',
        #             'days_since_last_fault': 'Days since last Fault', 
        #             'change_15d': st.column_config.NumberColumn('Change last 15 days', format = "%.2f %%"), 
        #             'change_1m': st.column_config.NumberColumn('Change last 30 days', format = "%.2f %%"), 
        #             'change_2m': st.column_config.NumberColumn('Change last 60 days', format = "%.2f %%"), 
        #             'change_3m': st.column_config.NumberColumn('Change last 90 days', format = "%.2f %%"), 
        #             "Current_Waveform_NR_Average": st.column_config.LineChartColumn(
        #                 "Average Current Waveform NR (past 6 months)", width='Large'),
        #             "Current_Waveform_RN_Average": st.column_config.LineChartColumn(
        #                 "Average Current Waveform RN (past 6 months)", width='Large'),
        #             "Current_Waveform_NR_Peak": st.column_config.LineChartColumn(
        #                 "Peak Current Waveform NR (past 6 months)", width='Large'),
        #             "Current_Waveform_RN_Peak": st.column_config.LineChartColumn(
        #                 "Peak Current Waveform RN (past 6 months)", width='Large'),
        #             "Current_Waveform_NR_Length": st.column_config.LineChartColumn(
        #                 "Length Current Waveform NR (past 6 months)", width='Large'),
        #             "Current_Waveform_RN_Length": st.column_config.LineChartColumn(
        #                 "Length Current Waveform RN (past 6 months)", width='Large'),
        #             "Circuit_Current": st.column_config.LineChartColumn(
        #                 "Circuit Current (past 6 months)", width='Large'),
        #         },
        #         use_container_width=True, hide_index=True)
        # else:
        #     st.write('Asset Not Found')


    with tab13:
        l2 = get_l2(route, asset_class)

        asset_number = st.selectbox('Select Asset Number',
                                    get_list_assets(asset_class),
                                    key='assets')
        
        ellipse_details = get_ellipse_details(asset_number)
        faults_list = get_faults_list(asset_number)

        with st.expander("Asset Details", expanded=True):
            st.subheader(f'Name and Type')
            st.dataframe(pd.concat([ellipse_details[['asset_number', 'colloquial_name_1', 'colloquial_name_2', 'colloquial_name_3', 'colloquial_name_4', 'colloquial_name_5', 'colloquial_name_6', 'colloquial_name_7', 'colloquial_name_8', 'egi_desc', 'egi_code', 'asset_status_code_desc']].dropna(axis=1), 
                                    faults_list[['delivery_unit', 'asset_description']].dropna(axis=1).drop_duplicates()], axis=1),
                         use_container_width=True, hide_index=True)
            st.subheader(f'Location')
            fms_ellipse = pd.concat([ellipse_details[['route', 'start_miles', 'start_yards', 'end_miles', 'end_yards']].dropna(axis=1), 
                                     faults_list[['delivery_unit', 'elr_description', 'place_name']].dropna(axis=1).drop_duplicates(),
                                     ellipse_details[['asset_position_code_desc', 'strategic_route_code_desc', 'asset_owner', 'engineer', 'section_manager']].dropna(axis=1)], axis=1)
            st.dataframe(fms_ellipse, use_container_width=True, hide_index=True)

        # st.markdown('West Coast South covers Euston, Bletchley and Stafford DUs')

        with st.expander("Faults Details", expanded=True):
            st.subheader(f'Faults List (Jan-July 2020)')
            st.dataframe(faults_list[['occurred_date', 'fault_number', 'delivery_unit', 'trust_incident_numbers', 'is_service_affecting', 'component_level_1_2', 'priority', 'symptom', 'risk', 'sincs_status']].dropna(axis=1), use_container_width=True)
            st.subheader(f'Alarms')
            st.write('Showing the number of alarms in the 14 days preceding each failure')
            radar = get_radar_data(asset_number)
            work_orders_asset = get_work_orders(asset_number)
            if asset_class == 'Signalling - TC - DC':
                st.dataframe(alarms_near_failures(faults_list, radar, work_orders_asset, d=14), use_container_width=True)
            else:
                st.dataframe(alarms_near_failures(faults_list, radar, work_orders_asset, d=14)[['Date', 'Fault Number', 'Days since last Fault', 'Days since last Work Order']], use_container_width=True)


        with st.expander("Work Orders Details", expanded=True):
            st.subheader(f'Work Orders List (Jan-July 2020)')
            try:
                work_orders_table = work_orders_asset.dropna(axis=1, how='all').drop(columns=['route', 'asset_class', 'asset_number']).fillna('-')
                column_config={'extended_text': st.column_config.TextColumn('extended_text', width='medium', disabled=True),
                            'defect_desc': st.column_config.TextColumn('defect_desc', width='medium', disabled=True),
                            'completion_comments': st.column_config.TextColumn('completion_comments', width='medium', disabled=True),
                            #'work_order_description': st.column_config.TextColumn('work_order_description', width='large', disabled=True)
                            }
                st.dataframe(work_orders_table, use_container_width=True, column_config=column_config)
            except:
                st.write('No Work Orders for this Asset')

        with st.expander("RADAR Data", expanded=True):
            st.subheader(f'Long Term Aggregation of RADAR Traces')

            c1, s1, c2, s2, c3 = st.columns([0.3, 0.04, 0.3, 0.04, 0.5])

            with c1:
                option = st.selectbox('Plot Custom Trend Line',
                                      ('No', 'Yes'))
            with s1:
                st.write('')
            with c2:
                if option == 'Yes':
                    trend_start, trend_end = st.date_input("Select days to calculate Trend",
                                                            (datetime.date(year=2020, month=7, day=6), datetime.date(year=2020, month=7, day=6)),
                                                            key='trend',
                                                            max_value=datetime.date(year=2020, month=7, day=6),
                                                            min_value=datetime.date(year=2020, month=1, day=1))
            with s2:
                st.write('')

            class_attributes_traces = pd.read_csv(f'class_attributes_traces_{asset_class}.csv')
            class_attributes_other = pd.read_csv(f'class_attributes_other_{asset_class}.csv')
            work_orders_asset = get_work_orders(asset_number)
            if asset_class=='Signalling - TC - DC':
                attribute = 'Circuit_Current'
                count_attribute = 'Total_Occupations_Count'
                if not attribute in radar.attribute.unique():
                    st.text('Not enough RADAR Data for this Asset')
                else:
                    if option == 'Yes' and trend_start!=datetime.date(year=2020, month=7, day=6) and trend_end!=datetime.date(year=2020, month=7, day=6):
                        fig, custom_trend_change_text = plot_max_smoothed_and_count_tc('2020-01', '2020-07', attribute, count_attribute, radar, faults_list, work_orders_asset, class_attributes_other, add_trend=True, trend_start=trend_start, trend_end=trend_end)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig, _ = plot_max_smoothed_and_count_tc('2020-01', '2020-07', attribute, count_attribute, radar, faults_list, work_orders_asset, class_attributes_other)
                        st.plotly_chart(fig, use_container_width=True)
            elif asset_class == 'S&C (Signalling) - Point Operating Equipment':
                attribute = st.selectbox('Select Attribute to plot',
                                ['Current_Waveform_NR_Average', 'Current_Waveform_RN_Average',
                                 'Current_Waveform_NR_Peak', 'Current_Waveform_RN_Peak',
                                 'Current_Waveform_NR_Length', 'Current_Waveform_RN_Length'],
                                key='att_long_term_poe')
                count_attribute = 'Total_Operations_NR' if 'NR' in attribute else 'Total_Operations_RN'
                if not attribute in radar.attribute.unique():
                    st.text('Not enough RADAR Data for this Asset')
                else:
                    if option == 'Yes' and trend_start!=datetime.date(year=2020, month=7, day=6) and trend_end!=datetime.date(year=2020, month=7, day=6):
                        fig, custom_trend_change_text = plot_max_smoothed_and_count_points('2020-01', '2020-07', attribute, count_attribute, radar, faults_list, work_orders_asset, class_attributes_other, add_trend=True, trend_start=trend_start, trend_end=trend_end)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig, _ = plot_max_smoothed_and_count_points('2020-01', '2020-07', attribute, count_attribute, radar, faults_list, work_orders_asset, class_attributes_other)
                        st.plotly_chart(fig, use_container_width=True)

            with c3:
                if option == 'Yes':
                    st.write('')
                    st.write('')
                    day_diff = (trend_end-trend_start).days
                    st.write(f'The change between {trend_start} and {trend_end} ({day_diff} days) was: {custom_trend_change_text}')
            

            st.subheader(f'Timeline of Failures')

            col1, space1, col2 = st.columns([0.65, 0.03, 0.55])
            
            with col1:
                st.write('')
                st.write('')
                faults_number = faults_list.fault_number.unique().tolist()
                all_faults_timeline = get_all_faults_timeline(asset_number, faults_number)
                fault_number = st.selectbox('Select Fault',
                                        faults_list['fault_number'].tolist(),
                                        key='faults')
                timeline = get_fault_timeline(asset_number, fault_number)
                timeline_fig = timeline_plot(timeline, fault_number, asset_number)
                st.plotly_chart(timeline_fig, use_container_width=True)
                
            with space1:
                st.write('')

            with col2:
                st.dataframe(timeline, use_container_width=True)#, height=100*(len(timeline)//2-1))


            st.subheader(f'Detail of RADAR Traces')
            radar_summary = get_radar_summary(asset_number)

            st.markdown('Select days to analyse or compare:')

            col3, space2, col4, space3, col5 = st.columns([0.4, 0.03, 0.5, 0.03, 0.1])

            with col3:   
                default_day1 = datetime.datetime.strptime(min(timeline.Time), '%Y-%m-%d %H:%M:%S').date()
                day1_start, day1_end = st.date_input("Select days",
                                                    (default_day1 - datetime.timedelta(days=1), default_day1 + datetime.timedelta(days=1)),
                                                    key='day1')

                day1_attributes = radar[(radar.datetime >= str(day1_start)) & (radar.datetime <= str(pd.to_datetime(day1_end) + datetime.timedelta(days=1)))]['attribute'].unique()
                st.write(day1_attributes)
                st.write(class_attributes_traces.values)
                
                st.write(class_attributes_traces.values.tolist())
                st.write([x for x in class_attributes_traces if x in day1_attributes])
                attribute_trace1 = st.selectbox('Select Attribute to plot',
                                                [x for x in class_attributes_traces.values if x in day1_attributes],
                                                key = 'att1')
                attribute_other1 = st.selectbox('Select other Attribute to plot',
                                                [None] + [x for x in class_attributes_other.values if x in day1_attributes],
                                                key = 'other1')
                
                day2_start = st.date_input("Select day to compare", None, key='day2', max_value=datetime.date(year=2020, month=7, day=6))
                day2_end = day2_start
                
            with space2:
                st.write('')    

            with col4:
                st.markdown(f"RADAR summary")# for {day1.strftime('%A')} {str(day1)}")
                radar_summary_day1 = radar_summary[(radar_summary['date']>=day1_start) 
                                                & (radar_summary['date']<=day1_end)
                                                & (radar_summary['attribute']!=attribute_trace1)].sort_values(['date', 'attribute'])
                radar_summary_day1['day'] = radar_summary_day1['date'].apply(lambda x: x.strftime('%A'))
                radar_summary_day1 = radar_summary_day1[['asset', 'attribute', 'day', 'date', 'records_count']]
                st.dataframe(radar_summary_day1, use_container_width=True, hide_index=True, height=300)

            with space3:
                st.write('') 

            with col5:
                st.markdown(f'Available Dates')
                st.dataframe(radar['datetime'].apply(lambda x: x.date()).drop_duplicates().sort_values(),
                            hide_index=True, use_container_width=True, height=300)

            if attribute_trace1 in radar_summary[(radar_summary['date']>=day1_start) & (radar_summary['date']<=day1_end)].attribute.tolist():      
                fig1 = radar_trace_plot(asset_class, radar, day1_start, day1_end, all_faults_timeline, attribute_trace1, attribute_other1)
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.text('No RADAR Traces for this Day')
                
            if day2_start is not None:
                col6, space4, col7, space5, col8 = st.columns([0.4, 0.03, 0.5, 0.03, 0.1])

                with col6:
                    day2_attributes = radar[(radar.datetime >= str(day2_start)) & (radar.datetime <= str(pd.to_datetime(day2_end) + datetime.timedelta(days=1)))]['attribute'].unique()

                    attribute_trace2 = st.selectbox('Select Attribute to plot',
                                                    [x for x in class_attributes_traces.values if x in day2_attributes],
                                                    key = 'att2')
                    attribute_other2 = st.selectbox('Select other Attribute to plot',
                                                    [None] + [x for x in class_attributes_other.values if x in day2_attributes],
                                                    key = 'other2')
                    
                with space4:
                    st.write('')

                with col7:
                    st.markdown(f"RADAR summary")# for {day2.strftime('%A')} {str(day2)}")
                    radar_summary_day2 = radar_summary[(radar_summary['date']>=day2_start) 
                                                    & (radar_summary['date']<=day2_end)
                                                    & (radar_summary['attribute']!=attribute_trace2)].sort_values(['date', 'attribute'])
                    radar_summary_day2['day'] = radar_summary_day2['date'].apply(lambda x: x.strftime('%A'))
                    radar_summary_day2 = radar_summary_day2[['asset', 'attribute', 'day', 'date', 'records_count']]
                    st.dataframe(radar_summary_day2, use_container_width=True, hide_index=True)

                with space5:
                    st.write('') 

                with col8:
                    st.write('') 

                if attribute_trace2 in radar_summary[(radar_summary['date']>=day2_start) & (radar_summary['date']<=day2_end)].attribute.tolist():      
                    fig2 = radar_trace_plot(asset_class, radar, day2_start, day2_end, all_faults_timeline, attribute_trace2, attribute_other2)     
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.text('No RADAR Traces for this Day')


with tab2:
    route = 'LNW South'
    # asset_class = 'Signalling - TC - DC'
    attribute = 'Circuit_Current'
    tab21, tab22, tab23 = st.tabs(["Example 1", "Example 2", "Example 3"])
   
    with tab21:
        asset_number = 563000
        fault_number = 891199
        radar, day, timeline, ellipse_details, faults_list = get_data_example(asset_number, fault_number)
        berth_steps = get_berth_steps(asset_number, fault_number)
        berth = '0054'
        td = 'KN'

        other = st.selectbox('Select other Attribute to plot',
                                        [None] + ['Total_Occupations_Count'],
                                    key = '1other')
        fig = headcodes_plot(radar, day, timeline, attribute, other, berth_steps, berth, td)
        st.plotly_chart(fig, use_container_width=True)


        with st.expander("Asset, Berth and Fault Details", expanded=False):

            st.dataframe(ellipse_details.dropna(axis=1), use_container_width=True)
            st.dataframe(faults_list[faults_list.fault_number==fault_number].dropna(axis=1), use_container_width=True)
            
            col1, col2 = st.columns([1,2])
            with col1:
                st.dataframe(timeline, use_container_width=True)
            with col2:
                st.image(f'{berth}.png', caption='Image courtesy of opentraintimes.com')

    with tab22:
        asset_number = 840343
        fault_number = 890321
        radar, day, timeline, ellipse_details, faults_list = get_data_example(asset_number, fault_number)
        berth_steps = get_berth_steps(asset_number, fault_number)
        berth = '5351'
        td = 'R2'

        other = st.selectbox('Select other Attribute to plot',
                                        [None] + ['Total_Occupations_Count'],
                                    key = '2other')
        fig = headcodes_plot(radar, day, timeline, attribute, other, berth_steps, berth, td)
        st.plotly_chart(fig, use_container_width=True)


        with st.expander("Asset, Berth and Fault Details", expanded=False):

            st.dataframe(ellipse_details.dropna(axis=1), use_container_width=True)
            st.dataframe(faults_list[faults_list.fault_number==fault_number].dropna(axis=1), use_container_width=True)
            
            col1, col2 = st.columns([1,2])
            with col1:
                st.dataframe(timeline, use_container_width=True)
            with col2:
                st.image(f'{berth}.png', caption='Image courtesy of opentraintimes.com')

    with tab23:
        asset_number = 840569
        fault_number = 893459
        radar, day, timeline, ellipse_details, faults_list = get_data_example(asset_number, fault_number)
        berth_steps = get_berth_steps(asset_number, fault_number)
        berth = '1031'
        td = 'R2'

        other = st.selectbox('Select other Attribute to plot',        
                                    [None] + ['Total_Occupations_Count'],
                                    key = '3other')
        fig = headcodes_plot(radar, day, timeline, attribute, other, berth_steps, berth, td)
        st.plotly_chart(fig, use_container_width=True)


        with st.expander("Asset, Berth and Fault Details", expanded=False):

            st.dataframe(ellipse_details.dropna(axis=1), use_container_width=True)
            st.dataframe(faults_list[faults_list.fault_number==fault_number].dropna(axis=1), use_container_width=True)
            
            col1, col2 = st.columns([1,2])
            with col1:
                st.dataframe(timeline, use_container_width=True)
            with col2:
                st.image(f'{berth}.png', caption='Image courtesy of opentraintimes.com')
