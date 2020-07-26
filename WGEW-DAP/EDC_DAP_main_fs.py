"""
    This file contains EDC main workflow and main query function

Two Modules:
    - EDC_DAP
    - query_rain_run_sm_sedi

Copyright 2020  Haiyan Wei
haiyan.wei@usda.gov

"""

import os
import re
import sys
import glob
import math
import logging
import datetime
import numpy as np
import pandas as pd
import logging
from query_fs import *
from modeling_fs import *
from dir_logging_fs import *
from plot_hydro_hyeto_fs import *
from plot_summary_table_fs import *

pd.options.display.float_format = '{:.2f}'.format
pd.options.display.max_rows = 99
pd.set_option('display.width', 10000)
t0 = datetime.datetime.now()


def EDC_DAP(watershed, flume, gauges, out, \
            runoffAggHr, rainAggHr, rainHr, \
            runoffTooLow, rainTooLow, runTooLateHr):

    goals='\
        1. Query runoff, rainfall, soil moisture, and sediment data from the DAP data server\n\
        2. Create .pre files\n\
        3. Plot hyetograph and hydrographs\n\
        4. Create a summary table for all events\n\n'

    goals += 'Definitions:\n\
        1. record: a measured rainfall (or runoff) value with a timestamp\n\
        2. segment: a set of rainfall (or runoff) records between two zero records\n\
        3. runoff-rainfall event: a series of rainfall and runoff records (or segments) that are appropriately aligned so that the selected runoff records form an independent runoff event that is responsive to the chosen rainfall records\n\n'

    gaugeStr = '_'.join(map(str, gauges))
    out = f'{out}_{flume}_Gauge{gaugeStr}_rainHr{rainHr}_RainAggHr{rainAggHr}_RunoffAggHr_{runoffAggHr}_runRespHr{runTooLateHr}'

    wspace = os.path.join(os.path.dirname(__file__), 'output', out)
    makefolders([wspace])
    os.chdir(wspace)
    logINFO('_.log', goals, t0)
    logInputs([wspace, watershed, flume, gauges, runoffTooLow, rainTooLow, \
               runoffAggHr, rainAggHr, rainHr, runTooLateHr])

    makefolders(['preFiles', 'hyeto_hydro_graphs', 'hyeto_hydro_graphs_bygroup'])

    logging.info(f'========= Query runoff segments from DAP =========\n\n')

    df_runSegments = query_DAP_runoff_segments(watershed, flume)

    # all segments include missing with a flag
    df_runSegments.loc[df_runSegments.runoffVolume_DAP.isnull(), 'missingRunDAPFlag'] = 1

    df_runEventsAgg = aggregate_runoff_events(df_runSegments, runoffAggHr)

    v(f'Counts in the segment table:\n {df_runSegments.count()}\n\n')
    v(f'Total number of runoff segments: {len(df_runSegments)}')
    v(f'  -number of analog segments: {sum(df_runSegments.dataType=="a")}')
    v(f'  -number of digital segments: {sum(df_runSegments.dataType=="d")}')
    v(f'  -{sum(df_runSegments.runoffVolume_DAP.isnull())} segments with missing runoff (blank)')
    v(f'  -{sum(df_runSegments.runoffVolume_DAP==0)} runoff segments with 0 runoff\n')
    v(f'"runoffAggHr" was applied to all runoff segments (including missing and 0)')
    v(f'Total Number of Runoff Events: {len(df_runEventsAgg)}')
    v(f'  -{len(df_runSegments) - len(df_runEventsAgg)} segments were aggreated')


    logging.info('\n\n========= Query rainfall, runoff, soil moisture and sediment =========')


    df_events, df_runoff, df_rain, df_sm, df_sedi, df_runRaw = query_rain_run_sm_sedi(df_runEventsAgg, watershed, flume, gauges, rainHr, rainAggHr, runoffAggHr)

    df_events = get_summary_table(df_events, df_rain, df_runoff, df_sm, df_sedi)

    df_events_flag = get_flags(df_events, runTooLateHr, rainTooLow, runoffTooLow)

    save_results(df_runSegments, df_runEventsAgg, df_events, df_rain,\
                 df_runoff, df_runRaw, df_events_flag, wspace)

    write_pre(df_events_flag, df_rain, df_runoff, df_sm, watershed)

    plot_event_summary(df_events)

    plot_hyeto_hydro_all_events(flume, df_events_flag, df_rain, df_runoff)

    group_hydrographs(df_events_flag)


    v(f'\n\n===== THE END =====\nTime to complete: {datetime.datetime.now()-t0}')



def query_rain_run_sm_sedi(df_events, watershed, flume, gauges, rainHr, rainAggHr, runoffAggHr):

    """ This module loops through all runoff events and:
        1. query runoff rates
        2. query rainfall based on gauge id and rainHr and rainAggHr
        3. query sediment based on runoff start time and end time
        4. query soil moisture

        df_rainRaw: all rainfall records available at selected gauges
                    when multiple gauges, df_rainRaw gets big, be aware.
        df_rain: rainfall records matched to runoff events
        df_runoff: runoff records matched to runoff events
        df_sm: soil moisture matched to runoff events (and gauges)
        df_sedi: total sediment in kg for each runoff event

        """

    df_rain, df_events = query_rainfall(df_events, watershed, gauges, rainHr, rainAggHr, runoffAggHr)

    df_rain = align_rainfall_timeStamps(df_rain)

    df_runoff, df_runRaw = query_runoff(df_events, watershed, flume, df_rain)

    df_runoff = align_runoff_timeStamps(df_events, df_runoff)

    df_sm = query_sm(flume, df_rain, df_events)

    df_sedi = query_sediment(df_events, watershed, flume)

    df_events, df_runoff, df_rain, df_sedi = smRainFlag_update(df_events, df_runoff, df_rain, df_sm, df_sedi)


    return df_events, df_runoff, df_rain, df_sm, df_sedi, df_runRaw

