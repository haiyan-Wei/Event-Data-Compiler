import os
import datetime
import numpy as np
import pandas as pd
from logging import info as v
from glob import glob1
from query_DAP_server_fs import *
from dir_logging_fs import reduce_df_size


def aggregate_runoff_events(df, hr):

    """merge two runoff events if interval less than assigned threshold:
       a flag of 2 indicates aggregated new events (thus, endtime remains the same and startTime=previous)
       a flag of 1 indicates original events, no need for aggregation
       a flag of 0 indicates the events that were aggreated and should be removed after the aggregation"""

    df = df.assign(intervalHr=(df.startTime-df.endTime.shift(1))/np.timedelta64(1, 'h'))

    if df.intervalHr.min() > hr:
        df_agg=df.assign(intervalNewHr=df.intervalHr)

    else:
        df = df.assign(agg_flag=1)

        for i in range(len(df)-1):

            if (df.loc[i+1, 'dataType']==df.loc[i, 'dataType']) & (df.loc[i+1, 'intervalHr'] < hr):

                df.loc[i+1, 'duration_DAP'] = np.sum(df.loc[i, 'duration_DAP'])
                df.loc[i+1, 'runoffVolume_DAP'] = np.sum(df.loc[i:i+1, 'runoffVolume_DAP'])
                df.loc[i+1, 'runoffPeak_DAP'] = max(df.loc[i:i+1, 'runoffPeak_DAP'])
                df.loc[i+1, 'startTime'] = df.loc[i, 'startTime']
                df.loc[i+1, 'preName'] = df.loc[i, 'preName']
                df.loc[i+1, 'agg_flag'] = 2
                df.loc[i, 'agg_flag'] = 0
                df.loc[i+1, 'missingRunDAPFlag'] = np.sum(df.loc[i:i+1, 'missingRunDAPFlag'])


        df_agg = df.loc[df.agg_flag>=1, :]
        df_agg = df_agg.reset_index(drop=True)

        intervalNewHr = [float('NaN')]
        for i in np.arange(1, len(df_agg)):
            intervalNewHr.append((df_agg.loc[i,'startTime']-df_agg.loc[i-1,'endTime'])/np.timedelta64(1, 'h'))

        df_agg = df_agg.assign(intervalNewHr=intervalNewHr)
        df_agg = df_agg.drop(columns=['intervalHr', 'agg_flag'])

    return df_agg


def query_rainfall(df_events, watershed, gauges, rainHr, rainAggHr, runoffAggHr):

    """
    input:
        df_events: aggregated runoff event list with startTime and endTime

    local:
        df_rainRaw_: all available rainfall records at selected gauge

    returns:
        df_rainRaw: all available rainfall records at all gauges
        df_rain: collection of selected rainfall records for all runoff events
        df_events: updated event table with rainfall information

    """
    df_events = df_events.reset_index(drop=True)
    df_rainRaw, df_rain = pd.DataFrame(), pd.DataFrame()

    for gauge in gauges:

        df_rainRaw_ = query_DAP_rain_rates(watershed, gauge)

        df_rainRaw_ = reduce_df_size(df_rainRaw_)

        v(f'\n\nat gauge{gauge}:\n{len(df_rainRaw_)} rainfall records are available from DAP. Ailability: {df_rainRaw_.timeStamp.min()} to {df_rainRaw_.timeStamp.max()}\n')

        for i in range(len(df_events)):

            preName, ad_type = df_events.preName[i],df_events.dataType[i]
            runStartTime, runEndTime = df_events.startTime[i], df_events.endTime[i]

            print(f'retrieve rainfall rates at gauge {gauge} for event {preName}')

            startTimeRainHr = runStartTime - pd.DateOffset(hours=rainHr)
            df_ = df_rainRaw_.loc[(df_rainRaw_.dataType==ad_type) & \
                                  (df_rainRaw_.rainStartTime >= startTimeRainHr ) & \
                                  (df_rainRaw_.rainStartTime <= runEndTime), :]

            if len(df_) == 0:
                v(f'no rainfall rates for runoff event {preName}')

            else:
                # calculate interval
                df_ = df_.assign(preName=preName,
                                 intervalHr= df_.timeStamp.diff(1).astype('timedelta64[s]')/3600.)
                df_=df_.reset_index(drop=True)

                # if more than one segment and there is interval > rainAggHr
                # EDC will not break a segment
                if ((sum(df_.intervalHr>rainAggHr)>0) & (len(df_.rainStartTime.unique())>1)):

                    for rainStart in df_.rainStartTime.unique()[1:]:

                        idx = df_[df_.rainStartTime==rainStart].index.min()

                        if (df_.intervalHr[idx] > rainAggHr) & \
                           (df_.timeStamp[idx-1] < (runStartTime - pd.DateOffset(hours=runoffAggHr))):
                            # (df_.timeStamp[idx-1] < runStartTime):
                                df_ = df_.loc[idx:, :]

                df_ = df_.reset_index(drop=True)
                df_rain = df_rain.append(df_)

        df_rain=df_rain.reset_index(drop=True)
        df_rainRaw = df_rainRaw.append(df_rainRaw_).reset_index(drop=True)

        df_rainRaw.to_csv('df_rainRaw.csv')


    # update df_events, calculate rainStartTime and rainEndTime

    for i in range(len(df_events)):
        preName = df_events.preName[i]
        df_ = df_rain.loc[df_rain.preName==preName, :]
        df_events.loc[i, 'rainStartTime'] = df_.timeStamp.min()
        df_events.loc[i, 'rainEndTime']= df_.timeStamp.max()

    return df_rain, df_events


def query_runoff(df_events, watershed, flume, df_rain):

    """
    This module
        - finds corresponding runoff for each event in table df_events

    Input:
        - df_events: for events starting and ending time
        - watershed: for query runoff records
        - flume: for query runoff records

    local:
        df_runRaw: all available runoff records from DAP

    return:
        df_runoff: runoff rates for each event

    """

    # query all runoff records at selected flume
    df_runRaw = query_DAP_runoff_rates(watershed, flume)
    df_runRaw.runCode = df_runRaw.runCode.astype('int32')
    df_runRaw = reduce_df_size(df_runRaw)

    v(f'\n\nat flume{flume}:\n{len(df_runRaw)} runoff records are available from DAP. Availability: {df_runRaw.timeStamp.min()} to {df_runRaw.timeStamp.max()}\n')

    df_runoff = pd.DataFrame()

    for i in range(len(df_events)):

        preName, ad_type = df_events.preName[i],df_events.dataType[i]
        runStartTime, runEndTime = df_events.startTime[i], df_events.endTime[i]

        print(f'retrieve runoff rates for event {preName}')

        df_ = df_runRaw.loc[(df_runRaw.dataType==ad_type) &
                            (df_runRaw.startTime>=runStartTime) & \
                            (df_runRaw.startTime<=runEndTime), :]

        if len(df_) == 0:
            v(f'no runoff rates for runoff event {preName}')

        else:
            df_ = df_.assign(preName=preName)
            df_runoff = df_runoff.append(df_)

    df_runoff = df_runoff.reset_index(drop=True)

    return df_runoff, df_runRaw


def query_sm(flume, df_rain, df_events):

    print(f'reading soil moisture data ...')

    df_sm_raw = pd.read_csv(os.path.join(os.path.dirname(__file__),\
                           'Lucky_hills_input_data', \
                            'sm_lucky_hills.csv'),
                            parse_dates=['DateTime'])

    df_sm_raw = reduce_df_size(df_sm_raw)

    v(f'\n\nSoil moisture available between {df_sm_raw.DateTime.min()} and {df_sm_raw.DateTime.max()}')

    sm5 = []
    for preName in df_events.preName:

        print(f'retrieve soil moisture for event {preName}')

        s = df_rain.loc[df_rain.preName==preName, 'rainStartTime'].min()
        # get the earlist rainStartTime for all gauges
        # if soil moisture data exists for different gauges, then add selection for gauge

        if s >= df_sm_raw.DateTime.min():

            closest_index = df_sm_raw.loc[df_sm_raw.DateTime <= s].index[-1]
            smTime = df_sm_raw.loc[closest_index, 'DateTime']

            if (s-smTime).total_seconds()/3600. > 24:
                sm5.append(np.nan)
            else:
                sm5.append(df_sm_raw.loc[closest_index, 'SM5'])

        else:
            sm5.append(np.nan)

    df_sm = pd.DataFrame().assign(preName=df_events.preName, sm5=sm5)

    return df_sm


def query_sediment(df_events, watershed, flume):

    df_sediRaw = query_DAP_sedi_mass(watershed, flume)

    totalSediKg = []
    for i in range(len(df_events)):

        print(f'retrieve sediment for event {df_events.preName[i]}')
        runStartTime, runEndTime = df_events.startTime[i], df_events.endTime[i]

        df_ = df_sediRaw.loc[(df_sediRaw.RunoffEventDateTime >= runStartTime) & \
                             (df_sediRaw.RunoffEventDateTime <= runEndTime), :]

        if len(df_) == 0:
            totalSediKg.append(np.nan)
        else:
            totalSediKg.append(df_.sum().SedimentYield * 0.453592)   # from lb to kg

    df_sedi = pd.DataFrame().assign(preName=df_events.preName, sediKg_DAP=totalSediKg)

    return df_sedi


def align_runoff_timeStamps(df_events, df_runoff):

    """
        - align runoff elapsedTime based on firstRainStart
        - calculate runoff volume/ rate and elapsedTime
    """

    df = pd.DataFrame()

    for preName in df_events.preName.unique():

        print(f'align runoff time stamps for event {preName}')

        df_ = df_runoff[df_runoff.preName==preName].reset_index(drop=True)

        if len(df_) > 1:

            firstRainStart = df_events[df_events.preName==preName].rainStartTime.values[0]

            if pd.isnull(firstRainStart):
                df_=df_.assign(elapsedTime=(df_.timeStamp-df_.timeStamp.min()).astype('timedelta64[s]')/60.)
            else:
                df_=df_.assign(elapsedTime=(df_.timeStamp-firstRainStart).astype('timedelta64[s]')/60.)

            df_=df_.assign(intervalHr=(df_.timeStamp.diff(1)).astype('timedelta64[s]')/3600.)
            df_.loc[0, 'intervalHr'] = 0
            df_=df_.assign(accDepth=(df_.runoffRate_DAP*df_.intervalHr).cumsum())

            df = df.append(df_)

    newColOrder = ['flume', 'preName', 'dataType', 'startTime', 'timeStamp', \
                   'elapsedTime_DAP', 'runoffRate_DAP', 'runoffDepth_DAP', 'accDepth',
                   'runCode', 'elapsedTime', 'intervalHr']

    df = df[newColOrder].reset_index(drop=True)

    return df


def align_rainfall_timeStamps(df0):

    """
    calculate elapsedTime based on the first rainStartTime
    calculate rainRate based on accDepth_DAP
        - treat accDepth_DAP as the original measurements from rain gauge (not rate)
        - calculate rainRate and they might be different from the rate from DAP database

    """

    df = pd.DataFrame()

    for preName in df0.preName.unique():

        print(f'align rainfall time stamps for event {preName}')

        df_pre = pd.DataFrame()

        for gauge in df0.gauge.unique():

            df_ = df0[(df0.preName==preName) & (df0.gauge==gauge)]

            if len(df_.rainStartTime.unique()) > 1:

                df_ = df_.sort_values(by=['rainStartTime', 'timeStamp'])

                startTimes = df_.rainStartTime.unique()
                accDepths = [df_[df_.rainStartTime==s].accDepth_DAP.max() for s in startTimes]
                accDepths = np.cumsum(accDepths[:-1])
                df_ = df_.assign(accDepth=df_.accDepth_DAP)
                for s, acd in zip(startTimes[1:], accDepths):
                    df_.loc[(df_.gauge==gauge) & (df_.rainStartTime==s), 'accDepth'] = \
                        df_.loc[(df_.gauge==gauge) & (df_.rainStartTime==s), 'accDepth_DAP'] + acd

            else:
                df_ = df_.assign(accDepth=df_.accDepth_DAP)

            df_pre = df_pre.append(df_)

        firstRainStart = df_pre.rainStartTime.min()
        df_pre = df_pre.assign(elapsedTime=(df_pre.timeStamp-firstRainStart).astype('timedelta64[s]')/60.)
        df_pre = df_pre.assign(rainRate=df_pre.accDepth_DAP.diff(1)/df_pre.intervalHr)
        df_pre = df_pre.assign(rainRate=[r if r>=0 else 0 for r in df_pre.rainRate])


        df = df.append(df_pre)

    newColOrder = ['preName', 'dataType', 'gauge', 'rainStartTime','timeStamp', \
                   'elapsedTime', 'intervalHr', 'accDepth_DAP', 'accDepth', 'rainRate',\
                   'rainCode', 'intensityCode_a', 'timeCode_a']

    df = df[newColOrder].reset_index(drop=True)

    return df


def get_sameRainFlag(df):

    df=df.assign(rainSprerainEHr=[(rain-rain0).total_seconds()/3600. \
                for rain, rain0 in zip(df.rainStartTime, df.rainEndTime.shift(1))])
    for i in range(1, len(df)):
        if (df.loc[i, 'rainSprerainEHr'] < 0) & \
           (df.loc[i, 'dataType']==df.loc[i-1, 'dataType']):
           df.loc[i-1:i, 'sameRainFlag'] = 1

    # add eventID (sameRainEvent) to df. This is done to differenciate connective bu independent events
    df0 = df[df.sameRainFlag==1]
    n = 1
    for i in df0.index:
        if (df0.loc[i, 'rainSprerainEHr'] > 0) & (df0.loc[i, 'sameRainFlag']>=0):
            df.loc[i:, 'sameRainEvent'] = n
            n += 1
    df.loc[df.sameRainFlag.isnull(), 'sameRainEvent']=float('NaN')

    return df


def smRainFlag_update(df, df_runoff, df_rain, df_sm, df_sedi):

    """

    """

    # get sameRainFlag
    df = get_sameRainFlag(df)


    # update rainfall
    df_rain_agg = pd.DataFrame()
    for e in np.arange(1, df.sameRainEvent.max()+1):
        events = df[df.sameRainEvent==e].preName.tolist()
        df_ = df_rain.loc[df_rain.preName.isin(events)]

        for gauge in df_.gauge.unique():

            print(f'aggregate rainfall at {gauge} for event {events}')

            df_g = df_[df_.gauge==gauge]
            df_g = df_g.assign(preName=events[0])
            df_g = df_g.sort_values(by=['timeStamp'])

            startTimes = df_g.rainStartTime.unique()
            accDepths = [df_g[df_g.rainStartTime==s].accDepth_DAP.max() for s in startTimes]
            accDepths = np.cumsum(accDepths[:-1])
            df_ = df_.assign(accDepth=df_.accDepth_DAP)
            for s, acd in zip(startTimes[1:], accDepths):
                df_g.loc[(df_g.gauge==gauge) & (df_g.rainStartTime==s), 'accDepth'] = \
                    df_g.loc[(df_g.gauge==gauge) & (df_g.rainStartTime==s), 'accDepth_DAP'] + acd

            df_g = df_g.assign(elapsedTime= \
                (df_g.timeStamp-df_.timeStamp.min()).astype('timedelta64[s]')/60.)

            df_rain_agg=df_rain_agg.append(df_g)

    df_rain =df_rain[df_rain.preName.isin(df.preName[df.sameRainFlag!=1])].append(df_rain_agg)
    df_rain = df_rain.sort_values(by=['dataType', 'preName', 'gauge','timeStamp'])


    # update runoff
    df_run_agg = pd.DataFrame()
    for e in np.arange(1, df.sameRainEvent.max()+1):
        events = df[df.sameRainEvent==e].preName.tolist()
        print(f'aggregate runoff for event {events}')
        df_ = df_runoff.loc[df_runoff.preName.isin(events)]
        firstRainStart = df_rain_agg.loc[df_rain_agg.preName==events[0], 'timeStamp'].min()
        df_=df_.assign(preName=events[0])
        df_=df_.assign(elapsedTime=(df_.timeStamp-firstRainStart).astype('timedelta64[s]')/60.)
        df_=df_.assign(intervalHr=(df_.timeStamp.diff(1)).astype('timedelta64[s]')/3600.)
        df_=df_.assign(accDepth=(df_.runoffRate_DAP*df_.intervalHr).cumsum())
        df_run_agg=df_run_agg.append(df_)

    df_runoff = df_runoff[df_runoff.preName.isin(df.preName[df.sameRainFlag!=1])].append(df_run_agg)
    df_runoff = df_runoff.sort_values(by=['dataType', 'preName', 'timeStamp'])

    # no need to change soil moisture

    # update sediment
    for e in np.arange(1, df.sameRainEvent.max()+1):
        events = df[df.sameRainEvent==e].preName.tolist()
        print(f'aggregate sediment for event {events}')
        df_sedi.loc[df_sedi.preName==events[0], 'sediKg_DAP'] = \
            df_sedi[df_sedi.preName.isin(events)].sediKg_DAP.sum()

    # update event table (update runoff peak and volume, keep the first one and delete the rest)
    # only these two variables directly from DAP are updated here
    # other variables (such as endTime, calculated runoff volume) will be updated in function get_summary_table
    for e in np.arange(1, df.sameRainEvent.max()+1):
        events = df[df.sameRainEvent==e].preName.tolist()
        df.loc[df.preName==events[0], 'runoffVolume_DAP'] = df[df.preName.isin(events)].runoffVolume_DAP.sum()
        df.loc[df.preName==events[0], 'runoffPeak_DAP'] = df[df.preName.isin(events)].runoffPeak_DAP.max()

    preToDelet = []
    for e in np.arange(1, df.sameRainEvent.max()+1):
        events = df[df.sameRainEvent==e].preName.tolist()
        preToDelet.append(events[1:])
    preToDelet = [item for sublist in preToDelet for item in sublist]
    df_updated = df[df.preName.isin(preToDelet)==False].reset_index(drop=True)


    v(f'\n\n Events with sameRainFlag aggregated:')
    for e in np.arange(1, df.sameRainEvent.max()+1):
        events = df[df.sameRainEvent==e].preName.tolist()
        v(events)

    v(f'\n\n {len(preToDelet)} Events with sameRainFlag removed:')
    for e in preToDelet:
        v(e)

    return df_updated, df_runoff, df_rain, df_sedi


def get_summary_table(df_events, df_rain, df_runoff, df_sm, df_sedi):


    df_events=df_events.assign(runTimeFlag=np.nan, runRainRatio=np.nan, \
                               noRainFlag=np.nan, noRunRateFlag=np.nan)

    for i in range(len(df_events)):

        preName = df_events.preName[i]
        print(f'get summary for event {preName}')

        #rainfall
        df_ = df_rain.loc[df_rain.preName==preName, :]
        if len(df_) == 0:
            df_events.loc[i, 'noRainFlag'] = 1
        else:
            df_events.loc[i, 'rainStartTime'] = df_.timeStamp.min()
            df_events.loc[i, 'rainEndTime']= df_.timeStamp.max()
            df_events.loc[i, 'maxRain']= df_.accDepth.max()
            df_events.loc[i, 'minRain']=df_.groupby('gauge').max().accDepth.min()
            df_events.loc[i, 'nGauges'] = len(df_.gauge.unique())
            df_events.loc[i, 'rainCode'] = df_.rainCode.sum()
            df_events.loc[i, 'intensityCode_a'] = df_.intensityCode_a.sum()
            df_events.loc[i, 'timeCode_a'] = df_.timeCode_a.sum()
            df_events.loc[i, 'rainDurMin'] = df_.elapsedTime.max()

        # runoff
        df_ = df_runoff.loc[df_runoff.preName==preName, :]
        if len(df_) == 0:
            df_events.loc[i, 'noRunRateFlag'] = 1
        else:
            df_events.loc[i, 'runoffDur_cal'] = df_.elapsedTime.max()-df_.elapsedTime.min()
            df_events.loc[i, 'startTime'] = df_.timeStamp.min()
            df_events.loc[i, 'endTime'] = df_.timeStamp.max()
            if min(np.diff(df_.elapsedTime))<0:
                df_events.loc[i, 'runTimeFlag'] = 1

        # event table
        df_events.loc[i, 'sm5'] = df_sm.loc[df_sm.preName==preName, 'sm5'].values[0]
        df_events.loc[i, 'sediKg_DAP'] = df_sedi.loc[df_sedi.preName==preName, 'sediKg_DAP'].values[0]

    if len(df_rain.gauge.unique())==1:
        df_events=df_events.assign(runRainRatio=df_events.runoffVolume_DAP/df_events.maxRain)


    return df_events


def get_flags(df, runTooLateHr, rainTooLow, runoffTooLow):

    print(f'calculate Flags for qualiy control')

    # calculate  indicators
    # rainDurMin will be used for calibration
    df = df.assign( \
        rainErunSLagHr=(df.rainEndTime- df.startTime).astype('timedelta64[s]')/3600.,
        rainSrunSLagHr=(df.rainStartTime-df.startTime).astype('timedelta64[s]')/3600.,
        rainDurMin=(df.rainEndTime-df.rainStartTime).astype('timedelta64[s]')/60,
        rainSprerainEHr=(df.rainStartTime-df.rainEndTime.shift(1)).astype('timedelta64[s]')/3600.)

    df.loc[df.maxRain < rainTooLow, 'lowRainFlag'] = 1
    df.loc[df.runoffVolume_DAP < runoffTooLow, 'lowRunFlag'] = 1

    try:
        df.loc[df.runRainRatio>0.99, 'ratioFlag'] = 1  # not available when multiple gauges
    except:
        df.loc[:, 'ratioFlag'] = np.nan

    df.loc[(df.dataType=='a') & (df.runEventCode > 0), 'runCodeFlag'] = 1
    df.loc[(df.dataType=='d') & (df.runEventCode != 1), 'runCodeFlag'] = 1
    df.loc[df.runCalc_d == 0 , 'runCalc_dFlag'] = 1

    df.loc[df.rainCode > 0, 'rainCodeFlag'] = 1
    df.loc[df.timeCode_a > 0, 'timeFlag'] = 1
    df.loc[df.intensityCode_a > 0, 'intensityFlag'] = 1


    for i in range(1, len(df)):
        if (df.loc[i, 'rainSprerainEHr'] < 0) and (df.loc[i, 'dataType'] ==df.loc[i-1, 'dataType']):
            df.loc[i-1:i, 'sameRainFlag'] = 1

    df.loc[df.rainErunSLagHr <= -runTooLateHr, 'runTooLateFlag'] = 1
    df.loc[df.rainSrunSLagHr <= -runTooLateHr, 'runTooLateFlag2'] = 1
    df.loc[df.rainSrunSLagHr > 0, 'rainLateFlag'] = 1

    newColOrder = ['flume', 'preName', 'dataType', 'startTime', 'endTime', 'runoffDur_cal', 'rainDurMin', \
                   'runoffVolume_DAP', 'runoffPeak_DAP', 'sediKg_DAP', 'runRainRatio','runEventCode',\
                   'nGauges', 'maxRain', 'minRain','rainStartTime', 'rainEndTime', \
                   'ratioFlag', 'runTimeFlag', 'missingRunDAPFlag', 'noRunRateFlag', 'noRainFlag',  'lowRainFlag', 'lowRunFlag',
                   'runCodeFlag', 'runCalc_dFlag', 'rainCodeFlag', 'timeFlag', 'intensityFlag', \
                   'sameRainFlag', 'rainLateFlag', 'runTooLateFlag', 'runTooLateFlag2']


    df = df[newColOrder].copy()

    return df


def save_results(df_runSegments, df_runEventsAgg, df_events, df_rain, \
                 df_runoff, df_runRaw, df_events_flag, wspace):

    print('Save results')

    df_runSegments.to_csv('df_runSegments.csv', index=False)
    df_runEventsAgg.to_csv('df_runEventsAgg.csv', index=False)
    df_rain.to_csv('df_rain.csv', index=False)
    df_runoff.to_csv('df_runoff.csv', index=False)
    df_events_flag.to_csv('df_events_w_flag.csv', index=False)

    # csv2excel(wspace, f'DEC_DAP_Results.xlsx')  # this takes time

    v('\n\n========= Event Summary =========\n\n')
    v(f'Number of runoff segments at this flume: {len(df_runSegments)}')
    v(f'Number of events after aggregation: {len(df_events)}')
    v(f'Number of events with rainfall: {len(df_rain.preName.unique())}')
    v(f'Number of events with missing DAP runoff: {sum((df_events_flag.missingRunDAPFlag>=1)==False)}')
    v(f'Number of Runoff Events with missing rainfall: {sum(df_events_flag.noRainFlag==1)}\n')
    v(f'{len(df_runoff)}/{len(df_runRaw)} runoff records were matched with rainfall\n\n')
