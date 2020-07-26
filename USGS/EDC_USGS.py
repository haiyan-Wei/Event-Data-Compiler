"""
This file contains the main EDC function for USGS surface water data.

Note: USGS revises data constantly. Download the most recent data

Copyright 2020  Haiyan Wei
haiyan.wei@usda.gov

"""


import os
import sys
import glob
import shutil
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import linregress
from distutils.dir_util import copy_tree
from dir_logging_fs import *

t0 = datetime.now()

goals = '\
    1. Download USGS rainfall and runoff data\n\
    2. Query runoff and rainfall rates for all events\n\
    3. Create .pre files\n\
    4. Plot hyetograph and hydrographs\n\
    5. Create a summary table for all events\n\n'

goals += 'Definitions:\n\
    1. record: a measured rainfall (or runoff) value with a timestamp\n\
    2. segment: a set of rainfall (or runoff) records between two zero records\n\
    3. runoff-rainfall event: a series of rainfall and runoff records (or segments) that are appropriately aligned so that the selected runoff records form an independent runoff event that is responsive to the chosen rainfall records\n\n'


out = 'July25'

flume = '07126415'
area = 126000000
gauges = ['07126415', '07126390', '07126480', '373315103493101',
          '373232103555201', '373706103410701', '373823103465601']

rainHr = 24
rainTooLow = 0.0
runoffTooLow = 0.0
rainAggHr = 6
runAggHr = 6
runTooLateHr = 6


dir_input = os.path.join(os.path.dirname(__file__), 'USGS_input_data')
gaugeXY = os.path.join(dir_input, 'gauge_flume_locations.csv')
out = f'{out}_USGS{flume}_rainHr{rainHr}_RainAgg{rainAggHr}hr_RunAgg{runAggHr}hr_runTooLateHr{runTooLateHr}'
wspace = os.path.join(os.path.dirname(__file__), 'output', out)
makefolders([wspace])
os.chdir(wspace)
logINFO('_.log', goals, t0)
logInputs([flume, gauges, wspace, dir_input])


def main():

    # downloading takes time
    # startDate, endDate = '2010-07-09', '2020-07-25'
    # donwload_USGS(dir_input, flume, gauges, startDate, endDate, t0)


    df_runRaw, df_run = read_download_text(dir_input, flume, 'runoff')
    df_rainRawS, df_rainS  = read_download_text(dir_input, gauges, 'rain')


    df_runES = aggregate_segments(df_run, runAggHr, 'runoff')
    df_rainES = aggregate_segments(df_rainS, rainAggHr, 'rain')


    df_hydroS, df_hyetoS = query_instantaneous(df_rainRawS, df_runRaw, df_rainS, df_rainES, df_runES)


    df_summary = get_summary_table(df_hydroS, df_hyetoS)
    df_summary = get_flags(df_summary, runTooLateHr, rainTooLow, runoffTooLow)


    write_pre(df_hyetoS, gaugeXY)

    plot_summary_table(df_summary)

    plot_hydro_hyeto(df_hydroS, df_hyetoS)

    group_hydrographs(df_summary)

    write_rain_runoff_excel(df_hydroS, df_hyetoS)

    df_summary.to_csv(os.path.join(wspace, 'summary.csv'))

    logging.info(f'\n\n===== THE END =====\nTime to complete: {datetime.now()-t0}')


def query_instantaneous(df_rainRawS, df_runRaw, df_rainS, df_rainES, df_runE):
    """
    """

    logging.info(f'\n\n==== Query rainfall and runoff rates ====\n\n')

    df_hyetoS, df_hydroS = {}, {}

    for i in range(len(df_runE)):

        startTime, endTime, preName = df_runE.startTime[i], df_runE.endTime[i], df_runE.preName[i]

        logging.info(f'retrieve rainfall and runoff for event {preName}')

        # initinalize firstRainStart as the end of runoff records
        firstRainStart = endTime
        df_hyeto = {}

        # get rainfall
        for gauge in list(df_rainS):

            print(f'retrieve rainfall at gauge {gauge} for event {preName}')

            df_rainE = df_rainES[gauge]
            df_rainRaw = df_rainRawS[gauge]

            # get event(s)
            #. there may be a segment across runEndTime
            #. use (startTime.endTime <= endTime), to include all segments started before runEndTime
            #. do not use (df_rainE.endTime <= endTime)
            #. also see three options of rainEndTime

            startTimeRainHr = startTime - pd.DateOffset(hours=rainHr)
            df_ = df_rainE.loc[(df_rainE.startTime >= startTimeRainHr) & \
                               (df_rainE.startTime <= endTime), :].reset_index(drop=True)


            if len(df_)== 0:
                # if there is no rainfall event(s)
                #. Two possible scenarios: all zero or no data (USGS data does include all zero measurements)
                #. all zero is needed, i.e. no rain is still data

                df = df_rainRaw.loc[(df_rainRaw.dateTime>=startTime) & \
                                    (df_rainRaw.dateTime<=endTime), :].reset_index(drop=True)

            else:
                # if there is rainfall event(s)
                # find rainfall events ended after [runStartTime adjusted with runTooLateHr]
                #. do not use df_[df_.startTime>(startTime - pd.DateOffset(hours=runTooLateHr))]
                #. b/c if all rainfall occured before runoff started, then no records will be selected

                # df_s = df_[df_.endTime>=(startTime - pd.DateOffset(hours=runTooLateHr))]

                df_s = df_.copy()

                if len(df_s) == 0:
                    df = df_rainRaw.loc[(df_rainRaw.dateTime>=startTime) & \
                                        (df_rainRaw.dateTime<=endTime), :].reset_index(drop=True)

                else:
                    # find rainStartTime
                    rainStartTime = df_s.startTime.min()

                    logging.info(f'    -No rainfall data at {gauge}')

                    # find rainEndTime
                    #. there are three options for rainEndTime,
                    #. b/c a rainfall segment may cross the runEndTime

                    #. option 1: use the end of aggregrated segment(s)
                    #. i.e. based on aggregated based on rainAggHr (greatest number of records)
                    rainEndTime = df_s.endTime.max()

                    #. option 2: use endTime (runoff ending timestamp)
                    #. i.e. based on rain record (least number of records)
                    rainEndTime = endTime

                    #. option 3 (default): use the end of the segment that across runEndTime
                    #. i.e. based on rain segment (number of records will be between 1 & 2)
                    #. if there is a segment across runEndTime,
                    #. then rainEndTime is the first value 0 after runoffEndTime
                    try:
                        rainEndTime = df_s.endTime.max()
                        df = df_rainRaw[(df_rainRaw.dateTime>=rainStartTime) & \
                                        (df_rainRaw.dateTime<=rainEndTime)].reset_index(drop=True)
                        rainEndTime = df[(df.dateTime>=endTime) & (df.value==0)].dateTime.tolist()[0]

                        df = df_rainRaw[(df_rainRaw.dateTime>=rainStartTime) & \
                                        (df_rainRaw.dateTime<=rainEndTime)].reset_index(drop=True)

                    # if not, use runEndTime
                    except:
                        df = df_rainRaw[(df_rainRaw.dateTime>=rainStartTime) & \
                                        (df_rainRaw.dateTime<=endTime)].reset_index(drop=True)

                    # get the earliest rainStartTime
                    firstRainStart = min(firstRainStart, df.dateTime.min())

            # no rainfall records
            if len(df) == 0:
                logging.info(f'    -No rainfall data at {gauge}')
                continue

            if sum(df.value>0) > 0 :
                # remove the 0 values at the end
                #. trim b/c no need to plot zero values after rain stops, and reduce size of .pre file
                #. will not trim data if all are 0, b/c df.value>0 will fail
                index0 = df[df.value>0].index[-1]
                df = df.loc[:index0+1,:]

            # unit conversion, get elapsed time and rain rates
            df = df.assign(depth_mm=df.value*25.4,
                           depthCum_mm=df.value.cumsum()*25.4,
                           elapsedTime_min=(df.dateTime - df.dateTime[0]).astype('timedelta64[m]'),
                           rainRate_mmhr=df.value*25.4/(df.usgs_interval/60.))

            df_hyeto[gauge] = df

        # adjust elapsed time to match firstRainStart
        #. if no rainfall occured before runoff ended, i.e. no rainfall data available or all are zeros
        #. then change firstRainStart to startTime, so hyetographs will match hydrographs
        #. without this, adjusted duration will be negative

        if firstRainStart == endTime:
            firstRainStart = startTime

        for gauge in list(df_hyeto):

            df_hyeto[gauge] = df_hyeto[gauge].assign( \
                elapsedTimeAdj_min=(df_hyeto[gauge].dateTime-firstRainStart).astype('timedelta64[m]'))

        # get runoff
        #. unit conversion and elapsed time shifted to match firstRainStart
        #. 1 cf/s = 0.02831713 cm/s

        print(f'retrieve runoff for event {preName}')
        df = df_runRaw.loc[(df_runRaw.dateTime>=startTime) & \
                           (df_runRaw.dateTime<=endTime), :].reset_index(drop=True)

        df=df.assign(runRate_mmhr=df.value*0.02831713/area*1000*60*60)
        df_hydro=df.assign(depthCum_mm=(df.runRate_mmhr*df.usgs_interval/60.).cumsum(),
                           elapsedTime_min=(df.dateTime-df.dateTime[0]).astype('timedelta64[m]'),
                           elapsedTimeAdj_min=(df.dateTime-firstRainStart).astype('timedelta64[m]'))

        df_hydroS[preName] = df_hydro
        df_hyetoS[preName] = df_hyeto

    return df_hydroS, df_hyetoS


def get_flags(df, runTooLateHr, rainTooLow, runTooLow):

    print(f'calculate flags for quality control')

    df=df.assign(runErainE=(df.runStart-df.rainEndFirst).astype('timedelta64[m]')/60.,
                 runSrainSL=(df.runStart-df.rainStartFirst).astype('timedelta64[m]')/60.,
                 runSrainEL=(df.runStart-df.rainEndLast).astype('timedelta64[m]')/60.,
                 lowRunFlag=[1 if x < runTooLow else np.nan for x in df.runoff],
                 lowRainFlag=[1 if x < rainTooLow else np.nan for x in df.minRain],
                 runAeFlag=[1 if 'A:e' in code else np.nan for code in df.runCode],
                 runPrFlag=[1 if 'P' in code else np.nan for code in df.runCode])

    df=df.assign(rainLateFlag=[1 if x < 0 else np.nan for x in df.runSrainSL],
                 runLateFlag=[1 if x > runTooLateHr else np.nan for x in df.runSrainEL],
                 runLateFlag2=[1 if x > runTooLateHr else np.nan for x in df.runErainE])

    if df.nGauges.max() == 1:
        df=df.assign(ratioFlag=[a> b for a, b in zip(df.runoff, df.avgRain)])

    return df


def write_pre(df_hyetoS, gaugeXY):

    logging.info(f'\n\n==== Create .pre file for RHEM ====\n\n')

    dir_pre = os.path.join(wspace, 'preFiles')
    makefolders([dir_pre])

    df_xy = pd.read_csv(gaugeXY)

    for preName in list(df_hyetoS):

        print(f'create .pre files for event {preName}')

        df_rain = df_hyetoS[preName]

        if len(df_rain) > 0:

            f = open(os.path.join(wspace, 'preFiles', f'{preName}.pre'), 'w')

            f.write(f'! {len(df_rain)} gauge(s)\n\n')

            for gauge in list(df_rain):

                X = df_xy.loc[(df_xy.ID==f'G{gauge}'), 'Easting'].values[0]
                Y = df_xy.loc[(df_xy.ID==f'G{gauge}'), 'Northing'].values[0]
                SAT = 0.2
                f.write(f'\nBEGIN GAUGE  {gauge}\n')
                f.write(f'X =     {X}\n')
                f.write(f'Y =     {Y}\n')
                f.write(f'SAT =   {SAT:.4f}\n')

                df = df_rain[gauge][['elapsedTimeAdj_min', 'depthCum_mm']]

                if df.elapsedTimeAdj_min.min() == 0.0:
                    f.write(f'N =     {len(df)}\n\n')
                else:
                    f.write(f'N =     {len(df) + 1}\n\n')

                f.write('TIME     DEPTH ! (mm)\n')

                if df.elapsedTimeAdj_min.min() != 0.0:
                    f.write('0.0   0.0 \n')

                for minute, depth in zip(df.elapsedTimeAdj_min, df.depthCum_mm):
                    f.write(f'{minute:.2f}     {depth:.4f}\n')

                f.write('END\n\n')

            f.close()

    logging.info(f'{len(list(df_hyetoS))} .pre files are created. \n')


def plot_hydro_hyeto(df_hydroS, df_hyetoS):

    logging.info(f'\n\n==== Create hyetographs and hydrographs for each event ====\n\n')

    def plot_rain(ax, df_rainS, preName, item):

        for gauge in list(df_rainS):

            df = df_rainS[gauge]
            df_codeAe = df.loc[df.code=='A:e', :]
            df_codeP = df.loc[df.code=='P', :]
            df_codeA = df.loc[df.code=='A', :]

            label = f'gauge {gauge}, rain={df.depthCum_mm.max():.2f}mm'

            if item == 'cumsum':
                plt.title(f'Runoff Event {preName} at USGS flume {flume}', fontweight='bold', fontsize=titleSize, color=greyText)
                plt.plot(df.elapsedTimeAdj_min, df.depthCum_mm, '-o', markersize=msize, label=label)

                if len(df_codeAe) > 0:
                    plt.plot(df_codeAe.elapsedTimeAdj_min, df_codeAe.depthCum_mm, '*', markersize=msize+3, label='estimated', color='r')

                if len(df_codeP) > 0:
                    plt.plot(df_codeP.elapsedTimeAdj_min, df_codeP.depthCum_mm, '^', markersize=msize+3, label='provisional', color='r')

                get_plt(ax, 'Accumulated Rainfall\n(mm)', get_xticks())
                legend_without_duplicate_labels(ax)


            if item == 'rate':
                plt.plot(df.elapsedTimeAdj_min, df.rainRate_mmhr, '-o', markersize=msize, label=label)
                if len(df_codeAe) > 0:
                    plt.plot(df_codeAe.elapsedTimeAdj_min, df_codeAe.rainRate_mmhr, '*', markersize=msize+3, label='estimated', color='r')

                if len(df_codeP) > 0:
                    plt.plot(df_codeP.elapsedTimeAdj_min, df_codeP.rainRate_mmhr, '^', markersize=msize+3, label='provisional', color='r')

                get_plt(ax, 'Rainfall Rate\n(mm/hr)', get_xticks())

                legend_without_duplicate_labels(ax)


    def plot_runoff(ax, df):

        df_codeAe = df.loc[df.code=='A:e', :]
        df_codeP = df.loc[df.code=='P', :]
        df_codeA = df.loc[df.code=='A', :]

        label = f'Runoff: {df_run.depthCum_mm.max():.2f}mm'
        plt.plot(df.elapsedTimeAdj_min, df.runRate_mmhr, '-o',
                 label=label, color='b', markersize=msize+2)
        if len(df_codeAe) > 0:
            plt.plot(df_codeAe.elapsedTimeAdj_min, df_codeAe.runRate_mmhr, '*',
                 label='estimated', color='r', markersize=msize+4)
        if len(df_codeP) > 0:
            plt.plot(df_codeP.elapsedTimeAdj_min, df_codeP.runRate_mmhr, '^', \
                 markersize=msize+4, label='provisional', color='r')
        get_plt(ax, 'Runoff Rate\n(mm/hr)', get_xticks())
        plt.xlabel('Time (min)', fontsize=labelSize, fontweight='bold', color=greyText)


    def get_xticks():
        if xlim > 600:
            xticks = np.arange(0, xlim, 120)
        elif xlim > 360:
            xticks = np.arange(0, xlim, 60)
        elif xlim> 60:
            xticks = np.arange(0, xlim, 30)
        else:
            xticks = np.arange(0, xlim, 10)
        return xticks


    def get_plt(ax, ylabel, hrticks):

        plt.xlim([min(xmin, xlim*(-0.01)), xlim*1.05])
        plt.xticks(hrticks)
        plt.xticks(fontsize=tickSize)
        plt.yticks(fontsize=tickSize)
        plt.legend(frameon=False, fontsize=legendSize)
        plt.ylabel(ylabel, fontsize=labelSize, fontweight='bold', color=greyText)


    def legend_without_duplicate_labels(ax):
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique), fontsize=legendSize)


    dir_plots = os.path.join(wspace, 'hyeto_hydrograph')
    makefolders([dir_plots])

    for preName in list(df_hydroS):

        print(f'plot hydrograph and hyetograph for event {preName}')

        msize, labelSize, tickSize, legendSize, titleSize, greyText = 3, 15, 12, 13, 15, '#3f3f3f'
        df_run, df_rainS = df_hydroS[preName], df_hyetoS[preName]
        xmin, xlim = df_run.elapsedTimeAdj_min.min(), df_run.elapsedTimeAdj_min.max()

        fig = plt.subplots(figsize=(15, 8))

        if len(df_rainS) > 0:

            xlim_rain = max([df_rainS[gauge].elapsedTimeAdj_min.max() for gauge in list(df_rainS)])
            xlim = max(xlim_rain, xlim)

            ax1 = plt.subplot(2, 1, 1)
            plot_rain(ax1, df_rainS, preName, 'cumsum')
            ax2 = plt.subplot(2, 1, 2)
            plot_runoff(ax2, df_run)

            # ax1 = plt.subplot(3, 1, 1)
            # plot_rain(ax1, df_rainS, preName, 'cumsum')
            # ax2 = plt.subplot(3, 1, 2)
            # plot_rain(ax2, df_rainS, preName, 'rate')
            # plt.subplot(3, 1, 3)
            # plot_runoff(ax2, df_run)

        else:
            xmin, xlim = df_run.elapsedTime_min.min(), df_run.elapsedTime_min.max()
            df_run.elapsedTimeAdj_min = df_run.elapsedTime_min
            plt.subplot(3, 1, 1)
            plt.title(f'Runoff Event {preName}', fontweight='bold', fontsize=titleSize, color=greyText)
            plt.subplot(3, 1, 2)
            ax3=plt.subplot(3, 1, 3)
            plot_runoff(ax3, df_run)

        plt.tight_layout()
        plt.savefig(os.path.join(dir_plots, f'{preName}.png'))
        plt.close()


def get_summary_table(df_hydroS, df_hyetoS):

    logging.info(f'\n\n==== Create a summary table ====\n\n')
    print('create a summary table')

    df = pd.DataFrame(index=list(df_hydroS))

    # get runoff summary
    df = df.assign(preName=df.index,
                   runStart=[df_hydroS[p].dateTime.min() for p in df_hydroS],
                   runEnd= [df_hydroS[p].dateTime.max() for p in df_hydroS],
                   runoff = [df_hydroS[p].depthCum_mm.max() for p in df_hydroS],
                   peak = [df_hydroS[p].runRate_mmhr.max() for p in df_hydroS],
                   runDur = [df_hydroS[p].elapsedTime_min.max() for p in df_hydroS],
                   runCode = [df_hydroS[p].code.unique() for p in df_hydroS])

    # get rainfall summary
    for p in df.index:
        if len(df_hyetoS[p]) > 0:
            df.loc[p, 'nGauges'] = len(df_hyetoS[p])
            df.loc[p, 'avgRain'] = np.mean([df_hyetoS[p][g].depthCum_mm.max() for g in list(df_hyetoS[p])])
            df.loc[p, 'minRain'] = np.min([df_hyetoS[p][g].depthCum_mm.max() for g in list(df_hyetoS[p])])
            df.loc[p, 'maxRain'] = np.max([df_hyetoS[p][g].depthCum_mm.max() for g in list(df_hyetoS[p])])
            df.loc[p, 'rainPeak'] = np.max([df_hyetoS[p][g].rainRate_mmhr.max() for g in list(df_hyetoS[p])])
            df.loc[p, 'rainStartFirst'] = np.min([df_hyetoS[p][g].dateTime.min() for g in list(df_hyetoS[p])])
            df.loc[p, 'rainStartLast'] = np.max([df_hyetoS[p][g].dateTime.min() for g in list(df_hyetoS[p])])
            df.loc[p, 'rainEndLast'] = np.max([df_hyetoS[p][g].dateTime.max() for g in list(df_hyetoS[p])])
            df.loc[p, 'rainEndFirst'] = np.min([df_hyetoS[p][g].dateTime.max() for g in list(df_hyetoS[p])])

    return df.reset_index(drop=True)


def write_rain_runoff_excel(df_hydroS, df_hyetoS):

    print('save results into an excel file for each event')
    dir_csv = os.path.join(wspace, 'xlsFiles')
    makefolders([dir_csv])
    os.chdir(dir_csv)

    for preName in list(df_hydroS):
        df_hydroS[preName].to_csv(f'Runoff.csv', index=False)
        for gauge in list(df_hyetoS[preName]):
            df_hyetoS[preName][gauge].to_csv(f'{gauge}.csv', index=False)

        writerXLS = pd.ExcelWriter(f'{preName}.xlsx')
        csvfiles = glob.glob('*.csv')
        for csvfile in csvfiles:
            pd.read_csv(csvfile).to_excel(writerXLS, csvfile[:-4])
            os.remove(csvfile)
        writerXLS.save()


def plot_summary_table(df):

    logging.info(f'\n\n==== Create scatter plot and box plot for outlier detection ====\n\n')

    greyText = '#3f3f3f'

    def plotxy(df, xl, yl):

        msize, labelsize, tickSize = 35, 18, 15

        df.dropna(how='any', axis=0, inplace=True)
        if len(df)>1:
            x = df.iloc[:, 0]
            y = df.iloc[:, 1]
            slope, intercept, _, _, _ = linregress(x, y)
            plt.scatter(x, y, s=msize, alpha=0.8, facecolor='#191919', edgecolor='None')
            plt.plot(x, intercept + slope*x, '#0c0c0c')
        plt.xlabel(xl, fontsize=labelsize, fontweight='bold', color=greyText)
        plt.ylabel(yl, fontsize=labelsize, fontweight='bold', color=greyText)
        plt.xticks(fontsize=tickSize)
        plt.yticks(fontsize=tickSize)

    fig = plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plotxy(df.loc[:, ['avgRain','runoff']],'Rainfall (mm)','Runoff (mm)')
    plt.subplot(1, 2, 2)
    plotxy(df.loc[:, ['runoff','peak']],'Runoff (mm)','Runoff Peak (mm/hr)')
    plt.tight_layout()
    plt.savefig(os.path.join(wspace, f'scatter_plots.png'), transparent=True)
    plt.close()


    df_norm = pd.DataFrame([])
    cols_to_norm = ['avgRain','minRain','maxRain','runoff', 'peak']
    df_norm[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    fig = plt.figure(figsize=(10, 5))
    df_norm.boxplot(return_type='dict', grid=False)
    plt.plot()
    plt.xticks(fontsize=15, fontweight='bold', color=greyText)
    plt.yticks(fontsize=12)
    plt.ylabel('Normalized Values', fontsize=18, fontweight='bold', color=greyText)
    plt.tight_layout()
    plt.savefig(os.path.join(wspace, f'normalized_boxplots.png'), transparent=True)
    plt.close()


def aggregate_segments(dfS, hr, item):

    logging.info(f'\n\n==== Aggregate {item} Segments Based on AggHr ====\n')

    if item == 'rain':
        dfES = {}
        for gauge in list(dfS):
            dfES[gauge] = aggregate(dfS[gauge], gauge, hr, item)

    if item == 'runoff':
        dfES = aggregate(dfS, flume, hr, item)
        dfES = dfES.assign(preName=[s.strftime('%Y%m%d-%H%M%S') for s in dfES.startTime])

    return dfES


def aggregate(df, location, hr, item):

    # get startTime and endTime
    startIndex=[i for i in df.loc[df.intervalHr>hr, 'dateTime'].index]
    startIndex.insert(0, 0)
    endIndex=[i-1 for i in startIndex[1:]]
    endIndex.append(len(df)-1)

    # get all events
    df_e = pd.DataFrame()
    for i, s, e in zip(range(len(startIndex)), startIndex, endIndex):
        # actual startTime is the timestep plus usgs_interval
        df_e.loc[i, 'startTime'] = df.loc[s, 'dateTime'] - pd.DateOffset(minutes=df.loc[s,'usgs_interval'])
        df_e.loc[i, 'endTime'] = df.loc[e, 'dateTime']

    df_e=df_e.assign(durMin=(df_e.endTime-df_e.startTime).astype('timedelta64[m]'))
    df_e=df_e.assign(intervalHr=(df_e.startTime - df_e.endTime.shift(1)).astype('timedelta64[m]')/60.)

    logging.info(f'Number of {item} events at {location}:  {len(df_e)}')
    # df_e.to_csv(os.path.join(wspace, f'{item}_{location}.csv'))

    return df_e


def read_download_text(dir_input, locations, item):

    """
        Data status codes:
        ***         Temporarily unavailable
        --          Parameter not determined
        Dis         Data-collection discontinued
        Eqp         Equipment malfunction
        Fld         Flood damage
        Ssn         Parameter monitored seasonally
        Tst         Value is affected by artificial test condition.
        A:e         estimated
        P           provisional
    """

    logging.info(f'\n\n========= Read {item} Text File =========\n')
    os.chdir(dir_input)

    if item == 'rain':
        df_rawS, df_noneZeroS = {}, {}
        for location in locations:
            try:
                df_raw, df_none0 = read_text(location, item)
                df_rawS[location] = df_raw
                df_noneZeroS[location] = df_none0
            except:
                pass

    if item == 'runoff':
        df_rawS, df_noneZeroS = read_text(flume, item)

    return df_rawS, df_noneZeroS


def read_text(location, item):

        df0 = pd.read_csv(f'{location}_{item}.txt', comment='#', header=None, sep='\t')[2:]

        # value is runoff rate for runoff and depth for rainfall
        df0.columns = ['agency', 'location', 'dateTime', 'tzone', 'value', 'code']
        df0 = df0.drop(['agency', 'tzone'], axis=1)
        df0.dateTime = pd.to_datetime(df0.dateTime)

        # remove non-numeric values
        df0['value'] = pd.to_numeric(df0['value'], errors='coerce')
        df = df0.dropna(subset=['value'])
        # calculate usgs_interval
        df = df.assign(usgs_interval=(df.dateTime - df.dateTime.shift(1)).astype('timedelta64[m]'))
        df_raw = df

        logging.info(f'{item} at location {location}:')
        logging.info(f'    -{len(df0) - len(df)} non-numeric values')
        logging.info(f'    -{len(df)} records available')

        # interval values
        intervals = np.sort([int(a) for a in df.usgs_interval[1:].unique()])
        intervals = ','.join(map(str, intervals))

        # remove 0
        df_none0 = df.loc[df.value != 0., :].reset_index()
        df_none0 = df_none0.assign(intervalHr=(df_none0.dateTime.diff(1).astype('timedelta64[m]')/60.))

        logging.info(f'    -{len(df_none0)} non-zero records')
        logging.info(f'    -intervals in min: {intervals}\n')

        return df_raw, df_none0


def donwload_USGS(dir_input, flume, gauges, startDate, endDate, t0):

    """ double check NWIS website to get most updated url  """

    import urllib.request

    os.chdir(dir_input)
    logging.info('\n\n========= Download Data from USGS =========\n')
    logging.info(f'startDate: {startDate}, endDate: {endDate}\n')

    try:
        logging.info(f'Download flow data at {flume}...')
        url = f'https://nwis.waterdata.usgs.gov/co/nwis/uv?cb_00060=on&format=rdb&site_no={flume}&period=&begin_date={startDate}&end_date={endDate}'
        urllib.request.urlretrieve(url, f'{flume}_runoff.txt')

    except:
        logging.info(f'Download flow data at {flume} failed')


    for gauge in gauges:
        print(f'download rainfall data at gauge {gauge}')
        try:
            logging.info(f'Download rainfall data at {gauge}...')
            url = f'https://nwis.waterdata.usgs.gov/co/nwis/uv?cb_00045=on&format=rdb&site_no={gauge}&period=&begin_date={startDate}&end_date={endDate}'
            urllib.request.urlretrieve(url, f'{gauge}_rain.txt')

        except:
            logging.info(f'Download rainfall data at {gauge} failed')

    logging.info(f'\nTime to complete download: {datetime.now()-t0}')


def group_hydrographs(df):

    """
    based on the flags in df:
    copy figures in hyeto_hydrograph into hyeto_hydrograph_bygroup
    enter hyeto_hydrograph_bygroup and group by flags
    """

    def move_fig(df, flags, newFolder):
        makefolders([newFolder])
        flagged_events=[]
        for flag in flags:
            # this condition is only for ratioFlag, which is only calculated for single gauge
            if flag in list(df):
                for preName in df.loc[df[flag] == 1, 'preName']:
                    fig = f'{preName}.png'
                    if os.path.exists(fig):
                        shutil.move(fig, os.path.join(newFolder, fig))  # move or copy
                        flagged_events.append(preName)
        return flagged_events

    os.chdir(wspace)
    copy_tree('hyeto_hydrograph', 'hyeto_hydrograph_bygroup')
    os.chdir('hyeto_hydrograph_bygroup')
    makefolders(['unflagged_events'])

    flagged_events=[]
    flagged_events.append(move_fig(df, ['lowRunFlag'], 'lowRun'))
    flagged_events.append(move_fig(df, ['lowRainFlag'], 'lowRain'))
    flagged_events.append(move_fig(df, ['runAeFlag', 'runPrFlag'], 'estimated_provisional'))
    flagged_events.append(move_fig(df, ['ratioFlag'], 'ratiogt1'))
    flagged_events.append(move_fig(df, ['runLateFlag1'], 'runTooLate'))
    flagged_events.append(move_fig(df, ['runLateFlag2'], 'runTooLate2'))

    for fig in glob.glob('*.png'):
        shutil.move(fig, os.path.join('unflagged_events', fig))


if __name__ == "__main__":
    main()
