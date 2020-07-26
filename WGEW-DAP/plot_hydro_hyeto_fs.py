"""
Modules to plot hydrograph and hyetograph

main modules:
    - plot_hyeto_hydro_all_events
    - group_hydrographs

Note:
    - If number of gauges is greater than 3, hyetographs will look busy.
      Recommend using the 2 panels with just accu. rainfall plotted. also, try line without markers
"""

import os
import glob
import shutil
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from distutils.dir_util import copy_tree
from dir_logging_fs import makefolders


greyText = '#3f3f3f'
labelSize, tickSize, legendSize, titleSize, msize = 12, 10, 10, 15, 5


def plot_hyeto_hydro_all_events(flume, df, df_rain, df_runoff):

    print(f'\n\n*Events without rainfall and runoff data\n')

    for preName in df.preName:

        df_rain_ = df_rain[df_rain.preName==preName]
        df_run_ = df_runoff[df_runoff.preName==preName]

        try:
            DAP_runoffVolume=df[df.preName==preName].runoffVolume_DAP.values[0]
            plot_hyeto_hydro_3panels(flume, preName, df_rain_, df_run_, DAP_runoffVolume)
            print(f'plot hydrograph and hyetograph for event {preName}')
        except:
            logging.info(f'no graph for event {preName}')


def get_xlim_hrticks(df_rain, df_run):

    if (len(df_rain) > 0) & (len(df_run)>0):
        xlim = max(df_rain.elapsedTime.max(), df_run.elapsedTime.max())*1.05
    elif len(df_rain) > 0:
        xlim = df_rain.elapsedTime.max()*1.05
    elif len(df_run) > 0:
        xlim = df_run.elapsedTime.max()*1.05

    if xlim > 360:
        hrticks = np.arange(0, xlim, 60)
    elif xlim> 60:
        hrticks = np.arange(0, xlim, 30)
    else:
        hrticks = np.arange(0, xlim, 10)
    return xlim, hrticks


def get_plt(xlable, ylabel, xlim, hrticks):
    plt.xlim([-10, xlim])
    plt.xticks(hrticks)
    plt.xticks(fontsize=tickSize)
    plt.yticks(fontsize=tickSize)
    plt.legend(frameon=False, fontsize=legendSize)
    plt.xlabel(xlable, fontsize=labelSize, fontweight='bold', color=greyText)
    plt.ylabel(ylabel, fontsize=labelSize, fontweight='bold', color=greyText)


def plot_hyeto_hydro_3panels(flume, preName, df_rain, df_run, runVDAP):

    xlim, hrticks = get_xlim_hrticks(df_rain, df_run)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax1 = plt.subplot(3, 1, 1)
    plt.title(f'Flume {flume}, Event {preName}', fontsize=titleSize, fontweight='bold', color=greyText)

    if len(df_rain) > 0:
        for gauge, col in zip(df_rain.gauge.unique(), ['g', 'k', 'gold']):
            df_ = df_rain.loc[df_rain.gauge==gauge, :]
            label = f'gauge {gauge}, total rainfall={df_.accDepth.max():.2f}mm'
            plt.plot(df_.elapsedTime, df_.accDepth, '-o', markersize=msize,label=label, color=col)
            plt.plot(df_[df_.intensityCode_a>0].elapsedTime, \
                        df_[df_.intensityCode_a>0].accDepth, '*', \
                        markersize=msize+3, label='estimated rain', color='r')
            plt.plot(df_[df_.rainCode>0].elapsedTime, df_[df_.rainCode>0].accDepth, '*', \
                         markersize=msize+3, label='estimated rain', color='r')
        get_plt('', 'Acc. Rainfall (mm)', xlim, hrticks)
        legend_without_duplicate_labels(ax1)


    ax2 = plt.subplot(3, 1, 2)
    if len(df_rain) > 0:
        for gauge, col in zip(df_rain.gauge.unique(), ['g', 'k', 'gold']):
            df_ = df_rain.loc[df_rain.gauge==gauge, :]
            label = f'gauge {gauge}, total rainfall={df_.accDepth.max():.2f}mm'
            plt.plot(df_.elapsedTime, df_.rainRate, '-o', markersize=msize,label=label, color=col)
            plt.plot(df_[df_.intensityCode_a>0].elapsedTime, df_[df_.intensityCode_a>0].rainRate, '*', \
                        markersize=msize+3, label='estimated rain', color='r')
            plt.plot(df_[df_.rainCode>0].elapsedTime, df_[df_.rainCode>0].rainRate, '*', \
                    markersize=msize+3, label='estimated rain', color='r')
        get_plt('', 'Rainfall Rate (mm/hr)', xlim, hrticks)
        legend_without_duplicate_labels(ax2)

    plt.subplot(3, 1, 3)
    obsRunoff = df_run.accDepth.max()
    label = f'Total Runoff: Cal:{obsRunoff:.2f}/DAP:{runVDAP:.2f}mm'
    if len(df_run) > 0:
        plt.plot(df_run.elapsedTime, df_run.runoffRate_DAP, '-o', label=label, color='b', markersize=7)
        if sum(df_run.runCode)>0:
            plt.plot(df_run[df_run.runCode>0].elapsedTime, df_run[df_run.runCode>0].runoffRate_DAP, '*',\
                     markersize=msize+3, label='estimated runoff', color='r')
        get_plt('Time (min)', 'Runoff Rate (mm/hr)', xlim, hrticks)
    else:
        plt.text(0.1, 0.8, f'No runoff rates. Total Runoff: Cal:{obsRunoff:.2f}/DAP:{runVDAP:.2f}mm', color=greyText, fontsize=labelSize)

    plt.tight_layout()
    figname = os.path.join('hyeto_hydro_graphs', f'{preName}.png')
    plt.savefig(figname, transparent=True)
    plt.close()


def group_hydrographs(df):

    """
    based on the flags in df:
    copy figures in hyeto_hydro_graphs into hyeto_hydro_graphs_bygroup
    enter hyeto_hydro_graphs_bygroup and group by flags    """

    print('group hydrographs and hyetographs by flags')

    def move_fig(df, flags, newFolder):
        makefolders([newFolder])
        flagged_events=[]
        for flag in flags:
            for preName in df.loc[df[flag] >= 1, 'preName']:
                fig = f'{preName}.png'
                if os.path.exists(fig):
                    shutil.copy(fig, os.path.join(newFolder, fig))
                    flagged_events.append(preName)
        return flagged_events

    if os.path.exists('hyeto_hydro_graphs_bygroup'):
        shutil.rmtree('hyeto_hydro_graphs_bygroup')
    copy_tree('hyeto_hydro_graphs', 'hyeto_hydro_graphs_bygroup')
    os.chdir('hyeto_hydro_graphs_bygroup')
    makefolders(['unflagged_events'])

    flagged_events=[]
    flagged_events.append(move_fig(df, ['runTimeFlag'], 'runTime'))
    flagged_events.append(move_fig(df, ['rainLateFlag'], 'rainLate'))
    flagged_events.append(move_fig(df, ['noRunRateFlag'], 'noRunRate'))
    flagged_events.append(move_fig(df, ['noRainFlag'], 'noRain'))
    flagged_events.append(move_fig(df, ['missingRunDAPFlag'], 'missingRunDAPFlag'))
    flagged_events.append(move_fig(df, ['ratioFlag'], 'ratiogt1'))
    flagged_events.append(move_fig(df, ['sameRainFlag'], 'sameRain'))
    flagged_events.append(move_fig(df, ['runTooLateFlag'], 'runTooLate'))
    flagged_events.append(move_fig(df, ['runTooLateFlag2'], 'runTooLate2'))
    flagged_events.append(move_fig(df, ['runCodeFlag' ,'rainCodeFlag', 'timeFlag' ,'intensityFlag'], 'estFlags'))

    for fig in glob.glob('*.png'):
        if fig not in [item + '.png' for sublist in flagged_events for item in sublist]:
            shutil.copy(fig, os.path.join('unflagged_events', fig))
        os.remove(fig)


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), fontsize=legendSize)
    return ax
