"""
This scripts contains functions for plotting summary table

Modules include:
    - histograms
    - scatter plots
    - box plots
    - by dataType

Note:
    - nan values are removed before plotting
    - warnings will show up for scatter plots without enough points for linregress
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from distutils.dir_util import copy_tree
import shutil
import glob
from dir_logging_fs import makefolders


greyText = '#3f3f3f'
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE, family='serif')
plt.rc('axes', labelsize=MEDIUM_SIZE, labelweight='bold', labelcolor=greyText)
plt.rc('xtick', labelsize=SMALL_SIZE, color=greyText)
plt.rc('ytick', labelsize=SMALL_SIZE, color=greyText)
plt.rc('legend', fontsize=MEDIUM_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)


def plot_event_summary(df):

    """ main function for creating all plots """

    print(f'plot histogram')
    plot_hist(df)
    if df.nGauges.max() == 1:
        plot_hist_single_gauge(df)

    print(f'plot scatter plots')
    plot_summary_table(df, 'analog and digital')
    plot_summary_table_ad(df, 'ad')
    plot_summary_table(df[df.dataType=='a'], 'analog')
    plot_summary_table(df[df.dataType=='d'], 'digital')

    print(f'plot box plots')
    plot_normalized_boxplot(df, 'all')
    plot_normalized_boxplot(df[df.dataType=='a'], 'analog')
    plot_normalized_boxplot(df[df.dataType=='d'], 'digital')


def custom_hist(ax, x0, xlabel, nbins, color, alpha):
    x = [x for x in x0 if ~np.isnan(x)]
    ax.hist(x, nbins, facecolor=color, edgecolor='k', alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    return ax


def plot_hist_single_gauge(df):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,6))
    nbins, color, alpha = 50, 'blue', 1.0
    custom_hist(ax[0][0], df.maxRain,'Rainfall Volume (mm)', nbins, color, alpha)
    custom_hist(ax[0][1], df.runoffVolume_DAP,'Runoff Volume (mm)', nbins, color, alpha)
    custom_hist(ax[1][0], df.runoffPeak_DAP,'Runoff Peak (mm/hr)', nbins, color, alpha)
    custom_hist(ax[1][1], df.runRainRatio, 'Rainfall Runoff Ratio', nbins, color, alpha)
    fig.tight_layout()
    plt.savefig('Histogram_single_gauge.png', dpi=300, transparent=True)
    plt.close()


def plot_hist(df):

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    nbins, color, alpha = 50, 'blue', 1.0
    custom_hist(ax[0], df.runoffVolume_DAP,'Runoff Volume (mm)', nbins, color, alpha)
    custom_hist(ax[1], df.runoffPeak_DAP,'Runoff Peak (mm/hr)', nbins, color, alpha)
    custom_hist(ax[2], df.sediKg_DAP, 'Sediment Yiled (kg)', nbins, color, alpha)
    fig.tight_layout()
    plt.savefig('Histograms.png', dpi=300, transparent=True)
    plt.close()


def plot_summary_table(df, tag):

    def custom_scatter(ax, x, y, xl, yl, label):
        if len(df)>1:
            scatter_w_lregression(ax, x, y, xl, yl, 25, 0.8, '#191919', label)
        return ax

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))
    custom_scatter(ax[0][0], df.maxRain, df.runoffVolume_DAP,'Rainfall (mm)','Runoff (mm)', tag)
    custom_scatter(ax[0][1], df.maxRain, df.runoffPeak_DAP,'Rainfall (mm)','Runoff Peak (mm/hr)', tag)
    custom_scatter(ax[0][2], df.maxRain, df.sediKg_DAP,'Rainfall (mm)','Sediment Yiled (kg)', tag)
    custom_scatter(ax[1][0], df.runoffVolume_DAP, df.runoffPeak_DAP,'Runoff (mm)','Runoff Peak (mm/hr)', tag)
    custom_scatter(ax[1][1], df.runoffVolume_DAP, df.sediKg_DAP,'Runoff (mm)','Sediment Yiled (kg)', tag)
    if max(df.nGauges.unique()) == 1:
        custom_scatter(ax[1][2], df.runoffVolume_DAP, df.runRainRatio, 'Runoff (mm)', 'Runoff/Rainfall Ratio',tag)
    plt.tight_layout()
    plt.savefig(f'scatter_plots_{tag}.png', transparent=True)
    plt.close()


def plot_normalized_boxplot(df, tag):
    columns = ['maxRain','runoffVolume_DAP', 'runoffPeak_DAP', 'sediKg_DAP']
    df_norm = pd.DataFrame([])
    df_norm[columns] = df[columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    df_norm = df_norm.rename(columns={'maxRain': 'Raifall', 'runoffVolume_DAP': 'Runoff', \
                                      'runoffPeak_DAP': 'Peak', 'sediKg_DAP': 'Sediment'})
    fig = plt.figure(figsize=(7, 5))
    df_norm.boxplot(return_type='dict', grid=False)
    plt.plot()
    plt.ylabel('Normalized Values', fontsize=18, fontweight='bold', color=greyText)
    plt.xticks(fontsize=BIGGER_SIZE, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'normalized_boxplots_{tag}.png', transparent=True)
    plt.close()


def scatter_w_lregression(ax, x, y, xl, yl, ms, alpha, color, label):
    df = pd.DataFrame().assign(x=x, y=y).dropna(how='any')
    ax.scatter(x, y, s=ms, alpha=alpha, facecolor=color, edgecolor=color, label=label)
    slope, intercept, r_value, p_value, slope_std_error = linregress(df.x.values, df.y.values)
    ax.plot(df.x, intercept + slope*df.x, color, lw=1, ls='--', label='_nolegend_')
    ax.legend()
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    return ax


def plot_summary_table_ad(df, tag):

    def custom_scatter_ad(ax, df, xv, yv, xl, yl):
        for dataType, color, label in zip(['a','d'], ['b', 'g'], ['analog', 'digital']):
            df_ad = df.loc[df.dataType==dataType, :]
            if len(df_ad)>1:
                try:
                    x = df_ad[xv]
                    y = df_ad[yv]
                    scatter_w_lregression(ax, x, y, xl, yl, 15, 1, color, label)
                except:
                    pass
        return ax

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))
    custom_scatter_ad(ax[0][0], df, 'maxRain','runoffVolume_DAP','Rainfall (mm)','Runoff (mm)')
    custom_scatter_ad(ax[0][1], df, 'maxRain','runoffPeak_DAP','Rainfall (mm)','Runoff Peak (mm/hr)')
    custom_scatter_ad(ax[0][2], df, 'maxRain','sediKg_DAP', 'Rainfall (mm)','Sediment Yiled (kg)')
    custom_scatter_ad(ax[1][0], df, 'runoffVolume_DAP','runoffPeak_DAP','Runoff (mm)','Runoff Peak (mm/hr)')
    custom_scatter_ad(ax[1][1], df, 'runoffVolume_DAP','sediKg_DAP','Runoff (mm)','Sediment Yiled (kg)')
    custom_scatter_ad(ax[1][2], df, 'runoffVolume_DAP','runRainRatio', 'Runoff (mm)', 'Runoff/Rainfall Ratio')
    plt.tight_layout()
    plt.legend()
    plt.savefig(f'scatter_plots_{tag}.png', transparent=True)
    plt.close()

