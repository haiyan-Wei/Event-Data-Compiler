import os
import math
import pandas as pd


def write_pre(df, df_rain, df_runoff, df_sm, watershed):

    """ if soil moisture available for different gauges, then add selection for gauge
        in this function, average soil moisture is used when it is not available
        users can choose other values
    """

    df_xy = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Lucky_hills_input_data', 'WGEW_gauge_locations.csv'))

    avgsm = df_sm.sm5.mean()

    for preName in df.preName:

        print(f'create .pre file for event {preName}')

        df_ = df_rain.loc[df_rain.preName==preName, :]
        if len(df_) > 0:

            preOut = f'{preName}.pre'
            f = open(os.path.join('preFiles', preOut), 'w')
            f.write(f'! {len(df_.gauge.unique())} gauge(s)\n\n')

            for gauge in df_.gauge.unique():

                X = df_xy.loc[(df_xy.Gage==gauge) & (df_xy.Watershed==watershed), 'East'].values[0]
                Y = df_xy.loc[(df_xy.Gage==gauge) & (df_xy.Watershed==watershed), 'North'].values[0]

                SAT = df_sm[df_sm.preName==preName].sm5.values[0]
                if math.isnan(SAT):
                    SAT = avgsm
                    f.write(f'! Estimated Initial Soil Moisture at gauge {gauge}\n')

                df_g = df_.loc[df_.gauge==gauge, ['elapsedTime', 'accDepth']]
                if len(df_g) > 0:
                    if df_g.elapsedTime.min() != 0.0:
                        df_g.loc[len(df_g), :]= [0,0]
                        df_g.sort_values(by=['elapsedTime'], inplace=True)

                    f.write(f'! Event Start at: {df_.timeStamp.min()} at GAUGE #{gauge}\n')
                    f.write(f'\nBEGIN GAUGE  {gauge}\n')
                    f.write(f'X =     {X}\n')
                    f.write(f'Y =     {Y}\n')
                    f.write(f'SAT =   {SAT:.4f}\n')
                    f.write(f'N =     {len(df_g)}\n\n')
                    f.write('TIME     DEPTH ! (mm)\n')
                    for minute, depth in zip(df_g.elapsedTime, df_g.accDepth):
                        f.write(f'{minute:.2f}     {depth:.4f}\n')
                    f.write('END\n\n')

            f.close()
