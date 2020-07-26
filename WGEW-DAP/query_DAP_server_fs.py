
""" query functions

This file contains functions to link to the WGEW-DAP database and make SQL queries

    modules include:
       - query_DAP_runoff_segments
       - query_DAP_runoff_rates
       - query_DAP_rain_rates
       - query_DAP_sedi_mass

    * make sure to have the correct username, password to connect to DAP
    * some variables are named with suffix '_DAP', which differentiate them from the calculated variables

"""
import os
import pyodbc
import pandas as pd
import pandas.io.sql as sql


# read username and password and connnect to DAP database
with open(os.path.join(os.path.dirname(__file__), 'DAP_credential.txt'), 'r') as f:
    lines = f.readlines()
server, userid, pw = lines[0][:-1], lines[1][:-1], lines[2][:-1]
conn = pyodbc.connect('Driver={SQL Server};Server='+server+';Database=dap;UID='+userid+';PWD='+pw)


def query_DAP_runoff_segments(watershed, flume):

    """
        objective:
        get ALL runoff events for the selected watershed and flume,
        output units: mm, mm/hr, min

        !!! 'runCode = 1' has different meaning for analog and digital runoff data

        runCalc_d: in digital data only, '1' means calculation process is done (good)
        runCode: in analog data, '1' means estimated runoff

        runCode: in digital data:
                -1	All Data not Present
                0	Not Quality Checked
                1	Quality Checked - Good
                2	Maintenance
                3	Vandalism
                4	Animal Disturbance
                5	Malfunction
                6	Unknown
                7	CONDENSATION
                8	OUTSIDE DISTURBANCE
                9	rain on flume
                10	Shovel left in intake
                11	Not an event
                12	Baseflow
                13	Bucket touching side of rain gage
                14	cone fell in bucket
                15	Intake full of sand not a flow
                16	fire destroyed wires on rain gage
                17	hole in bucket depth is small.
                18	No corresponding precipitation
                19	No Volumes Yet
                20	No Data for ths Year
                21	Volumes rough calc
                25	Single Point Event
                23	Zero or Less Max Depth
                24	Weir Level Set Too Low
                26	Zero Volume Event
                27	Insufficient precipitation to justify event
                22	Fall in Tank Level

        """

    flume = f'({flume})'

    sql_str_a = """select eventid, station as flume
                    , convert(varchar(10), date_time,120) as 'Date'
                    , convert(varchar(20), date_time,108) as 'time'
                    , duration as duration_DAP
                    , WatershedVol as 'runoffVolume_DAP'
                    , stationpeakrunoffrate as 'runoffPeak_DAP'
                    , estcode as runEventCode
                    from AnalogRunOffEvents
                    where watershed = """ + str(watershed) + """
                    and station in """ + flume + """
                     order by date, time
                    """

    sql_str_d = """select rp.eventid, f.webid%1000 as 'flume',
                     convert(varchar(10), starttime, 120) as 'Date',
                     convert(varchar(20), starttime, 108) as 'time',
                     duration_DAP,
                     volume as 'runoffVolume_DAP',
                     runoffPeak_DAP,
                     code as runEventCode,
                     calc as runCalc_d
                     from runoffevents re
                     join (select eventid, max(elapsedTime) as 'duration_DAP',
                     max(runoffrate) as 'runoffPeak_DAP'
                     from runoffpoints group by eventid) rp on rp.eventID = re.id
                     join flume f on sensorid = f.id
                     where f.webID/1000 = """ + str(watershed) + """
                     and f.webid%1000 in """ + flume + """ order by date, time"""

    df_a = sql.read_sql(sql_str_a,conn).assign(dataType='a')
    df_d = sql.read_sql(sql_str_d,conn).assign(dataType='d')

    df = pd.concat([df_a, df_d])
    df = df.assign(runoffVolume_DAP= df.runoffVolume_DAP*25.4,
                   runoffPeak_DAP=df.runoffPeak_DAP*25.4,
                   startTime= pd.to_datetime(df.Date + ' ' + df.time))

    df = df.assign(endTime=[s + pd.DateOffset(minutes=d) for s, d in zip(df.startTime, df.duration_DAP)],
                   preName=[ad + s.strftime('%Y%m%d-%H%M%S') for ad, s in zip(df.dataType, df.startTime)])

    df = df.drop(columns=['Date', 'time', 'eventid'])
    df = df.sort_values(by=['dataType', 'startTime']).reset_index(drop=True)

    return df


def query_DAP_runoff_rates(watershed, flume):

    flume = f'({flume})'

    sql_str_d = """select webid%1000 as 'flume'
                , convert(varchar(10), starttime, 120) as 'Date'
                , convert(varchar(20),starttime,108) as 'time'
                , rp.runoffrate as 'runoffRate_DAP'
                , rp.elapsedtime as elapsedTime_DAP
                , rp.depth as 'runoffDepth_DAP'
                , str(rp.code) as 'runCode'
                from runoffevents re
                join runoffpoints rp on re.id = rp.eventID
                join flume f on sensorid = f.id
                where f.webID/1000 = """ + str(watershed)+ """
                and f.webid%1000 in  """ + flume

    sql_str_a = """select station as flume
                  , convert(varchar(10), date_time, 120) as 'Date'
                  , convert(varchar(20), date_time,108) as 'time'
                  , ab.watershedRunoffRate as 'runoffRate_DAP'
                  , ab.elapsedtime as elapsedTime_DAP
                  , ab.Depth as 'runoffDepth_DAP'
                  , ab.Estcode as runCode
                  from analogrunoffevents ae
                  join analogrunoffbreaks  ab on ae.eventid = ab.eventid
                  where Watershed = """ + str(watershed) + """
                  and station in """ + flume + """
                  """

    df_a = sql.read_sql(sql_str_a, conn).assign(dataType='a')
    df_d = sql.read_sql(sql_str_d, conn).assign(dataType='d')
    df = df_a.append(df_d)

    df = df.assign(runoffDepth_DAP=df.runoffDepth_DAP*25.4,
                   runoffRate_DAP=df.runoffRate_DAP*25.4,
                   startTime= pd.to_datetime(df.Date + ' ' + df.time))

    df = df.assign(timeStamp=[s + pd.DateOffset(minutes=e) \
                      for s, e in zip(df.startTime, df.elapsedTime_DAP)])

    df = df.drop(columns=['Date', 'time'])
    df = df.sort_values(by=['dataType', 'timeStamp']).reset_index(drop=True)

    return df


def query_DAP_rain_rates(watershed, gauge):

    '''connect to DAP sql seaver and query breakpoint rainfall data by watershed, gauge'''

    gauge = f'({gauge})'

    sql_str_a = """ select ape.gage as gauge,
                convert(date,date_time) as 'Date',
                convert(varchar,date_time,8) as 'starttime',
                apb.duration as 'elapsedTime',
                apb.AccumRainfall as accDepth_DAP,
                apb.RainfallCode as rainCode, apb.Intensitycode as intensityCode_a, apb.TimeCode as timeCode_a
                from analogprecipbreaks apb
                join analogprecipevents ape on ape.id = apb.eventID
                where Watershed = """ + str(watershed) + """
                and Gage in """ + gauge + """
                order by gauge, date, starttime, elapsedTime """

    sql_str_d = """ select r.webid%1000 as 'gauge',
                convert(date,pe.starttime) as 'Date',
                convert(varchar, pe.starttime, 8) as 'starttime',
                pp.depth as accDepth_DAP,
                pp.elapsedTime,
                pp.code as rainCode,
                pe.starttime as 'rainStartTime'
                from precipEvents pe
                join precippoints pp on pe.id = pp.eventID
                join raingage r on r.id = pe.sensorID
                where r.webid/1000 = """ + str(watershed) + """
                and r.webid%1000 in """ + gauge + """
                order by 'gauge', date, pe.starttime, elapsedTime """

    df_a = sql.read_sql(sql_str_a, conn).assign(dataType='a')
    df_d = sql.read_sql(sql_str_d, conn).assign(dataType='d')
    df = df_a.append(df_d)
    df = df.assign(rainStartTime=pd.to_datetime(df.Date + ' ' + df.starttime),
                   accDepth_DAP=df.accDepth_DAP*25.4)
    df = df.assign(timeStamp=[s + pd.DateOffset(minutes=e) \
                   for s, e in zip(df.rainStartTime, df.elapsedTime)])
    df = df.drop(columns=['Date', 'starttime', 'elapsedTime'])
    df = df.sort_values(by=['dataType', 'rainStartTime']).reset_index(drop=True)

    return df


def query_DAP_sedi_mass(watershed, flume):

    flume = f'({flume})'
    sediment_str = """select * from SedimentEvents where Flume in """ + flume
    df = sql.read_sql(sediment_str,conn)

    return df

