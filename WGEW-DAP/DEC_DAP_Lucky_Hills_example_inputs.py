"""
This script gather requried inputs for script EDC_DAP.py

    - Please test different thresholds to create meaningful results
    - If number of gauges is greater than 3, hyetographs will look busy.
      Recommend using the 2 panel (with accu. rainfall and runoff rate). Also try line without marker.

Copyright 2020  Haiyan Wei
haiyan.wei@usda.gov

"""

from EDC_DAP_main_fs import EDC_DAP

out = 'July25'
flume = 102

watershed = 63
gauges = [83]
rainHr = 24
rainTooLow = 0.01*25.4
runoffTooLow = 0.003

rainSepHr = 6
runoffAggHr = 6
runTooLateHr = 6


EDC_DAP(watershed, flume, gauges, out,
        runoffAggHr, rainSepHr, rainHr,
        runoffTooLow, rainTooLow, runTooLateHr)

