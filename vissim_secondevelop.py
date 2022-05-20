# -*- coding: utf-8 -*-
# @Author: KeyangZhang
# @Email: 3238051@qq.com
# @Date:   2019-04-07 15:00:44
# @Last Modified by:   KeyangZhang
# @Last Modified time: 2019-04-12 11:17:20

import win32com.client
import os
import time

# dispatch Vissim com
vissim = win32com.client.Dispatch("VISSIM.Vissim.430")

# inpfile Contains all input data for VISSIM’s traffic and transit network 
inppath = os.path.abspath('./vissim_config/signal0409.inp') 
vissim.LoadNet(inppath)


sim = vissim.Simulation
net = vissim.net
vehs = net.Vehicles # vehicles on net per simtime
scs = net.SignalControllers
sc_ramp = scs.GetSignalControllerByNumber(1)

dets = sc_ramp.Detectors

for det in dets:
    print(det.AttValue('ID'),det.AttValue('NAME'))


sim.RandomSeed = 40
sim.RunIndex = 0
sim.Resolution = 1


sc_ramp.SetAttValue('CYCLETIME', 8) # cycle of signal
sc_ramp.SetAttValue('OFFSET', 2) # offset -> green time

print(sim.AttValue('RESOLUTION'))


for i in range(5):
    sim.RunSingleStep()


time.sleep(1)
sim.Stop()
time.sleep(1)
vissim.exit()