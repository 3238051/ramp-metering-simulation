# -*- coding: utf-8 -*-
# @Author: Keyang Zhang
# @E-mail: 3238051@qq.com
# @Date:   2019-03-24 12:11:24
# @Last Modified by:   KeyangZhang
# @Last Modified time: 2019-12-10 16:59:26

import random
from copy import deepcopy
import csv
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

INF = float('inf')


class Signal(object):
    """docstring for Signal"""

    def __init__(self, cycle, greentime, origin=0):
        super(Signal, self).__init__()
        self.cycle = cycle
        self.greentime = greentime
        self.time = origin
        self.status = None

    def update_status(self):
        temp = self.time % self.cycle
        if temp < self.greentime:
            self.time += 1
            self.status = 'green'
        else:
            self.time += 1
            self.status = 'red'
        return self.status

    def __str__(self):
        info = '信号周期:{0},显示绿灯时长:{1}'.format(self.cycle, self.greentime)
        return info


class Vehicle(object):
    """Vehicle"""

    count = 0
    rampNo = None  # 匝道车辆编号
    interstart = None  # 强制换道起点
    interend = None  # 强制换道终点

    def __init__(self, vehnum, lanenum, init_v, veh_length=5, cell_length=5, a=5, max_v=20, slow_p=0.5, safe_d=10):
        '''
        vehnum：车辆编号
        lanenum：初始到达时所在车道编号,传入从1开始，内部储存从0开始
        init_v：初始到达时的速度
        veh_length：车辆长度，默认5m
        cell_length：仿真元胞长度，默认5m
        a：期望加速度，默认5m
        max_v：速度上限，默认20m/s
        slow_p：随机慢化概率，默认0.5
        safe_d：安全换道距离，默认5m
        '''
        super(Vehicle, self).__init__()
        self.init_lnnum = lanenum-1  # 初始到达位置
        self.vehnum = vehnum
        self.length = int(veh_length/cell_length+0.5)  # 车辆占据元胞数
        self.cell_length = cell_length
        self.v = int(init_v/cell_length+0.5)  # 也是这一秒的纵向位移
        self.dx = 0  # 0表示不变，-1向左(向上)，1向右(向下)
        self.a = int(a/cell_length+0.5)  # 期望加速度
        self.slow_p = slow_p  # 随机慢化概率
        self.max_v = int(max_v/cell_length+0.5)  # 速度上限 m/s 用元胞离散
        self.safe_d = int(safe_d/cell_length+0.5)  # 换道安全距离

        # 注意以下距离都是障碍物距离本车车头的间距
        self.front_d = 0
        self.leftfront_d = 0
        self.leftrear_d = 0
        self.rightfront_d = 0
        self.rightrear_d = 0
        self.x = lanenum-1  # 车头所在的元胞横坐标
        self.y = 0  # 车头所在的元胞纵坐标，初始均为0

        Vehicle.count += 1

    def calculate_distance(self, road):
        """根据传入的道路状态表，计算车辆的间距"""
        x0 = self.x
        y0 = self.y
        area_length = road.mainLength  # 仿真区域长度,从0开始
        lannum_range = range(road.mainLanes+road.rampLanes)  # 仿真区域车道数范围

        # 计算前车距
        y = 0
        if y0 <= area_length-1:
            for status in road.cellStatus[x0][y0:]:
                if status:
                    break
                y += 1
        if y >= area_length-y0:
            y = INF
        self.front_d = y+1

        # 计算右前车距
        if x0+1 in lannum_range:
            y = 0
            for status in road.cellStatus[x0+1][y0:]:
                if status:
                    break
                y += 1
            if y > area_length-y0:
                y = INF
            self.rightfront_d = y
        else:
            self.rightfront_d = 0

        # 计算右后车距
        if x0+1 in lannum_range:
            y = 0
            for status in road.cellStatus[x0+1][:y0+1][::-1]:
                if status:
                    break
                y += 1
            self.rightrear_d = y
        else:
            self.rightrear_d = 0

        # 计算左前车距
        if x0-1 in lannum_range:
            y = 0
            for status in road.cellStatus[x0-1][y0:]:
                if status:
                    break
                y += 1
            if y > area_length-y0:
                y = INF
            self.leftfront_d = y
        else:
            self.leftfront_d = 0

        # 计算左后车距
        if x0-1 in lannum_range:
            y = 0
            for status in road.cellStatus[x0-1][:y0+1][::-1]:
                if status:
                    break
                y += 1
            self.leftrear_d = y
        else:
            self.leftrear_d = 0

    def calculate_site(self):
        '''计算车辆更新后的位置'''
        self.x += self.dx
        self.y += self.v

    def calculate_move(self, road):
        '''传入道路连接情况表，计算车车辆的横纵位移'''

        exp_v = min(self.v + self.a, self.max_v)  # 期望速度

        if random.random() < self.slow_p:
            if exp_v-1 >= 0:
                exp_v = exp_v-1
            else:
                exp_v = 0

        x0, y0 = self.x, self.y
        l, r = 0, 0  # 是否会选择左右转

        # 换道条件1--换道动机
        factor1 = (exp_v > self.front_d and self.leftfront_d > self.front_d,
                   exp_v > self.front_d and self.rightfront_d > self.front_d)
        # 换道条件2--安全条件
        factor2 = (self.leftrear_d > self.safe_d,
                   self.rightrear_d > self.safe_d)
        # 换道条件3--换道概率
        pl = road.leftTurnPro[x0][y0]
        pr = road.rightTurnPro[x0][y0]
        factor3 = (random.random() < pl, random.random() < pr)

        l = factor1[0] and factor2[0] and factor3[0]
        r = factor1[1] and factor2[1] and factor3[1]
        if l and not r:
            self.dx = -1
            self.v = min(exp_v, self.leftfront_d)
        elif not l and r:
            self.dx = 1
            self.v = min(exp_v, self.rightfront_d)
        elif not l and not r:
            self.dx = 0
            self.v = min(exp_v, self.front_d-1)
        else:
            if self.rightfront_d > self.leftfront_d:
                self.dx = 1
                self.v = min(exp_v, self.rightfront_d)
            else:
                self.dx = -1
                self.v = min(exp_v, self.leftfront_d)

        # 合流区强制换道
        f1 = x0 in self.rampNo
        f2 = self.interstart+5 <= y0 <= self.interend
        if f1 and f2 and factor2[0]:
            self.dx = -1
            self.v = min(1, self.leftfront_d)

    def incellspace(self, road):
        area_index = road.mainLength-1
        return self.y < area_index

    def info(self):
        return [self.init_lnnum, self.vehnum, self.v*self.cell_length, self.x, self.y*self.cell_length]


class RoadSection(object):
    def __init__(self, cellLength=5, mainLanes=3, mainLength=400, rampLanes=2, rampLength=220.5, interLength=128):
        super(RoadSection, self).__init__()
        self.cellLength = cellLength           # 元胞长度
        self.mainLanes = int(mainLanes+0.5)    # 主路车道数
        self.mainLength = int(mainLength/self.cellLength +
                              0.5)    # 主路长度（离散化，单位为元胞）
        self.rampLanes = int(rampLanes+0.5)                    # 匝道车道数
        self.rampLength = int(
            rampLength/self.cellLength+0.5)  # 匝道长度（离散化，单位为元胞）
        self.interLength = int(
            interLength/self.cellLength+0.5)  # 交汇区域长度（离散化，单位为元胞）

        self.mainNo = range(mainLanes)
        self.rampNo = range(mainLanes, mainLanes+rampLanes)
        self.cellSpace = []      # 元胞空间
        self.cellStatus = []     # 元胞空间状态
        self.leftTurnPro = []    # 左转概率矩阵
        self.rightTurnPro = []   # 右转概率矩阵

        # 构造cellSpace
        for i in range(self.mainLanes):
            lane = []
            for j in range(self.mainLength):
                lane.append(0)
            self.cellSpace.append(lane)

        for i in range(self.rampLanes):
            lane = []
            if(i == 0):
                for j in range(self.rampLength):
                    lane.append(0)
                for k in range(self.mainLength-self.rampLength):
                    lane.append(-1)
            else:
                tpl = int((self.interLength/2)+0.5)
                for j in range(self.rampLength-tpl):
                    lane.append(0)
                for k in range(self.mainLength-(self.rampLength-tpl)):
                    lane.append(-1)
            self.cellSpace.append(lane)

        # 初始化cellStatus
        self.cellStatus = self.cellSpace.copy()

    def createLeftTurnPro(self, pTramp, pMain1, pMain2, pMain3):
        '''构造LeftTurnPro，pTramp交织区左转概率，pMain1, pMain2, pMain3分别是主路三段的概率'''
        lane = []
        for j in range(self.mainLength):
            lane.append(0)
        self.leftTurnPro.append(lane)

        for i in range(self.mainLanes-1):
            lane = []
            for j in range(self.rampLength-self.interLength):
                lane.append(pMain1)
            for j in range(self.interLength):
                lane.append(pMain2)
            for j in range(self.mainLength-self.rampLength):
                lane.append(pMain3)
            self.leftTurnPro.append(lane)

        lane = []
        for j in range(self.rampLength-self.interLength):
            lane.append(0)
        for j in range(self.interLength):
            lane.append(1)
        for j in range(self.mainLength-self.rampLength):
            lane.append(0)  # None
        self.leftTurnPro.append(lane)

        tpl = int((self.interLength / 2) + 0.5)
        for i in range(self.rampLanes-1):
            lane = []
            for j in range(self.rampLength-self.interLength):
                lane.append(pTramp)
            for j in range(tpl):
                lane.append(1)
            for j in range(self.mainLength-tpl-(self.rampLength-self.interLength)):
                lane.append(0)  # None
            self.leftTurnPro.append(lane)

    def createRightTurnPro(self, pMain1, pMain2, pMain3):
        '''构造RightTurnPro，pMain1, pMain2, pMain3分别是主路三段的概率'''
        for i in range(self.mainLanes-1):
            lane = []
            for j in range(self.rampLength - self.interLength):
                lane.append(pMain1)
            for j in range(self.interLength):
                lane.append(pMain2)
            for j in range(self.mainLength - self.rampLength):
                lane.append(pMain3)
            self.rightTurnPro.append(lane)

        lane = []
        for j in range(self.mainLength):
            lane.append(0)
        self.rightTurnPro.append(lane)

        lane = []
        for j in range(self.rampLength):
            lane.append(0)
        for j in range(self.mainLength-self.rampLength):
            lane.append(0)  # None
        self.rightTurnPro.append(lane)

        tpl = int((self.interLength / 2) + 0.5)
        for i in range(self.rampLanes-1):
            lane = []
            for j in range(self.rampLength-tpl):
                lane.append(0)
            for j in range(self.mainLength-(self.rampLength-tpl)):
                lane.append(0)  # None
            self.rightTurnPro.append(lane)


class Simulator(object):
    """simulator of simulation for waving section"""
    baisc_para = {'cell_length': '元胞长度',
                  'left_lnchgpro': '左转换道概率组合',
                  'right_lnchgpro': '右转换道概率组合',
                  'acceleration': '车辆加速度',
                  'max_speed': '车辆限速',
                  'safe_d': '换道安全车距',
                  'slow_p': '随机慢化概率',
                  'dmdfile': '实测需求文件'}

    def __init__(self):
        super(Simulator, self).__init__()

        # 静态参数
        self.cell_length = None
        self.left_lnchgpro = None
        self.right_lnchgpro = None
        self.acceleration = None  # 最大加速度m/s
        self.max_speed = None  # 最大速度m/s
        self.safe_d = None  # 换道安全车距m
        self.slow_p = None
        self.dmdfile = None

        # 动态参数
        self.road = None
        self.signal = None  # 信号方案
        self.sigsite = None  # 信号灯架设位置
        self.demand = None
        self.vehicles = []
        self.time = 0
        self.prdmdfile = None  # 预测需求文件路径(采用相对路径)

    def set_signal(self, signal, site=80):
        '''设置信号方案以及信号灯位置'''
        self.sigsite = int(site/self.cell_length+0.5)
        self.signal = signal

    def check_paraset(self):
        '''检查仿真所需要的必要参数是否都已设置'''
        for key in self.baisc_para:
            if eval('self.{0}'.format(key)) is None:
                raise ValueError('{0}未设置！'.format(self.baisc_para[key]))
        if self.road is None:
            raise ValueError('道路未创建！')

    def create_road(self, road=None):
        '''create or update simulator.road'''
        if road is None:
            road = RoadSection()
            road.createLeftTurnPro(*self.left_lnchgpro)
            road.createRightTurnPro(*self.right_lnchgpro)

        self.road = road

    def get_demand(self, file):
        '''根据文件名读入需求'''
        with open(file, 'r') as f:
            reader = csv.reader(f)
            demand = [list(map(eval, row)) for row in reader]
        self.demand = demand

    def update_vehilces(self):
        '''更新self.time时刻的车列表'''

        # 读取当前时刻的需求
        if self.time > len(self.demand)-1:
            dt = []
        else:
            dt = self.demand[self.time]

        # 添加当前时刻需求到车列表
        for vehinfo in dt:
            vehnum = eval(vehinfo['number'])
            init_v = eval(vehinfo['speed'])
            lanenum = eval(vehinfo['lane'])
            veh = Vehicle(vehnum, lanenum, init_v,
                          veh_length=5,
                          cell_length=self.cell_length,
                          a=self.acceleration,
                          max_v=self.max_speed,
                          slow_p=self.slow_p,
                          safe_d=self.safe_d)
            self.vehicles.append(veh)

        # 更新车列表
        for i, veh in enumerate(self.vehicles):
            veh.calculate_distance(self.road)
            veh.calculate_move(self.road)
            veh.calculate_site()
            if not veh.incellspace(self.road):
                del self.vehicles[i]

    def update_roadstatus(self):
        '''更新道路元胞状态'''
        self.road.cellStatus = deepcopy(
            self.road.cellSpace)  # reset the status
        for veh in self.vehicles:
            x0, y0 = veh.x, veh.y
            y = y0-veh.length
            if y >= 0:
                self.road.cellStatus[x0][y:y0] = [veh.vehnum]*(y0-y)
            else:
                self.road.cellStatus[x0][0:y0] = [veh.vehnum]*(y0-0)

    def update_signal(self):
        '''根据信号信息设置道路通行权'''
        status = self.signal.update_status()
        site = self.sigsite
        rampNo = self.road.rampNo  # 匝道编号
        if status == 'green':
            for i in rampNo:
                #self.road.cellSpace[i][site] = 0
                self.road.cellStatus[i][site] = 0
        else:
            for i in rampNo:
                #self.road.cellSpace[i][site] = -2
                self.road.cellStatus[i][site] = -2

    def record(self):
        records = []
        if self.signal is None:
            sigstatus = 'None'
        else:
            sigstatus = self.signal.status
        for veh in self.vehicles:
            rec = [self.time]+veh.info()+[sigstatus]
            records.append(rec)
        return records

    def simulate(self, simtime, dmdfile=None):
        '''
        simtime: 仿真时长
        dmdfile: 仿真所需的到达数据文件
        '''

        #   判断是否是仅仅进行仿真的情况
        if dmdfile is None:
            dmdfile = self.dmdfile

        self.vehicles = []  # 避免多次仿真车道上还停留上次仿真的残留车辆
        Vehicle.count = 0  # 通过类中的变量统计总共创建的车辆数

        # 通过类变量 控制车辆在合流区域的换道行为
        Vehicle.rampNo = self.road.rampNo
        Vehicle.interstart = self.road.rampLength-self.road.interLength
        Vehicle.interend = self.road.rampLength

        self.check_paraset()  # 检查必要的参数是否设置完成
        self.get_demand(dmdfile)  # 获取仿真所需要的需求

        with open('record.csv', 'w', newline='') as f:

            recwriter = csv.writer(f)
            recwriter.writerow(
                ['time', 'init_lnnum', 'vehnum', 'speed', 'x', 'y', 'signal'])

            # 开始迭代仿真
            for t in range(simtime):
                self.time = t
                if self.signal is not None:
                    self.update_signal()
                self.update_vehilces()
                self.update_roadstatus()

                # 记录仿真数据
                records = self.record()
                recwriter.writerows(records)

    def report(self):
        '''生成报表'''

        main_length = self.road.mainLength*self.cell_length
        ramp_length = self.road.rampLength*self.cell_length
        rampNo = self.road.rampNo
        mainNo = self.road.mainNo
        with open('record.csv', 'r') as f:
            reader = csv.DictReader(f)
            recs = [[row['time'], row['speed'], row['x'], row['y']]
                    for row in reader]

        for i, rec in enumerate(recs):
            recs[i] = list(map(eval, rec))

        data = []
        for i in range(recs[-1][0]+1):
            tdata = []
            for rec in recs:
                if rec[0] == i:
                    tdata.append(rec)
            data.append(tdata)

        info = []
        for t, tdata in enumerate(data):
            sumnum = len(tdata)
            sumspeed = 0
            mainnum = 0
            rampnum = 0
            mainsumspeed = 0
            rampsumspeed = 0
            for vehinfo in tdata:
                sumspeed += vehinfo[1]
                if vehinfo[2] in rampNo:
                    rampnum += 1
                    rampsumspeed += vehinfo[1]
                else:
                    mainnum += 1
                    mainsumspeed += vehinfo[1]

            if sumnum == 0:
                avespeed = 0
            else:
                avespeed = round(sumspeed/sumnum*3.6, 2)

            if mainnum == 0:
                mainavespeed = 0
            else:
                mainavespeed = round(mainsumspeed/mainnum*3.6, 2)

            if rampnum == 0:
                rampavespeed = 0
            else:
                rampavespeed = round(rampsumspeed/rampnum*3.6, 2)

            maindensity = round(mainnum/main_length*1000, 2)
            rampdensity = round(rampnum/ramp_length*1000, 2)
            info.append([t, sumnum, avespeed, maindensity,
                         mainavespeed, rampdensity, rampavespeed])

        with open('report.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['时间', '交通量', '平均车速(km/h)', '主线密度(veh/km)',
                             '主线平均车速(km/h)', '进口道密度(veh/km)', '进口道平均车速(km/h)'])
            writer.writerows(info)

    def demand_forcast(self, t):
        '''需求预测
            t: 预测时长(s)
        '''
        if self.dmdfile is None:
            raise ValueError('请先设置dmdfile(实测需求文件)')

        with open(self.dmdfile, 'r') as f:
            reader = csv.reader(f)
            dmd = [row for row in reader]

        l = len(dmd)

        prdmd = dmd[-60:]*int(t/60+0.5)

        with open('predicted_demand.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(prdmd)

        self.prdmdfile = 'predicted_demand.csv'

        return self.prdmdfile

    def optimize_signal(self, optcyc, cul=8):
        '''
        返回最优信号方案和对应的滞留车辆数以及平均速度
        optcyc:优化周期
        cul:控制信号周期的上限，默认下限为2
        '''

        def standarize(l):
            new = []
            zd = max(l)
            zx = min(l)
            if zd == zx:
                return [0]*len(l)
            for i in l:
                new.append((i-zx)/(zd-zx))
            return new

        prdmdfile = self.demand_forcast(optcyc)

        signals = [Signal(c, 2) for c in range(2, cul+1)]
        throughs = []  # 每个信号方案的通过的车辆数
        avespeeds = []  # 每个信号方案的主线平均行程速度
        for sig in signals:
            self.set_signal(sig)
            self.simulate(optcyc, prdmdfile)
            with open('record.csv', 'r') as f:
                reader = csv.DictReader(f)
                mainNo = self.road.mainNo
                speed = [eval(veh['speed'])
                         for veh in reader if eval(veh['x']) in mainNo]
            avespeed = sum(speed)/len(speed)
            throughs.append(Vehicle.count-len(self.vehicles))
            avespeeds.append(avespeed)

        # 两个参数0-1标准化

        temp1 = standarize(throughs)
        temp2 = standarize(avespeeds)

        overall = []
        for a, b in zip(temp1, temp2):
            overall.append(a+b)
        i = overall.index(max(overall))

        self.signal = signals[i]

        with open('OptimizationResult.txt', 'w') as f:
            for sig, thru, avev, score in zip(signals, throughs, avespeeds, overall):
                f.writelines([str(sig), '\n通过车辆：', str(thru),
                              '\n平均行程速度:', str(avev), '\n综合得分:', str(score), '\n'])
                f.write('\n')

            f.write('最优方案\n')
            f.writelines([str(signals[i]), '\n通过车辆：', str(throughs[i]),
                          '\n平均行程速度:', str(avespeeds[i]),
                          '\n综合得分:', str(overall[i]), '\n'])

        return signals[i], throughs[i], avespeeds[i]*3.6

    def visualize(self):
        '''可视化展示'''

        def update(frame):
            '''动画更新函数'''

            (x, y_main, y_ramp), sca_dataset_main, sca_dataset_ramp = frame

            ln_xdata.append(x)
            ln_ydata_main.append(y_main)
            ln_ydata_ramp.append(y_ramp)
            # 更新密度曲线
            ln_main.set_data(ln_xdata, ln_ydata_main)
            ln_ramp.set_data(ln_xdata, ln_ydata_ramp)

            sig.update_status()
            # 在元胞图上现实信号
            if self.signal is not None:
                axe_celspa.axvline(self.sigsite*self.cell_length,
                                   1-(max(mainNo)+1)/(main_lanes+ramp_lanes),
                                   1-(max(rampNo)+1)/(main_lanes+ramp_lanes),
                                   color=sig.status)
            # 元胞图
            if sca_dataset_ramp != []:
                sca_ramp.set_offsets(sca_dataset_ramp)
            if sca_dataset_main != []:
                sca_main.set_offsets(sca_dataset_main)

            # 在密度曲线下方同步现实信号状态
            if x != 0:
                xmin = ln_xdata[-2]/XLIM
                xmax = x/XLIM
                axe_descur.axhline(
                    0, xmin, xmax, linewidth=10, color=sig.status)

            return ln_main, ln_ramp, sca_main, sca_ramp

        # 因为可视化需要用到生成报表的数据，所以先检测有无生成报表

        # 可视化所需要的参数信息
        main_length = self.road.mainLength*self.cell_length
        ramp_length = self.road.rampLength*self.cell_length
        inter_length = self.road.interLength*self.cell_length
        main_lanes = self.road.mainLanes
        ramp_lanes = self.road.rampLanes
        mainNo = self.road.mainNo
        rampNo = self.road.rampNo
        if self.signal is not None:
            sig = Signal(self.signal.cycle, 2)  # 按照最终的信号相位从0开始
        else:
            sig = Signal(2, 2)

        # 创建画布，添加元胞空间，密度曲线两个子图
        fig, axes = plt.subplots(2, 1)
        axe_celspa, axe_descur = axes
        plt.subplots_adjust(hspace=0.5)

        # 设置 元胞空间 背景
        axe_celspa.set_ylabel('Lane')
        axe_celspa.set_title('Realtime Simulation')
        axe_celspa.set_xlim(0, main_length)
        axe_celspa.set_ylim(main_lanes + ramp_lanes+0.5, -1.5)  # y轴坐标反序

        #   设置路缘带
        axe_celspa.axhline(y=-1, linewidth=16, color='darkgrey')
        axe_celspa.axhline(y=main_lanes+ramp_lanes,
                           linewidth=16, color='darkgrey')

        #   设置主线背景
        for n in mainNo:
            axe_celspa.axhline(n, color='steelblue', linewidth=16, alpha=0.3)
        #   设置匝道
        for n in rampNo:
            if n == main_lanes:
                # 靠近主线的匝道
                p = ramp_length/main_length
                axe_celspa.axhline(n, xmax=p+0.05, linewidth=16,
                                   color='darkgoldenrod', alpha=0.3)
                axe_celspa.axhline(n, xmin=p+0.05, linewidth=16,
                                   color='darkgrey')
            else:
                # 远离主线的匝道（如果有的话执行该语块）
                p = (ramp_length-int(inter_length/2+0.5))/main_length
                axe_celspa.axhline(n, xmax=p+0.05, linewidth=16,
                                   color='darkgoldenrod', alpha=0.3)
                axe_celspa.axhline(n, xmin=p+0.05, linewidth=16,
                                   color='darkgrey')
        #   设置主线与匝道之间的隔离带
        p = (ramp_length-inter_length)/main_length
        axe_celspa.axhline(y=main_lanes-0.5, xmax=p,
                           linewidth=4, color='darkgrey')

        #   散点图 主线车是蓝色，匝道是红色
        sca_ramp = axe_celspa.scatter([], [], marker='s', s=30, c='orange')
        sca_main = axe_celspa.scatter([], [], marker='s', s=30, c='blue')

        # 设置密度曲线背景

        YLIM = 120  # 最大密度veh/km
        XLIM = self.time  # 周期时长 s
        axe_descur.set_xlim(0, XLIM)
        axe_descur.set_ylim(0, YLIM)
        axe_descur.set_title('Simu-Time Density')
        axe_descur.set_ylabel('Density-veh/km')
        axe_descur.set_xlabel('Time-sec')

        axe_descur.axhline(y=YLIM * 0.6, linewidth=1, color='firebrick')
        axe_descur.axhline(y=YLIM * 0.4, linewidth=1, color='gold')
        axe_descur.axhline(y=YLIM * 0.2, linewidth=1, color='lime')

        ln_main, = axe_descur.plot([], [], color='orange', label='Main Lane')
        ln_ramp, = axe_descur.plot([], [], color='blue', label='Entry Ramp')
        axe_descur.legend()

        # 提取密度曲线图所需的数据
        with open('report.csv', 'r') as f:
            reader = csv.DictReader(f)
            des_data = []
            for row in reader:
                t = eval(row['时间'])
                ms = eval(row['主线密度(veh/km)'])
                rs = eval(row['进口道密度(veh/km)'])
                des_data.append([t, ms, rs])

        ln_xdata, ln_ydata_main, ln_ydata_ramp = [], [], []

        # 提取密度曲线所需要的数据
        with open('record.csv', 'r') as f:
            reader = csv.DictReader(f)
            recs = []
            simt = 0
            for row in reader:
                t = eval(row['time'])
                init_x = eval(row['init_lnnum'])
                x = eval(row['x'])
                y = eval(row['y'])
                recs.append([t, init_x, x, y])
                simt = t
            cell_data_main = {key: [] for key in range(simt+1)}
            cell_data_ramp = {key: [] for key in range(simt+1)}
            for t, init_x, x, y in recs:
                if init_x in mainNo:
                    cell_data_main[t].append([y, x])
                else:
                    cell_data_ramp[t].append([y, x])

        data = zip(des_data, cell_data_main.values(), cell_data_ramp.values())

        ani = FuncAnimation(fig, update, frames=data,
                            blit=False, interval=500)

        # ani.save('仿真预测过程.gif',writer="pillow")

        plt.show()

    def GAcalibrate(self, popN=5, genN=10, CXPB=0.5, MUTPB=0.2):
        '''fake GA'''
        time.sleep(3)
        return 0


def main():
    '''主函数'''
    start = time.time()

    sim = Simulator()

    #  设置参数，创建道路，否则仿真不能开始
    sim.cell_length = 5
    sim.left_lnchgpro = (0.9, 0.8, 0.8, 0.8)
    sim.right_lnchgpro = (0.8, 0.8, 0.8)
    sim.acceleration = 10  # 车辆加速度m/s
    sim.max_speed = 20  # 最大速度m/s
    sim.safe_d = 10  # 换道安全车距m
    sim.slow_p = 0.5
    sim.dmdfile = 'vissim_demand.csv'

    sim.create_road()

    # # 仿真
    # sim.simulate(120)

    # 信号优化
    sim.optimize_signal(60, 10)

    # 生成报表
    sim.report()

    # 可视化
    sim.visualize()

    print('time using:', time.time()-start)
    time.sleep(5)


if __name__ == '__main__':
    main()
