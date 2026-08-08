# 假定该代码只会运行一次，而且在运行之前设置了GUR
# 比如在调用之前 sys.argv.append('some_arg')
import sys
# import matplotlib.pyplot as plt
import numpy as np
import pickle
import re
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory
import copy
from cplex_solu import *
from tqdm import tqdm
import logging
pyomo_logger = logging.getLogger('pyomo')
pyomo_logger.setLevel(logging.ERROR)

datadir = "./"
modeldir = datadir
# 设置参数
# fpattern = re.compile(r'GUR[\s]*=([\d]+[.]?[\d]*)')
# for j in sys.argv:
#     match = fpattern.search(j)
#     if match != None:
#         GUR = float(match.group(1))
# if GUR == None:
#     exit("error, must setting GUR")
# 如果在使用玩gur后，要去掉该参数，下面是如果gur对应参数在最后一条的表现
# sys.argy.pop()

class Flimit:
    def __init__(self, llo, lup, vlo, vup, ri1, ri2, ri3, ri4):
        self.ri1 = ri1
        self.ri2 = ri2
        self.ri3 = ri3
        self.ri4 = ri4
        # self.ri1 = ri1 & ri2 & ri3 & ri4 # 对cond1()
        # self.ri2 = self.ri1 # 对应cond2()
        # self.ri3 = ri1 & ri2 & ri3 # 对应cond3()
        # self.ri4 = ri1 & ri2 & ri4 # 对应cond4()
        self.llo = llo
        self.lup = lup
        self.vlo = vlo
        self.vup = vup
        self.con1= (not (ri1 & ri2)) | (llo != lup)
        self.con2= (not (ri3 & ri4)) | (vlo <= vup)
        self.con3= (not (ri3)) | (vlo != 0)
        self.con4= (not (ri4)) | (vup != 0)
        self.con5= (not (ri1 & ri3)) | (llo <= vlo)
        self.con6= (not (ri2 & ri4)) | (lup >= vup)
        self.loout = ri1 # 当为True的时候，表明可以输出llo
        self.upout = ri2 # 当为True的时候，可以输出lup
    def conAssign(self, confun, value, name):
        if confun(self):
            if name == 'llo':
                self.llo = value
                self.ri1 = True
                self.con1= self.llo != self.lup # for that the condition value will change after the llo changed
                self.con5= self.llo <= self.vlo
                self.loout = True # 当完成对llo赋值之后，一定能输出
            elif name == 'lup':
                self.lup = value
                self.ri2 = True
                self.con1= self.llo != self.lup
                self.con6= self.lup >= self.vup
                self.upout = True # 当完成对于lup的赋值之后，一定能输出
            elif name == 'vlo': # vlo and vup value change will not happened. such that not add the change of con?
                self.vlo = value
            elif name == 'vup':
                self.vup = value
            else:
                print("error, illeagy character")


class SoluA:
    def __init__(self, index, obj0, soluDict0, obj1, soluDict1, state, termin):
        self.index = index
        self.obj0 = obj0
        self.soluDict0 = soluDict0
        self.obj1 = obj1
        self.soluDict1 = soluDict1
        self.state = state
        self.termin = termin

class MyIter:
    def __init__(self, listA, func):
        self.listA = listA
        self.index = 0
        self.lengt = len(listA)
        self.funct = func

    def __iter__(self):
        return self

    def __next__(self):
        if self.index > self.lengt:
            raise StopIteration
        result = self.funct(self.listA[self.index])
        self.index +=  1
        return result

with open(datadir + "data.pkl", "rb") as f:
    data = pickle.load(f)

rxns, rxn, mets, met, comps, comp, grps, grp, exchanges, sndlaw, gav, bndID, bndI, consdrGerr, consNS, nsIDs, nsID, blocked, lncL, vlncL, vgLimits, V_vgLimits, drGLimits, V_drGLimits, dfGgLimits, V_dfGgLimits, drGgLimits, V_drGgLimits, dfGLimits, sigmaL, vsigmaL, vgexg, ratio, dfG0, S, Sg, drGt0tr, dfGt0, drGSE, K, var_drG, var_drGg, var_lnc, drGerrMax, drGerrMin = data

# 预设变量？
def make(scond):
    with open(datadir + f"data_{scond}.pkl", "rb") as f:
        data = pickle.load(f)

    rxns, rxn, mets, met, comps, comp, grps, grp, exchanges, sndlaw, gav, bndID, bndI, consdrGerr, consNS, nsIDs, nsID, blocked, lncL, vlncL, vgLimits, V_vgLimits, drGLimits, V_drGLimits, dfGgLimits, V_dfGgLimits, drGgLimits, V_drGgLimits, dfGLimits, sigmaL, vsigmaL, vgexg, ratio, dfG0, S, Sg, drGt0tr, dfGt0, drGSE, K, var_drG, var_drGg, var_lnc, drGerrMax, drGerrMin = data

    gur = 500
    Temp = 310.15
    GasCons = 0.008314
    epsilon = 0.5
    nan = float('nan')

    # 对于下述的transExcelToArr而设置的read函数，只针对1d
    def read_Arr(fA, index):
        if fA[1][index]:
            return fA[0][index]
        else:
            return nan

    # 给定两个字符序列A,B，假定是1d, 确定是否B包含在A里面，返回两个list，一个是是否在，另一个是位置，位置为负表示不存在
    def isExist(listA, listB):
        aDict = {}
        i = 0
        for a in listA:
            aDict[a] = i
            i = i + 1
        A = np.zeros(len(listB), dtype=bool)
        B = np.zeros(len(listB), dtype=int)
        j = 0
        for b in listB:
            if b in aDict:
                A[j] = True
                B[j] = aDict[b]
            else:
                A[j] = False
                B[j] = -1
            j = j + 1
        return (A, B)

    ratioH = copy.deepcopy(ratio[0])
    ratioH[np.logical_not(ratio[1])] = 0

    if bndID[0] == 'lo':
        lo = 0
        up = 1
    else:
        lo = 1
        up = 0

    lncF = []
    for i in range(met):
        row = []
        for j in range(comp):
            obj = Flimit(lncL[0][i, j, lo], lncL[0][i, j, up], vlncL[0][i, j, lo], vlncL[0][i, j, up],
                         lncL[1][i, j, lo], lncL[1][i, j, up], vlncL[1][i, j, lo], vlncL[1][i, j, up])
            row.append(obj)
        lncF.append(row)

    def toFlimit1d(lim, vlim, leng):
        resu = []
        for i in range(leng):
            resu.append(
                Flimit(lim[0][i, lo], lim[0][i, up], vlim[0][i, lo], vlim[0][i, up], lim[1][i, lo], lim[1][i, up],
                       vlim[1][i, lo], vlim[1][i, up]))
        return resu

    vglF = toFlimit1d(vgLimits, V_vgLimits, grp)
    drGF = toFlimit1d(drGLimits, V_drGLimits, rxn)
    drgF = toFlimit1d(drGgLimits, V_drGgLimits, grp)
    dfGF = toFlimit1d(dfGgLimits, V_dfGgLimits, grp)

    def cond1(flim):
        return flim.con1 and flim.con3 and flim.con5 and flim.con2

    def cond2(flim):
        return flim.con1 and flim.con4 and flim.con6 and flim.con2

    def trans(objs):
        for obj in objs:
            obj.conAssign(lambda x: cond1(x) & obj.ri3, obj.vlo, 'llo')
            # for obj in objs: # for the llo and lup changed will not affect each other
            obj.conAssign(lambda x: cond2(x) & obj.ri4, obj.vup, 'lup')

    for obj in lncF:
        trans(obj)  # for that llo and lup will not affect each other

    trans(vglF)
    trans(drGF)
    trans(drgF)

    ## 重写，因为这里对于一些这表有假设，就是额rateSet的第二指标
    rateSet = [(lo, rxns.index('EX_fru')), (up, rxns.index('EX_fru')),
               (lo, rxns.index('EX_gal')), (up, rxns.index('EX_gal')),
               (lo, rxns.index('EX_glcn')), (up, rxns.index('EX_glcn')),
               (lo, rxns.index('EX_glyc')), (up, rxns.index('EX_glyc')),
               (up, rxns.index('EX_ac')),
               (lo, rxns.index('EX_pyr')), (up, rxns.index('EX_pyr')),
               (lo, rxns.index('EX_succ')), (lo, rxns.index('EX_etoh')), (lo, rxns.index('EX_for')),
               (lo, rxns.index('EX_glc')), (up, rxns.index('EX_glc')),
               (lo, rxns.index('EX_co2')),
               (lo, rxns.index('EX_ala-L')), (up, rxns.index('EX_ala-L')),
               (lo, rxns.index('EX_arg-L')), (up, rxns.index('EX_arg-L')),
               (lo, rxns.index('EX_asn-L')), (up, rxns.index('EX_asn-L')),
               (lo, rxns.index('EX_asp-L')), (up, rxns.index('EX_asp-L')),
               (lo, rxns.index('EX_cys-L')), (up, rxns.index('EX_cys-L')),
               (lo, rxns.index('EX_gln-L')), (up, rxns.index('EX_gln-L')),
               (lo, rxns.index('EX_glu-L')), (up, rxns.index('EX_glu-L')),
               (lo, rxns.index('EX_gly')), (up, rxns.index('EX_gly')),
               (lo, rxns.index('EX_his-L')), (up, rxns.index('EX_his-L')),
               (lo, rxns.index('EX_ile-L')), (up, rxns.index('EX_ile-L')),
               (lo, rxns.index('EX_leu-L')), (up, rxns.index('EX_leu-L')),
               (lo, rxns.index('EX_lys-L')), (up, rxns.index('EX_lys-L')),
               (lo, rxns.index('EX_met-L')), (up, rxns.index('EX_met-L')),
               (lo, rxns.index('EX_phe-L')), (up, rxns.index('EX_phe-L')),
               (lo, rxns.index('EX_pro-L')), (up, rxns.index('EX_pro-L')),
               (lo, rxns.index('EX_ser-L')), (up, rxns.index('EX_ser-L')),
               (lo, rxns.index('EX_thr-L')), (up, rxns.index('EX_thr-L')),
               (lo, rxns.index('EX_trp-L')), (up, rxns.index('EX_trp-L')),
               (lo, rxns.index('EX_tyr-L')), (up, rxns.index('EX_tyr-L')),
               (lo, rxns.index('EX_val-L')), (up, rxns.index('EX_val-L'))]
    right = np.zeros((grp, bndI), dtype='bool')
    for (aa, bb) in rateSet:
        right[:, aa] = right[:, aa] | (ratioH[:, bb] != 0)

    exglc = rxns.index(scond)
    if (scond == 'EX_ac') or (scond == 'EX_pyr'):
        for i in range(grp):
        # 之前写的是
        # vglF[i].conAssign(lambda _: vglF[i].ri1 & right[i,lo], 0, 'llo')
        # 之所以不这么写的原因是，ri1为false但是vgLimits可以原本就有lo的值，由此可以进行赋值。对于lup同理
           vglF[i].conAssign(lambda _: right[i, lo], 0, 'llo')
           vglF[i].conAssign(lambda _: right[i, up], 0, 'lup')
           vglF[i].conAssign(lambda _: ratioH[i, exglc] == -1, gur, 'lup')

    else:
        for i in range(grp):
        # 之前写的是
        # vglF[i].conAssign(lambda _: vglF[i].ri1 & right[i,lo], 0, 'llo')
        # 之所以不这么写的原因是，ri1为false但是vgLimits可以原本就有lo的值，由此可以进行赋值。对于lup同理
           vglF[i].conAssign(lambda _: right[i, lo], 0, 'llo')
           vglF[i].conAssign(lambda _: right[i, up], 0, 'lup')

           vglF[i].conAssign(lambda _: ratioH[i, exglc] == 1, -gur, 'llo')

    ## now code for tcbm_v2bnd.inc
    sigmalo = read_Arr(sigmaL, lo)
    sigmaup = 15.9

    # 对于所有没有读取的内容在 **lo **up的变量中一律定义为nan
    # tolo toup 均假定输入的是一个1d Flimit list
    # 第二个参数，是当不能输出时，在该位置增补的量
    def toLoUp(x, fill=nan):
        len0 = len(x)
        lowb = np.zeros(len0)
        upbo = np.zeros(len0)
        for i in range(len0):
            if x[i].loout:
                lowb[i] = x[i].llo
            else:
                lowb[i] = fill
            if x[i].upout:
                upbo[i] = x[i].lup
            else:
                upbo[i] = fill
        return (lowb, upbo)

    def toLoUp2(x, fill=nan):
        len0 = len(x)
        len1 = len(x[0])
        lowb = np.zeros((len0, len1))
        upbo = np.zeros((len0, len1))
        for i in range(len0):
            for j in range(len1):
                inter = x[i][j]
                if inter.loout:
                    lowb[i, j] = inter.llo
                else:
                    lowb[i, j] = fill
                if inter.upout:
                    upbo[i, j] = inter.lup
                else:
                    upbo[i, j] = fill
        return (lowb, upbo)

    vglo, vgup = toLoUp(vglF)
    vgI = list(range(grp))
    vlo = np.zeros(rxn)
    vup = np.zeros(rxn)
    for i in range(rxn):
        vglF1 = copy.deepcopy(vglF)
        for j in range(grp):
            vglf = vglF1[j]
            if ratioH[j, i] > 0:
                vglf.conAssign(lambda _: vglf.loout & vglf.ri1, vglf.llo * ratioH[j, i], 'llo')
                vglf.conAssign(lambda _: vglf.upout & vglf.ri2, vglf.lup * ratioH[j, i], 'lup')
            elif ratioH[j, i] < 0:
                vglf.conAssign(lambda _: vglf.loout & vglf.ri2, vglf.lup * ratioH[j, i], 'llo')
                vglf.conAssign(lambda _: vglf.upout & vglf.ri1, vglf.llo * ratioH[j, i], 'lup')
        inlo, inup = toLoUp(vglF1, 0)
        vlo[i] = sum(inlo) - 1e-4
        vup[i] = sum(inup) + 1e-4

    vI = list(range(rxn))
    drGlo, drGup = toLoUp(drGF)
    drGglo, drGgup = toLoUp(drgF)
    drGgI = list(range(grp))
    dfGglo, dfGgup = toLoUp(dfGF)
    dfGgI = list(range(grp))
    dfGlo = dfGLimits[0][:, lo]
    for i in range(grp):  # 利用第二个right的信息，将不存在的地方规整为nan
        if not dfGLimits[1][i, lo]:
            dfGlo[i] = nan

    dfGup = dfGLimits[0][:, up]
    for i in range(grp):
        if not dfGLimits[1][i, up]:
            dfGup[i] = nan

    dfGI = list(range(rxn))

    lnclo, lncup = toLoUp2(lncF)

    # sigmagI = list(range(grp))
    sigmag = np.zeros((grp, bndI))
    sigmaLup = read_Arr(sigmaL, up)
    sigmag[:, up] = np.ones(grp) * sigmaLup
    asigmag = np.zeros(grp, bool)
    # 计算rxns在sndlaw中的情况
    sndrxns, _ = isExist(sndlaw, rxns)
    gavrxns, _ = isExist(gav, rxns)

    ratioSndNe0 = np.zeros(grp)
    for i in range(grp):
        inte = sum(ratioH[i, gavrxns] != 0)
        ratioSndNe0[i] = sum(ratioH[i, sndrxns] != 0)
        if inte:
            asigmag[i] = True
        if ratioSndNe0[i] == inte:
            sigmag[i, lo] = 0
        else:
            sigmag[i, lo] = - sigmaLup

    sigmagI = np.where(asigmag)[0]

    sigmaF = toFlimit1d((sigmag, np.ones((grp, bndI), dtype=bool)), vsigmaL, grp)
    trans(sigmaF)
    sigmaglo, sigmagup = toLoUp(sigmaF)
    gI = np.where(sndrxns | (gavrxns & (np.logical_not(sndrxns))))[0]
    glo = np.zeros(rxn)
    gup = np.zeros(rxn)
    glo[sndrxns] = - Temp * sigmaLup
    for i in range(rxn):
        if sndrxns[i]:
            gup[i] = 0
        elif gavrxns[i]:
            gup[i] = Temp * sigmaLup
            glo[i] = - np.inf

    excrxns, _ = isExist(exchanges, rxns)
    gexg = np.zeros((grp, bndI))
    agexg = np.zeros(grp, bool)
    for i in range(grp):
        # 如果有nan，那么 != 0 一定返回false
        if any(ratioH[i, excrxns] != 0):
            gexg[i, lo] = -1e5
            gexg[i, up] = +1e5
            agexg[i] = True

    gexgF = toFlimit1d((gexg, np.ones((grp, bndI), dtype=bool)), vgexg, grp)
    trans(gexgF)
    gexglo, gexgup = toLoUp(gexgF)
    gexgI = np.where((gexglo != 0) | (gexgup != 0))[0]
    sqrdrGlo = np.ones(rxn) * np.pi ** 0.5
    sqrdrGI = list(range(rxn))

    ## tcbm_v2bnd.inc 45 - 50

    # 假定上述两者处理后，变为下述的两个1d数列
    drGeMA = drGerrMax
    drGeMI = drGerrMin

    drGerrorlo = np.zeros(rxn)
    drGerrorup = np.zeros(rxn)
    consdrGerxns, _ = isExist(consdrGerr, rxns)
    # drGerrorindex = np.where(consdrGerxns)[0]
    drGerrorlo[consdrGerxns] = drGeMI[0][consdrGerxns] - 0.15 * np.abs(drGeMI[0][consdrGerxns])
    drGerrorup[consdrGerxns] = drGeMA[0][consdrGerxns] + 0.15 * np.abs(drGeMI[0][consdrGerxns])
    drGerrorlo[np.logical_not(drGeMI[1])] = nan
    drGerrorup[np.logical_not(drGeMI[1]) & np.logical_not(drGeMA[1])] = nan  # ?
    drGerrorI = np.where(consdrGerxns)[0]

    maps = np.zeros((met, comp), dtype=bool)
    mapfilter = np.zeros(met, dtype=bool)
    sinte = copy.deepcopy(S[0])
    sinte[np.logical_not(S[1])] = 0
    sinte = sinte != 0
    for i in range(met):
        if (mets[i] != 'h') & (mets[i] != 'charge'):
            mapfilter[i] = True
            for j in range(comp):
                if any(sinte[i, j, :]):
                    maps[i, j] = True
    mapfilter1 = np.zeros((met, comp), dtype=bool)
    for i in range(met):
        if mapfilter[i]:
            for j in range(comp):
                mapfilter1[i, j] = maps[i, j]
    eInComp = comps.index('e')
    mapfilter2 = maps[:, eInComp]

    asnd = np.zeros(rxn, bool)
    blockrxns, _ = isExist(blocked, rxns)
    nblock = np.logical_not(blockrxns)
    signdrG = np.sign(drGlo) != np.sign(drGup)
    asnd[sndrxns & signdrG & nblock] = True

    asndg = np.zeros(grp, bool)
    for i in range(grp):
        if (grps[i] != 'gl') & any((ratioH[i, sndrxns] != 0)):
            asndg[i] = True

    vglo[asndg & (drGgup < 0) & (vglo < 0)] = 0
    vgup[asndg & (drGglo > 0) & (vgup > 0)] = 0
    negratio = (ratioH < 0) * ratioH
    posratio = (ratioH > 0) * ratioH
    intelo = np.zeros((grp, rxn))
    inteup = np.zeros((grp, rxn))
    vlimfilter1 = np.isnan(vglo)
    vlimfilter2 = np.isnan(vgup)
    # vlimfilter = np.logical_not(ratio[1])
    for i in range(grp):
        if not vlimfilter1[i]:
            intelo[i, :] = posratio[i, :] * vglo[i]
            inteup[i, :] = negratio[i, :] * vglo[i]
        if not vlimfilter2[i]:
            intelo[i, :] = intelo[i, :] + negratio[i, :] * vgup[i]
            inteup[i, :] = inteup[i, :] + posratio[i, :] * vgup[i]

    for i in range(rxn):
        vlo[i] = max(-500, sum(intelo[:, i])) - 1e-4
        vup[i] = min(500, sum(inteup[:, i])) + 1e-4

    for i in range(grp):
        if (grps[i] != 'gl') and sum(ratioH[i, sndrxns] != 0) == sum(ratioH[i, sndrxns & np.logical_not(asnd)] != 0):
            asndg[i] = False

    adrG = np.zeros(rxn, bool)
    inte = np.zeros(rxn, bool)
    inte1 = lnclo[maps] == lncup[maps]
    for i in range(rxn):
        inte[i] = (sum(sinte[maps, i]) > sum(sinte[maps, i] & inte1)) > 0

    adrG[gavrxns & (drGlo != drGup) & nblock & inte] = True
    adrGg = np.zeros(grp, bool)
    inte = np.zeros(grp, bool)
    for i in range(grp):
        inte[i] = any(adrG[ratioH[i, :] != 0])

    adrGg[asigmag & (drGglo != drGgup) & inte] = True

    asndList = list(np.where(asnd)[0])
    inter = set()
    gavrxnsList = list(np.where(gavrxns)[0])
    gavrxnsArr = np.array(gavrxnsList)
    for i in range(grp):
        if adrGg[i]:
            inter = inter | set(np.where(ratio[0][i, gavrxnsArr] != 0)[0])
    drGI = set(np.where(adrG)[0]) | set(asndList) | inter
    adfG = np.zeros(rxn, bool)
    adfG[excrxns & (dfGlo != dfGup)] = True

    adfGg = np.zeros(grp, bool)
    inte = np.zeros(grp, bool)
    for i in range(grp):
        inte[i] = any(adfG[ratioH[i, :] != 0])

    adfGg[agexg & (dfGglo != dfGgup) & inte] = True

    # nparray to list
    def nan2none(i):
        if np.isnan(i):
            return None
        else:
            return i

    def nan2other(i, n):
        if np.isnan(i):
            return n
        else:
            return i

    def transs1D(narr1, narr2, n1, n2):
        def index(m, i):
            return (nan2other(narr1[i], n1), nan2other(narr2[i], n2))

        return index

    def transs2D(narr1, narr2, n1, n2):
        def index(m, i, j):
            return (nan2other(narr1[i, j], n1), nan2other(narr2[i, j], n2))

        return index

    tcbm_lp = ConcreteModel()

    (len0, len1) = lnclo.shape
    lncI = []
    for i in range(len0):
        # inter = []
        for j in range(len1):
            if maps[i, j] & (len(np.where((S[0][i, j, :] != 0) & (S[1][i, j, :]))[0]) != 0):
                lncI.append((i, j))

    Mmets = list(range(met))
    Mcomps = list(range(comp))
    Mgrps = list(range(grp))
    Mrxns = list(range(rxn))

    vpindex = np.where(asnd)[0]
    vnindex = np.where(asnd)[0]
    drGpindex = np.where(asnd)[0]
    drGnindex = np.where(asnd)[0]
    sqrdrGindex = np.where(asnd)[0]
    bindex = np.where(asnd)[0]
    vpI = vpindex
    vnI = vnindex
    drGpI = drGpindex
    drGnI = drGnindex
    sqrdrGI = sqrdrGindex
    bI = bindex
    tcbm_lp.sigmac = Var(within=Reals, bounds=(nan2other(sigmalo, - 0), nan2other(sigmaup, 0)))
    # tcbm_lp.vg = Var(Mgrps, within=Reals, bounds = transs1D(vglo, vgup, -0, 0))
    tcbm_lp.vg = Var(vgI, within=Reals, bounds=transs1D(vglo, vgup, -0, 0))
    # tcbm_lp.drGg = Var(Mgrps, within=Reals, bounds = transs1D(drGglo, drGgup, -0, 0))
    tcbm_lp.drGg = Var(drGgI, within=Reals, bounds=transs1D(drGglo, drGgup, -0, 0))
    # tcbm_lp.dfGg = Var(Mgrps, within=Reals, bounds = transs1D(dfGglo, dfGgup, -0, 0))
    tcbm_lp.dfGg = Var(dfGgI, within=Reals, bounds=transs1D(dfGglo, dfGgup, -0, 0))
    # tcbm_lp.sigmag = Var(Mgrps, within=Reals, bounds = transs1D(sigmaglo, sigmagup, -0, 0))
    tcbm_lp.sigmag = Var(sigmagI, within=Reals, bounds=transs1D(sigmaglo, sigmagup, -0, 0))
    # tcbm_lp.gexg = Var(Mgrps, within=Reals, bounds = transs1D(gexglo, gexgup, -0, 0))
    tcbm_lp.gexg = Var(gexgI, within=Reals, bounds=transs1D(gexglo, gexgup, -0, 0))
    # tcbm_lp.drG = Var(Mrxns, within=Reals, bounds = transs1D(drGlo, drGup, -0, 0))
    tcbm_lp.drG = Var(drGI, within=Reals, bounds=transs1D(drGlo, drGup, -0, 0))
    # tcbm_lp.dfG = Var(Mrxns, within=Reals, bounds = transs1D(dfGlo, dfGup, -0, 0))
    tcbm_lp.dfG = Var(dfGI, within=Reals, bounds=transs1D(dfGlo, dfGup, -0, 0))
    # tcbm_lp.lnc = Var(Mmets, Mcomps, within=Reals, bounds = transs2D(lnclo, lncup, -0, 0))
    tcbm_lp.lnc = Var(lncI, within=Reals, bounds=transs2D(lnclo, lncup, -0, 0))
    # tcbm_lp.v   = Var(Mrxns, within=Reals, bounds = transs1D(vlo, vup, -0, 0))
    tcbm_lp.v = Var(vI, within=Reals, bounds=transs1D(vlo, vup, -0, 0))
    # tcbm_lp.g   = Var(Mrxns, within=Reals, bounds = transs1D(glo, gup, -0, 0))
    tcbm_lp.g = Var(gI, within=Reals, bounds=transs1D(glo, gup, -0, 0))
    # tcbm_lp.drGerror = Var(Mrxns, within=Reals, bounds = transs1D(drGerrorlo, drGerrorup, -0, 0))
    tcbm_lp.drGerror = Var(drGerrorI, within=Reals, bounds=transs1D(drGerrorlo, drGerrorup, -0, 0))

    tcbm_lp.vp = Var(vpI, within=NonNegativeReals, bounds=(0, +np.inf))
    tcbm_lp.vn = Var(vnI, within=NonNegativeReals, bounds=(0, +np.inf))
    tcbm_lp.drGp = Var(drGpI, within=NonNegativeReals, bounds=(0, +np.inf))
    tcbm_lp.drGn = Var(drGnI, within=NonNegativeReals, bounds=(0, +np.inf))
    tcbm_lp.sqrdrG = Var(sqrdrGI, within=NonNegativeReals, bounds=(0.25, +np.inf))
    # tcbm_lp.conc= Var(Mmets, tcbm_lp) 原代码没有使用
    # tcbm_lp.concomb = Var(Mmets, within=NonNegativeReals)
    tcbm_lp.b = Var(bI, within=pyo.Binary)

    # 考虑到本身pyomo本身不接受bool数组作为指标，所以这里将原有bool数组变为指标list
    def trans3(boolArr):
        resu = []
        for i in range(len(boolArr)):
            if boolArr[i]:
                resu.append(i)
        return resu

    def trans32D(boolArr):
        resu = []
        len0, len1 = boolArr.shape
        for i in range(len0):
            for j in range(len1):
                if boolArr[i, j]:
                    resu.append((i, j))
        return resu

    def trans4(nArr, bArr):
        resu = {}
        len0 = len(nArr)
        if nArr.ndim > 1:
            for i in range(len0):
                inter = trans4(nArr[i], bArr[i])
                if len(inter) > 0:
                    resu[i] = inter
        else:
            for i in range(len0):
                if bArr[i]:
                    resu[i] = nArr[i]
        return resu

    # 为应对某些情况下直接使用无用index为零
    def trans5(value, index, index1):
        for i in index1:
            if not (i in index):
                value[i] = 0

    Sgn, Sgb = Sg
    SgD = trans4(Sgn, Sgb)

    def massbalances(m, meti, compj):
        inter = SgD.get(meti).get(compj)
        return sum(inter[k] * m.vg[k] for k in inter.keys() if (k in vgI)) == 0

    massbalList = []
    for meti in SgD.keys():
        inter1 = SgD.get(meti)
        for compj in inter1.keys():
            massbalList.append((meti, compj))

    tcbm_lp.massbalances = Constraint(massbalList, rule=massbalances)

    # v, vg transformation, GEB & sigmac
    ration, ratiob = ratio
    ratioD = trans4(ratioH.transpose(), ratiob.transpose())

    def vgtov1(m, rxni):
        inter1 = ratioD[rxni]
        return m.v[rxni] == sum(inter1[j] * m.vg[j] for j in inter1.keys())

    def vgtov2(m, rxni):
        return m.v[rxni] == 0

    vgtovList1 = list(set(ratioD.keys()) & set(vI))
    vgtovList2 = list(set(vI) - set(ratioD.keys()))
    tcbm_lp.vgtov1 = Constraint(vgtovList1, rule=vgtov1)
    tcbm_lp.vgtov2 = Constraint(vgtovList2, rule=vgtov2)

    agexgIn = list(set(trans3(agexg)) & set(gexgI))

    def gibbsenergybalance(m):
        return -Temp * m.sigmac == sum(m.gexg[j] for j in agexgIn)

    tcbm_lp.gibbsenergybalance = Constraint(rule=gibbsenergybalance)
    asigmagIn = list(set(trans3(asigmag)) & set(sigmagI))

    def defsigmac(m):
        return m.sigmac == sum(m.sigmag[j] for j in asigmagIn)

    tcbm_lp.defsigmac = Constraint(rule=defsigmac)

    # drG, dfG + drGg, dfGg transformation
    Stn = np.transpose(S[0], axes=(2, 0, 1))
    Stb = np.transpose(S[1], axes=(2, 0, 1))
    StD = {}
    stdlen0, stdlen1, stdlen2 = Stn.shape
    for i in range(stdlen0):
        interD = {}
        for j in range(stdlen1):
            for k in range(stdlen2):
                if Stb[i, j, k]:
                    interD[(j, k)] = Stn[i, j, k]
        if len(interD) != 0:
            StD[i] = interD

    trans5(drGt0tr[0], np.where(drGt0tr[1])[0], range(rxn))
    trans5(drGSE[0], np.where(drGSE[1])[0], range(rxn))
    # defdrGH =  adrG & drGt0tr[1] & consdrGerxns & drGSE[1]
    # 考虑到当不存在时直接以零代替其中的值，所以为了简便，直接将所有的rxns分为两部分，一部分为adrG True 且consdrGerxns True，另一部分仅为前者true
    defdrGH1 = adrG & consdrGerxns
    defdrGH2 = adrG & np.logical_not(consdrGerxns)
    # 实际上这里的StD.keys()没必要，因为其为range(rxn), 因此这里不考虑输入的rxni无对应key的状况
    # 这里的set(drGI) 也没必要，因为drGI本身含有adrG
    defdrGList1 = list(set(np.where(defdrGH1)[0]) & set(StD.keys()) & set(drGI) & set(drGerrorI))
    defdrGList2 = list(set(np.where(defdrGH2)[0]) & set(StD.keys()) & set(drGI))
    mapf1InH1, mapf1InH2 = np.where(mapfilter1)
    h2oInMe = mets.index('h2o')
    mapf1Dict1 = {}
    for i in defdrGList1:
        inter1 = StD[i]
        inter2 = []
        for (j, k) in inter1.keys():
            if j == h2oInMe:
                continue
            if mapfilter1[j, k] & ((j, k) in lncI):
                inter2.append((j, k))
        mapf1Dict1[i] = inter2
    mapf1Dict2 = {}
    for i in defdrGList2:
        inter1 = StD[i]
        inter2 = []
        for (j, k) in inter1.keys():
            if j == h2oInMe:
                continue
            if mapfilter1[j, k] & ((j, k) in lncI):
                inter2.append((j, k))
        mapf1Dict2[i] = inter2

    def defdrG1(m, rxni):
        inter1 = StD.get(rxni)
        return m.drG[rxni] == drGt0tr[0][rxni] + m.drGerror[rxni] * drGSE[0][rxni] + GasCons * Temp * sum(
            StD[rxni][index] * m.lnc[index] for index in mapf1Dict1[rxni])

    tcbm_lp.defdrG1 = Constraint(defdrGList1, rule=defdrG1)

    def defdrG2(m, rxni):
        inter1 = StD.get(rxni)
        return m.drG[rxni] == drGt0tr[0][rxni] + GasCons * Temp * sum(
            StD[rxni][index] * m.lnc[index] for index in mapf1Dict2[rxni])

    tcbm_lp.defdrG2 = Constraint(defdrGList2, rule=defdrG2)

    ratioD2 = trans4(ration, ratiob)
    defdrGgList = list(set(np.where(adrGg)[0]) & set(drGgI) & set(ratioD2.keys()) & set(drGgI))
    defdrGgDict = {}
    for i in ratioD2.keys():
        inter1 = ratioD2[i]
        inter2 = []
        for j in inter1.keys():
            if (j in gavrxnsList) & (j in drGI) & (ration[i, j] != 0):
                inter2.append(j)
        defdrGgDict[i] = inter2

    def defdrGg(m, grpi):
        return m.drGg[grpi] == sum(ratioD2[grpi][rxnj] * m.drG[rxnj] for rxnj in defdrGgDict[grpi])

    tcbm_lp.defdrGg = Constraint(defdrGgList, rule=defdrGg)

    trans5(dfG0[0][:, eInComp], np.where(dfG0[1][:, eInComp])[0], range(met))
    # inter1 = np.where(dfG0[1][:, eInComp] & mapfilter2)[0] 这里因为dfG0[1] 为false的时候，直接用0代替
    inter1 = np.where(mapfilter2)[0]
    # 这里的一个前提时dfGI是全覆盖
    defdfList = list(set(np.where(adfG)[0]) & set(StD.keys()) & set(dfGI))
    defdfGDict = {}
    for i in defdfList:
        inter2 = StD.get(i)
        inter3 = []
        for j in inter1:
            if ((j, eInComp) in inter2.keys()) & ((j, eInComp) in lncI):
                inter3.append(j)
        defdfGDict[i] = inter3

    def defdfG(m, rxni):
        return m.dfG[rxni] == - sum(
            StD[rxni][(j, eInComp)] * (dfG0[0][j, eInComp] + GasCons * Temp * m.lnc[j, eInComp]) for j in
            defdfGDict[rxni])

    tcbm_lp.defdfG = Constraint(defdfList, rule=defdfG)

    excrxnsList = list(np.where(excrxns)[0])
    # dfGgI全覆盖
    defdfGgList = list(set(np.where(adfGg)[0]) & set(dfGgI))
    defdfGgDict = {}
    for i in ratioD2.keys():
        inter1 = ratioD2[i]
        defdfGgDict[i] = list(set(inter1.keys()) & set(excrxnsList) & set(dfGI))

    def defdfGg(m, grpi):
        return m.dfGg[grpi] == sum(ratioD2[grpi][rxnj] * m.dfG[rxnj] for rxnj in defdfGgDict[grpi])

    tcbm_lp.defdfGg = Constraint(defdfGgList, rule=defdfGg)

    # ATP maintenance
    g62InGrp = grps.index('g62')
    glInGrp = grps.index('g1')

    def atpm(m):
        return m.vg[g62InGrp] >= 3.15 + 31.2622 * m.vg[glInGrp]

    tcbm_lp.atpm = Constraint(rule=atpm)

    # MIP
    tcbm_mip = tcbm_lp.clone()

    # MIP 2ed law
    def defvgn(m, rxni):  # vI 全覆盖，其余两者等同asndList
        return m.v[rxni] == m.vp[rxni] - m.vn[rxni]

    tcbm_mip.defvgn = Constraint(asndList, rule=defvgn)
    # vlo 以及vup应该是全覆盖， 所以下者和 asndList 没什么区别
    # vsndList = list(set(np.where(np.logical_not(np.isnan(vlo)) & np.logical_not(np.isnan(vup)))[0]) & set(asndList))
    vsndList = asndList

    def defvp(m, rxni):
        return m.vp[rxni] <= max(abs(m.v[rxni].lb), abs(m.v[rxni].ub)) * m.b[rxni]

    # tcbm_mip.defvp = Constraint(vsndList, rule = defvp)

    def defvn(m, rxni):
        return m.vn[rxni] <= max(abs(m.v[rxni].lb), abs(m.v[rxni].ub)) * (1 - m.b[rxni])

    # tcbm_mip.defvn = Constraint(vsndList, rule = defvn)

    # drGI 包含asnd, drGpI drGnI韩宇asnd
    def defdrGpn(m, rxni):
        return m.drG[rxni] == m.drGp[rxni] - m.drGn[rxni];

    tcbm_mip.defdrGpn = Constraint(asndList, rule=defdrGpn)
    # drGI 全覆盖， bI  drGpI 含于asnd
    drGsndlist = list(
        set(np.where(np.logical_not(np.isnan(drGlo)) & np.logical_not(np.isnan(drGup)))[0]) & set(asndList))

    def defdrGp(m, rxni):
        return m.drGp[rxni] <= max(abs(m.drG[rxni].lb), abs(m.drG[rxni].ub)) * (1 - m.b[rxni]);

    # tcbm_mip.defdrGp = Constraint(asndList, rule = defdrGp)
    # drGnI含于asnd， 同上
    def defdrGn(m, rxni):
        return m.drGn[rxni] <= max(abs(m.drG[rxni].ub), abs(m.drG[rxni].lb)) * m.b[rxni];

    # tcbm_mip.defdrGn = Constraint(asndList, rule = defdrGn)

    def defstrict(m, rxni):
        return m.drGp[rxni] + m.drGn[rxni] >= epsilon;

    tcbm_mip.defstrict = Constraint(asndList, rule=defstrict)

    ## minlp
    tcbm_minlp = tcbm_mip.clone()
    asigmagList = list(np.where(asigmag)[0])

    # 这里 sigmagI drGgI vgI 均全覆盖
    def defsigma(m, grpi):
        return - Temp * m.sigmag[grpi] == m.drGg[grpi] * m.vg[grpi]

    tcbm_minlp.defsigma = Constraint(asigmagList, rule=defsigma)
    # 同样dfGgI 也是全覆盖
    # 因为agexgList中全部元素为gexg定义域，所以不用考虑gexg不存在的情况
    agexgList = np.where(agexg)[0]

    def defgex(m, grpi):
        return m.gexg[grpi] == m.dfGg[grpi] * m.vg[grpi]

    tcbm_minlp.defgex = Constraint(agexgList, rule=defgex)

    ## nlp
    tcbm_nlp = tcbm_lp.clone()

    # 这里gI 含于 asndList,
    # drGI 含有 asndList
    def defgb(m, rxni):
        return m.g[rxni] == m.v[rxni] * m.drG[rxni]

    tcbm_nlp.defgb = Constraint(asndList, rule=defgb)

    # 两者均全覆盖，故直接使用asndList
    def strinctnlp(m, rxni):
        return m.drG[rxni] ** 2 == m.sqrdrG[rxni]

    tcbm_nlp.strinctnlp = Constraint(asndList, rule=strinctnlp)

    # drGI和sqrdrGI 全覆盖，
    def strinctnlp_lp(m, rxni):
        return (m.drG[rxni].lb + m.drG[rxni].ub) * m.drG[rxni] - m.drG[rxni].lb * m.drG[rxni].ub >= m.sqrdrG[rxni]

    # tcbm_nlp.strinctnlp_lp = Constraint(asndList, rule = strinctnlp_lp)
    ### 所以这些含有边界的到底要放在什么地方？
    # 对于gI含于 asndList
    drGuVuSndList = list(
        set(np.where(np.logical_not(np.isnan(drGup)) & np.logical_not(np.isnan(vup)))[0]) & set(asndList))

    def defg_mccormicklb(m, rxni):
        return m.g[rxni] >= m.v[rxni] * m.drG[rxni].ub + m.v[rxni].ub * m.drG[rxni] - m.v[rxni].ub * m.drG[rxni].ub

    # tcbm_nlp.defg_mccormicklb = Constraint(asndList, rule = defg_mccormicklb)
    # 同上
    drGlVlSndList = list(
        set(np.where(np.logical_not(np.isnan(drGlo)) & np.logical_not(np.isnan(vlo)))[0]) & set(asndList))

    def defg_mccormick2b(m, rxni):
        return m.g[rxni] >= m.v[rxni] * m.drG[rxni].lb + m.v[rxni].lb * m.drG[rxni] - m.v[rxni].lb * m.drG[rxni].lb

    # tcbm_nlp.defg_mccormick2b = Constraint(asndList, rule = defg_mccormick2b)
    # 同上
    drGuVlSndList = list(
        set(np.where(np.logical_not(np.isnan(drGup)) & np.logical_not(np.isnan(vlo)))[0]) & set(asndList))

    def defg_mccormick3b(m, rxni):
        return m.g[rxni] <= m.v[rxni] * m.drG[rxni].ub + m.v[rxni].lb * m.drG[rxni] - m.v[rxni].lb * m.drG[rxni].ub

    # tcbm_nlp.defg_mccormick3b = Constraint(asndList, rule = defg_mccormick3b)
    # 同上
    drGlVuSndList = list(
        set(np.where(np.logical_not(np.isnan(drGlo)) & np.logical_not(np.isnan(vup)))[0]) & set(asndList))

    def defg_mccormick4b(m, rxni):
        return m.g[rxni] <= m.v[rxni] * m.drG[rxni].lb + m.v[rxni].ub * m.drG[rxni] - m.v[rxni].ub * m.drG[rxni].lb

    # tcbm_nlp.defg_mccormick4b = Constraint(asndList, rule = defg_mccormick4b)
    # 检查上述四个list相同
    if not (drGuVuSndList == drGlVlSndList):
        print("error")
    if not (drGlVuSndList == drGlVlSndList):
        print("error")
    if not (drGuVlSndList == drGlVlSndList):
        print("error")
    tcbm_nlp.defsigma = Constraint(asigmagList, rule=defsigma)
    tcbm_nlp.defgex = Constraint(agexgList, rule=defgex)

    ## nullspace
    # consNS nsID 未定义
    # dfGt0 未定义
    cInComps = comps.index('c')
    # 假定 nsID 为nsID元的数目，而nsIDs为字符串数组
    consNSnsID, _ = isExist(consNS, nsIDs)
    MnsIDs = list(range(nsID))
    KtD = trans4(K[0].transpose(), K[1].transpose())
    nullsList = list(set(MnsIDs) & set(KtD.keys()))
    # 保证k的选取成功，S选取成功，drGSE存在且小于1000
    inter21 = set(np.where(drGSE[0] < 1000)[0]) & set(np.where(drGSE[1])[0]) & set(StD.keys())
    nullsDict = {}

    for i in KtD.keys():
        inter1 = KtD[i]
        inter2 = list(set(inter1.keys()) & inter21)
        inter3 = {}
        for j in inter1.keys():
            inter4 = StD[j].keys()
            inter5 = []
            if j in inter21:
                for (k1, k2) in inter4:
                    if maps[k1, k2] & dfGt0[1][k1, cInComps]:
                        inter5.append((k1, k2))
            inter3[j] = inter5
        if len(inter3) != 0:
            nullsDict[i] = inter3

    nullsDict2 = {}

    nullsList = nullsDict.keys()
    # 因为下面计算中合并同类项较多， 因此建议对rxnj compm 求和，使得最后只对于metk有部分
    for nsIDi in nullsList:
        interD = {}
        interi = nullsDict[nsIDi]
        for rxnj in interi.keys():
            k = (KtD[nsIDi][rxnj])
            interij = interi[rxnj]
            for (metk, compm) in interij:
                if metk in interD.keys():
                    interD[metk] = interD[metk] + k * (StD[rxnj][(metk, compm)])
                else:
                    interD[metk] = k * (StD[rxnj][(metk, compm)])
        inter0 = 0
        for metk in interD.keys():
            inter0 = inter0 + interD[metk] * dfGt0[0][metk, cInComps]
        nullsDict2[nsIDi] = inter0

    def nullspace(m, nsIDi):
        resu = nullsDict2[nsIDi]
        inter1 = nullsDict[nsIDi]
        for rxnj in inter1.keys():
            k = KtD[nsIDi][rxnj]
            if consdrGerxns[rxnj]:
                resu = resu + k * m.drGerror[rxnj] * drGSE[0][rxnj]
        return resu == 0

    # def nullspace(m, nsIDi):
    #     resu = 0
    #     inter1 = nullsDict[nsIDi]
    #     for rxnj in inter1.keys():
    #         k = KtD[nsIDi][rxnj]
    #         inter2 = inter1[rxnj]
    #         resu = resu + sum(StD[rxnj][(metk, compm)] * dfGt0[0][metk, cInComps] * k for (metk, compm) in inter2)
    #         if consdrGerxns[rxnj]:
    #             resu = resu + k * m.drGerror[rxnj] * drGSE[0][rxnj]
    #     return resu == 0

    drGgups = np.logical_not(np.isnan((drGgup)))
    drGglos = np.logical_not(np.isnan((drGglo)))
    vgups = np.logical_not(np.isnan((vgup)))
    vglos = np.logical_not(np.isnan((vglo)))

    # drGguVguSigmaList = list(set(np.where(drGgups & vgups)[0]) & set(asigmagList))
    drGguVguSigmaList = asigmagList

    def defsigma_mccormick1(m, grpi):
        return - Temp * m.sigmag[grpi] >= m.vg[grpi] * m.drGg[grpi].ub + m.vg[grpi].ub * m.drGg[grpi] - m.vg[grpi].ub * \
            m.drGg[grpi].ub;

    # drGglVglSigmaList = list(set(np.where(drGglos & vglos)[0]) & set(asigmagList))
    drGglVglSigmaList = asigmagList

    def defsigma_mccormick2(m, grpi):
        return - Temp * m.sigmag[grpi] >= m.vg[grpi] * m.drGg[grpi].lb + m.vg[grpi].lb * m.drGg[grpi] - m.vg[grpi].lb * \
            m.drGg[grpi].lb;

    # drGguVglSigmaList = list(set(np.where(drGgups & vglos)[0]) & set(asigmagList))
    drGguVglSigmaList = asigmagList

    def defsigma_mccormick3(m, grpi):
        return - Temp * m.sigmag[grpi] <= m.vg[grpi] * m.drGg[grpi].ub + m.vg[grpi].lb * m.drGg[grpi] - m.vg[grpi].lb * \
            m.drGg[grpi].ub

    # drGglVguSigmaList = list(set(np.where(drGglos & vgups)[0]) & set(asigmagList))
    drGglVguSigmaList = asigmagList

    def defsigma_mccormick4(m, grpi):
        return - Temp * m.sigmag[grpi] <= m.vg[grpi] * m.drGg[grpi].lb + m.vg[grpi].ub * m.drGg[grpi] - m.vg[grpi].ub * \
            m.drGg[grpi].lb

    dfGgups = np.logical_not(np.isnan((dfGgup)))
    dfGglos = np.logical_not(np.isnan((dfGglo)))

    # 同样dfGgI vgI也是全覆盖
    # dfGguVguSigmaList = list(set(np.where(dfGgups & vgups)[0]) & set(agexgList))
    dfGguVguSigmaList = agexgList

    def defgex_mccormick1(m, grpi):
        return m.gexg[grpi] >= m.vg[grpi] * m.dfGg[grpi].ub + m.vg[grpi].ub * m.dfGg[grpi] - m.vg[grpi].ub * m.dfGg[
            grpi].ub

    # dfGglVglSigmaList = list(set(np.where(dfGglos & vglos)[0]) & set(agexgList))
    dfGglVglSigmaList = agexgList

    def defgex_mccormick2(m, grpi):
        return m.gexg[grpi] >= m.vg[grpi] * m.dfGg[grpi].lb + m.vg[grpi].lb * m.dfGg[grpi] - m.vg[grpi].lb * m.dfGg[
            grpi].lb

    # dfGguVglSigmaList = list(set(np.where(dfGgups & vglos)[0]) & set(agexgList))
    dfGguVglSigmaList = agexgList

    def defgex_mccormick3(m, grpi):
        return m.gexg[grpi] <= m.vg[grpi] * m.dfGg[grpi].ub + m.vg[grpi].lb * m.dfGg[grpi] - m.vg[grpi].lb * m.dfGg[
            grpi].ub

    # dfGglVguSigmaList = list(set(np.where(dfGglos & vgups)[0]) & set(agexgList))
    dfGglVguSigmaList = agexgList

    def defgex_mccormick4(m, grpi):
        return m.gexg[grpi] <= m.vg[grpi] * m.dfGg[grpi].lb + m.vg[grpi].ub * m.dfGg[grpi] - m.vg[grpi].ub * m.dfGg[
            grpi].lb

    # 返回主文件



    tcbm_mip.defvp = Constraint(vsndList, rule=defvp)
    tcbm_mip.defvn = Constraint(vsndList, rule=defvn)
    tcbm_mip.defdrGp = Constraint(asndList, rule=defdrGp)
    tcbm_mip.defdrGn = Constraint(asndList, rule=defdrGn)
    biomassInRxns = rxns.index('biomass')
    obj_exp_List = list(ratioD[biomassInRxns].keys())

    ATP = ['ACCOACr', 'ACKr', 'ADK1', 'ALAALAr', 'AP5AH', 'ASPK', 'ATPS4r', 'CBIAT', 'CBLAT', 'DHBSr', 'DTMPK', 'GALKr',
           'GK1', 'NDPK1', 'NDPK2', 'NDPK3', 'NDPK4', 'NDPK5', 'NDPK6', 'NDPK7', 'NDPK8', 'PGK', 'PPAKr', 'PRAGSr',
           'PRASCS', 'PRPPS', 'PYK', 'SERASr', 'SUCOAS', 'TMKr', 'TMPKr', 'UMPK', 'URIDK2r']

    def obj_exp(m):
        #Stn = np.transpose(S[0], axes=(0, 2, 1))
        #Stb = np.transpose(S[1], axes=(0, 2, 1))
        #obj = 0
        #stdlen0, stdlen1, stdlen2 = Stn.shape
        #for j in ATP:
         #   for k in range(stdlen2):
         #       if Stb[175, rxns.index(j), k]:
         #           obj = obj + Stn[i, rxns.index(j), k] * m.v[rxns.index(j)]
        #return obj
         return sum(ratioD[biomassInRxns][i] * m.vg[i] for i in obj_exp_List)

    # FBA_mip.add_components(tcbm_McCormick.component_objects())
    FBA_mip = tcbm_mip.clone()
    FBA_mip.defsigma_mccormick1 = Constraint(drGguVguSigmaList, rule=defsigma_mccormick1)
    FBA_mip.defsigma_mccormick2 = Constraint(drGglVglSigmaList, rule=defsigma_mccormick2)
    FBA_mip.defsigma_mccormick3 = Constraint(drGguVglSigmaList, rule=defsigma_mccormick3)
    FBA_mip.defsigma_mccormick4 = Constraint(drGglVguSigmaList, rule=defsigma_mccormick4)
    FBA_mip.defgex_mccormick1 = Constraint(dfGguVguSigmaList, rule=defgex_mccormick1)
    FBA_mip.defgex_mccormick2 = Constraint(dfGglVglSigmaList, rule=defgex_mccormick2)
    FBA_mip.defgex_mccormick3 = Constraint(dfGguVglSigmaList, rule=defgex_mccormick3)
    FBA_mip.defgex_mccormick4 = Constraint(dfGglVguSigmaList, rule=defgex_mccormick4)
    FBA_mip.nullspace = Constraint(nullsList, rule=nullspace)
    FBA_mip.OBJ = Objective(rule=obj_exp, sense=maximize)

    # 因为fba_nlp无法直接复制，这里复制tcbm_nlp，然后加上各种约束，obj以及新界：
    def nlpCopy():
        nlp = tcbm_nlp.clone()
        ## 注意如下五个限制，可能有问题
        nlp.strinctnlp_lp = Constraint(asndList, rule=strinctnlp_lp)
        nlp.defg_mccormicklb = Constraint(asndList, rule=defg_mccormicklb)
        nlp.defg_mccormick2b = Constraint(asndList, rule=defg_mccormick2b)
        nlp.defg_mccormick3b = Constraint(asndList, rule=defg_mccormick3b)
        nlp.defg_mccormick4b = Constraint(asndList, rule=defg_mccormick4b)
        nlp.defsigma_mccormick1 = Constraint(drGguVguSigmaList, rule=defsigma_mccormick1)
        nlp.defsigma_mccormick2 = Constraint(drGglVglSigmaList, rule=defsigma_mccormick2)
        nlp.defsigma_mccormick3 = Constraint(drGguVglSigmaList, rule=defsigma_mccormick3)
        nlp.defsigma_mccormick4 = Constraint(drGglVguSigmaList, rule=defsigma_mccormick4)
        nlp.defgex_mccormick1 = Constraint(dfGguVguSigmaList, rule=defgex_mccormick1)
        nlp.defgex_mccormick2 = Constraint(dfGglVglSigmaList, rule=defgex_mccormick2)
        nlp.defgex_mccormick3 = Constraint(dfGguVglSigmaList, rule=defgex_mccormick3)
        nlp.defgex_mccormick4 = Constraint(dfGglVguSigmaList, rule=defgex_mccormick4)
        nlp.nullspace = Constraint(nullsList, rule=nullspace)
        nlp.OBJ = Objective(rule=obj_exp, sense=maximize)

        return nlp

    FBA_nlp = nlpCopy()

    # FBA_mip.write(modeldir + 'fba_mip.lp', io_options={'symbolic_solver_labels': True})

    ## 重新规划，假定该文件只提供有用的东西为 模型文件 FBA_mip,以及能够生成后续nlp模型的函数 nlpgenerate, 而相关的模型求解依赖于其他文件
    ## 所以如下改成函数，输入量为soluArr,
    ## 假定利用cplex_solu 求解完成，得到resu
    # soluArr = solvemip('fba_mip.lp', tee=False)
    ## 补足解集
    ## 解 名 index, 只保留单index的
    naliLi = [('vg', vgI), ('drGg', drGgI), ('dfGg', dfGgI), ('sigmag', list(sigmagI)), ('gexg', list(gexgI)),
              ('drG', list(drGI)), ('dfG', dfGI), ('v', vI), ('drGerror', list(drGerrorI)), ('vp', list(vpI)),
              ('vn', list(vnI)), ('drGp', list(drGpI)), ('drGn', list(drGnI)), ('sqrdrG', list(sqrdrGI)),
              ('b', list(bI)), ('g', list(gI)), ('lnc', lncI)]

    # 用来处理得到的初值答案, tee 表示是否打印某些警告信息
    def resultProcess(soluArr, tee=True):
        len0 = len(soluArr)
        for i in range(len0):
            inter = soluArr[i].soluDict.keys()
            if not ('sigmac' in inter):
                print("ERROR, sigmac must be in result dict Array")
            for j0, j1 in naliLi:
                if j0 == 'g':
                    # mip 模型中并没有g，所以利用定义式直接定义
                    gg = {}
                    set1 = soluArr[i].soluDict['v'].keys()
                    set2 = soluArr[i].soluDict['drG'].keys()
                    for k in j1:
                        if (k in set1) & (k in set2):
                            gg[k] = soluArr[i].soluDict['v'][k] * soluArr[i].soluDict['drG'][k]
                    soluArr[i].soluDict['g'] = gg
                    continue
                if j0 == 'sqrdrG':
                    # mip 模型中没有drGerror，所以直接赋值
                    sqrdrGD = {}
                    set1 = soluArr[i].soluDict['drG'].keys()
                    for k in j1:
                        if k in set1:
                            sqrdrGD[k] = soluArr[i].soluDict['drG'][k] ** 2
                    soluArr[i].soluDict['sqrdrG'] = sqrdrGD
                    continue
                if not (j0 in inter):
                    print("error, one variable must be ont in result dict Array")
                key = soluArr[i].soluDict[j0].keys()
                var = getattr(FBA_mip, j0)
                for k in j1:
                    if not (k in key):
                        lb = var[k].lb
                        ub = var[k].ub
                        if ((lb == None) | (ub == None)) & tee:
                            print("one bounds is None, continue")
                            continue
                        if (lb != ub) & tee:
                            print("lowbounds not eq to upper bounds for ", j0, " ", k, " continue")
                            continue
                        soluArr[i].soluDict[j0][k] = lb
        return soluArr

    # 利用之前求得的解的序列，生成nlp模型序列
    def nlpIter(soluArr):
        soluArr = resultProcess(soluArr, False)

        def trans(solus):
            nlp = nlpCopy()
            dict0 = solus.soluDict
            nlp.sigmac.value = dict0['sigmac'][0]
            for j0, _ in naliLi:
                key = dict0[j0].keys()
                var = getattr(nlp, j0)
                for k in key:
                    var[k].value = dict0[j0][k]
            for j in bI:
                nlp.b[j].fix(dict0['b'][j])
            return nlp

        return MyIter(soluArr, trans)

    # def nlpgenerate(soluArr):
    #     soluArr = resultProcess(soluArr, False)
    #     modelArr = []
    #     for solus in soluArr:
    #         nlp = nlpCopy()
    #         dict0 = solus.soluDict
    #         nlp.sigmac.value = dict0['sigmac'][0]
    #         for j0, _ in naliLi:
    #             key = dict0[j0].keys()
    #             var = getattr(nlp, j0)
    #             for k in key:
    #                 var[k].value = dict0[j0][k]
    #         for j in bI:
    #             nlp.b[j].fix(dict0['b'][j])
    #         modelArr.append(nlp)
    #     return modelArr

    # # len0 = len(soluArr)
    # # for i in range(len0):
    # #     inter = soluArr[i].soluDict.keys()
    # #     if not ('sigmac' in inter):
    # #         print("ERROR, sigmac must be in result dict Array")
    # #     for j0, j1 in naliLi:
    # #         if j0 == 'g':
    # #             # mip 模型中并没有g，所以利用定义式直接定义
    # #             gg = {}
    # #             set1 = soluArr[i].soluDict['v'].keys()
    # #             set2 = soluArr[i].soluDict['drG'].keys()
    # #             for k in j1:
    # #                 if (k in set1) & (k in set2):
    # #                     gg[k] = soluArr[i].soluDict['v'][k] * soluArr[i].soluDict['drG'][k]
    # #             soluArr[i].soluDict['g'] = gg
    # #             continue
    # #         if j0 == 'sqrdrG':
    # #             # mip 模型中没有drGerror，所以直接赋值
    # #             sqrdrGD = {}
    # #             set1 = soluArr[i].soluDict['drG'].keys()
    # #             for k in j1:
    # #                 if k in set1:
    # #                     sqrdrGD[k] = soluArr[i].soluDict['drG'][k] ** 2
    # #             soluArr[i].soluDict['sqrdrG'] = sqrdrGD
    # #             continue
    # #         if not (j0 in inter):
    # #             print("error, one variable must be ont in result dict Array")
    # #         key = soluArr[i].soluDict[j0].keys()
    # #         var = getattr(FBA_mip, j0)
    # #         for k in j1:
    # #             if not (k in key):
    # #                 lb = var[k].lb
    # #                 ub = var[k].ub
    # #                 if (lb == None) | (ub == None):
    # #                     print("one bounds is None, continue")
    # #                     continue
    # #                 if (lb != ub):
    # #                     print("lowbounds not eq to upper bounds for ", j0, " ", k, " continue")
    # #                     continue
    # #                 soluArr[i].soluDict[j0][k] = lb

    ## 定义函数能返回naliLi内所有变量的值，这里是varL
    def returnval(model, varL):
        m = model
        soluDict3 = {}
        for i0, i1 in varL:
            var = getattr(model, i0)
            inter = {}
            if i1 == None:
                inter[0] = var.value
            else:
                for j in i1:
                    inter[j] = var[j].value
            soluDict3[i0] = inter
        return soluDict3

    return (returnval, nlpIter, FBA_mip, naliLi)

# FBA_nlp.write(modeldir + 'fba_nlp.lp', io_options={'symbolic_solver_labels': True})

# resu = []
# ipopt = SolverFactory('ipopt')
# ipopt.options['bound_relax_factor'] = 1e-9
# ipopt.options['acceptable_tol'] = 1e-9
# ipopt.options['acceptable_constr_viol_tol'] = 1e-9
# ipopt.options['limited_memory_max_history'] = 0
# ipopt.options['max_cpu_time'] = 43200
# ipopt.options['print_level'] = 1
# ipopt.options['max_iter'] = 1000
# ipostr = 'print_level'
# ipoptpl = 2
# ipopmpl = 6
# solver = ipopt
#
# knitro = SolverFactory('knitro', executable='/usr/bin/knitroampl')
# knitro.options['feastol'] = 1e-9
# knitro.options['maxtime'] = 43200
# knitro.options['outlev'] = 0
# knitol = 0
# knimol = 3
# # solver = knitro
#
# knistr = 'print_level'
#
# # moreal = knimol
# # nowmal = knitol
# # logstr = knistr
# moreal = ipopmpl
# nowmal = ipoptpl
# logstr = ipostr

# def solvenlp(num):
#     nlp = nlpCopy()
#     dict0 = soluArr[num].soluDict
#     nlp.sigmac.value = dict0['sigmac'][0]
#     for j0, _ in naliLi:
#         key = dict0[j0].keys()
#         var = getattr(nlp, j0)
#         for k in key:
#             var[k].value = dict0[j0][k]
#     for j in bI:
#         nlp.b[j].fix(dict0['b'][j])
#     try:
#         results = solver.solve(nlp, tee=False)
#     except Exception as e:
#         print("对于初始值", num ,"求解过程中出现错误")
#         print("对于该初始值，重新计算，并输出日志")
#         try:
#             solver.options[logstr] = moreal
#             results = solver.solve(nlp, tee=True)
#         except Exception as e:
#             print("初始值求解完成，不再输出日志")
#             solver.options[logstr] = nowmal
#             return SoluA(num, soluArr[num].obj, soluArr[num].soluDict, 0, {}, 'false', 'restoration failed')
#
#     if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
#         soluDict2 = returnval(nlp, naliLi)
#         return SoluA(num, soluArr[num].obj, soluArr[num].soluDict, value(nlp.OBJ), soluDict2, 'ok', 'optimal')
#     else:
#         return SoluA(num, soluArr[num].obj, soluArr[i].soluDict, None, {}, str(results.solver.status) , str(results.solver.termination_condition))
#
# def allsolve():
#     nums = range(len0)
#     with multiprocessing.Pool(processes = int(multiprocessing.cpu_count() / 2)) as pool:
#         results = []
#         for result in tqdm(pool.imap_unordered(solvenlp, nums), total=len(nums)):
#             results.append(result)
#     return results
#
# allsolve()
# for i in tqdm(range(len0)):
#     ## 输入nlpM ,也就是原初的FBA_nlp
#     nlp = nlpCopy()
#     ## 输入dict
#     dict0 = soluArr[i].soluDict
#     nlp.sigmac.value = dict0['sigmac'][0]
#     for j0, _ in naliLi:
#         key = dict0[j0].keys()
#         var = getattr(nlp, j0)
#         for k in key:
#             var[k].value = dict0[j0][k]
#     for j in bI:
#         nlp.b[j].fix(dict0['b'][j])
#
#     # results = solver.solve(nlp, tee=True)
#     try:
#         results = solver.solve(nlp, tee=True)
#     except Exception as e:
#         print("对于初始值", i ,"求解过程中出现错误")
#         print("对于该初始值，重新计算，并输出日志")
#         print("初始值求解完成，不再输出日志")
#         solver.options[logstr] = nowmal
#         resu.append(SoluA(i, soluArr[i].obj, soluArr[i].soluDict, 0, {}, 'false', 'restoration failed'))
#         continue
#         # try:
#         #     solver.options[logstr] = moreal
#         #     results = solver.solve(nlp, tee=True)
#         # except Exception as e:
#         #     print("初始值求解完成，不再输出日志")
#         #     solver.options[logstr] = nowmal
#         #     resu.append(SoluA(i, soluArr[i].obj, soluArr[i].soluDict, 0, {}, 'false', 'restoration failed'))
#         #     continue
#     # print("for this init x0, the iter nums is", results.nit)
#     if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
#         soluDict2 = returnval(nlp, naliLi)
#         resu1 = SoluA(i, soluArr[i].obj, soluArr[i].soluDict, value(nlp.OBJ), soluDict2, 'ok', 'optimal')
#     else:
#         resu1 = SoluA(i, soluArr[i].obj, soluArr[i].soluDict, None, {}, str(results.solver.status) , str(results.solver.termination_condition))
#
#     resu.append(resu1)









