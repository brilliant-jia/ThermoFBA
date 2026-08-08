from cplex_solu import solvemip
import csv

from tqdm import tqdm
#from model import returnval, nlpIter, FBA_mip, SoluA, naliLi, make
from model import make, SoluA
from nlpsolver import nlpsolvers
import pyomo.environ as pyo
from pyomo.environ import *
import pickle
from pyomo.opt import SolverFactory
modeldir = './'

#scondlist=['EX_fru','EX_gal','EX_glcn','EX_glc','EX_glyc','EX_pyr','EX_succ']
scondlist=['EX_glyc','EX_pyr','EX_succ']

for l in scondlist:
    returnval, nlpIter, FBA_mip, naliLi = make(l)
    filename = f'fba_mip_{l}.lp'
    FBA_mip.write(modeldir + filename, io_options={'symbolic_solver_labels': True})
    soluArr = []
    #for i in range(1,1001):
    #    soluArr.append(toSolu('./1.25/'+prefix+str(i)+tailed))
    soluArr = solvemip(modeldir + filename, tee=False)
    nlpIt = nlpIter(soluArr)

    len0 = len(soluArr)
    # 假定solvers 可以返回两个一个是solver, 另一个是optiondict, 后者是为了输出求解错误时给出全部信息，dict包含两部分，分别是key 为'on''off'的两个字典，而solver里面对应的参数默认均在off上
    solver, optiondict = nlpsolvers('ipopt')
    # rerun = True
    rerun = False
    resu = []
    for i in tqdm(range(len0)):
        # nlp = nlpArr[i]
        nlp = next(nlpIt)
        try:
            results = solver.solve(nlp, tee=True)
        except Exception as e:
            print("对于初始值", i, "求解过程中出现错误")
            print("对于该初始值，重新计算，并输出日志")
            if rerun:
                try:
                    for i in optiondict['on'].keys():
                        solver.option[i] = optiondict['on'][i]
                    results = solver.solve(nlp, tee=True)
                except Exception as e:
                    print("初始值求解完成，不再输出日志")
                    for i in optiondict['off'].keys():
                        solver.option[i] = optiondict['off'][i]
            # solver.options[logstr] = nowmal
            resu.append(SoluA(i, soluArr[i].obj, soluArr[i].soluDict, 0, {}, 'false', 'restoration failed'))
            continue

        soluDict2 = returnval(nlp, naliLi)
        resu1 = SoluA(i, soluArr[i].obj, soluArr[i].soluDict, value(nlp.OBJ), soluDict2, str(results.solver.status),
                      str(results.solver.termination_condition))
        resu.append(resu1)
        '''
        if (results.solver.status == SolverStatus.ok) and (
                results.solver.termination_condition == TerminationCondition.optimal):
            soluDict2 = returnval(nlp, naliLi)
            resu1 = SoluA(i, soluArr[i].obj, soluArr[i].soluDict, value(nlp.OBJ), soluDict2, 'ok', 'optimal')
        if (results.solver.status == SolverStatus.ok) and (
                results.solver.termination_condition == TerminationCondition.maxIterations):
            soluDict2 = returnval(nlp, naliLi)
            resu1 = SoluA(i, soluArr[i].obj, soluArr[i].soluDict, value(nlp.OBJ), soluDict2, 'ok', 'maxIterations')
        else:
            resu1 = SoluA(i, soluArr[i].obj, soluArr[i].soluDict, None, {}, str(results.solver.status),
                          str(results.solver.termination_condition))
        '''
        resu.append(resu1)

    filename = f'output_{l}.pkl'  # CSV 文件的文件名

    with open(filename, 'wb') as file:
        pickle.dump(resu, file)
'''

l=7
returnval, nlpIter, FBA_mip, naliLi = make(l+0.25)
filename = f'fba_mip_{l+0.25}.lp'
FBA_mip.write(modeldir + filename, io_options={'symbolic_solver_labels': True})
soluArr = []
for i in range(1,1001):
    soluArr.append(toSolu('./1.25/'+prefix+str(i)+tailed))
#soluArr = solvemip(modeldir + filename, tee=False)
nlpIt = nlpIter(soluArr)

len0 = len(soluArr)
# 假定solvers 可以返回两个一个是solver, 另一个是optiondict, 后者是为了输出求解错误时给出全部信息，dict包含两部分，分别是key 为'on''off'的两个字典，而solver里面对应的参数默认均在off上
solver, optiondict = nlpsolvers('ipopt')
# rerun = True
rerun = False
resu = []
for i in tqdm(range(len0)):
    # nlp = nlpArr[i]
    nlp = next(nlpIt)
    try:
         results = solver.solve(nlp, tee=True)
    except Exception as e:
            print("对于初始值", i, "求解过程中出现错误")
            print("对于该初始值，重新计算，并输出日志")
            if rerun:
                try:
                    for i in optiondict['on'].keys():
                        solver.option[i] = optiondict['on'][i]
                    results = solver.solve(nlp, tee=True)
                except Exception as e:
                    print("初始值求解完成，不再输出日志")
                    for i in optiondict['off'].keys():
                        solver.option[i] = optiondict['off'][i]
            # solver.options[logstr] = nowmal
            resu.append(SoluA(i, soluArr[i].obj, soluArr[i].soluDict, 0, {}, 'false', 'restoration failed'))
            continue

    soluDict2 = returnval(nlp, naliLi)
    resu1 = SoluA(i, soluArr[i].obj, soluArr[i].soluDict, value(nlp.OBJ), soluDict2, str(results.solver.status), str(results.solver.termination_condition))
    resu.append(resu1)

    if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal):
        soluDict2 = returnval(nlp, naliLi)
        resu1 = SoluA(i, soluArr[i].obj, soluArr[i].soluDict, value(nlp.OBJ), soluDict2, 'ok', 'optimal')
    if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.maxIterations):
        soluDict2 = returnval(nlp, naliLi)
        resu1 = SoluA(i, soluArr[i].obj, soluArr[i].soluDict, value(nlp.OBJ), soluDict2, 'ok', 'maxIterations')
    else:
        resu1 = SoluA(i, soluArr[i].obj, soluArr[i].soluDict, None, {}, str(results.solver.status),
                      str(results.solver.termination_condition))

 #   resu.append(resu1)

filename = f'output_{l+0.25}.pkl'  # CSV 文件的文件名

with open(filename, 'wb') as file:
    pickle.dump(resu, file)
    
'''