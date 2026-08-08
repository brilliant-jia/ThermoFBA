import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory

ipopt = SolverFactory('ipopt')
ipopt.options['bound_relax_factor'] = 1e-9
# ipopt.options['acceptable_tol'] = 1e-9
# ipopt.options['acceptable_constr_viol_tol'] = 1e-9
# ipopt.options['limited_memory_max_history'] = 0
ipopt.options['constr_viol_tol'] = 1e-9
#ipopt.options['constraint_violation_norm_type'] = 'max-norm'
ipopt.options['max_cpu_time'] = 43200
ipopt.options['print_level'] = 1
ipopt.options['max_iter'] = 1000
ipopt.options['mu_strategy'] = 'adaptive'
ipopt.options['honor_original_bounds'] = 'yes'
#ipopt.options['pardiso_max_iterative_refinement_steps'] = 1
ipostr = 'print_level'
ipoptpl = { ipostr : 3}
ipopmpl = { ipostr : 6}
ipopOpDi = {}
ipopOpDi['on'] = ipopmpl
ipopOpDi['off']= ipoptpl

# solver = ipopt

knitro = SolverFactory('knitro', executable='/usr/bin/knitroampl')
knitro.options['feastol'] = 1e-9
knitro.options['maxtime'] = 43200
knitro.options['outlev'] = 0
knistr = 'outlev'
knitol = {knistr : 0}
knimol = {knistr : 3}
kniOpDi = {}
kniOpDi['on'] = knimol
kniOpDi['off'] = knitol

solverDict = {}
solverDict['ipopt'] = (ipopt, ipopOpDi)
solverDict['knitro'] = (knitro, kniOpDi)
def nlpsolvers(name):
    if name not in solverDict.keys():
        print("no such solver")
    else:
        return solverDict[name]
# solver = knitro



# moreal = knimol
# nowmal = knitol
# logstr = knistr
# moreal = ipopmpl
# nowmal = ipoptpl
# logstr = ipostr
