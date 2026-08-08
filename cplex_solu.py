from cplex import Cplex
import re
class Solu:
    def __init__(self, obj, soluDict):
        self.obj = obj
        self.soluDict = soluDict

pattern = re.compile(r'([A-Za-z_]+)(?:\(([\d_]+)\))?')
# 将如下定义为一个函数，直接调用，返回的
# Create a new Cplex object
def solvemip(lpfile, tee=True):
    cpx = Cplex()
    # Read the LP model file
    cpx.read(lpfile)
    if tee == True:
        cpx.parameters.mip.display.set(4)
    else:
        cpx.parameters.mip.display.set(0)
    cpx.parameters.timelimit.set(54000)
    cpx.parameters.tune.timelimit.set(10800)
    cpx.parameters.threads.set(1)
    cpx.parameters.parallel.set(-1)
    cpx.parameters.emphasis.memory.set(1)
    cpx.parameters.mip.tolerances.mipgap.set(1e-9)

    cpx.parameters.mip.tolerances.integrality.set(1e-9)


    cpx.parameters.mip.tolerances.absmipgap.set(1e-9)


    cpx.parameters.mip.strategy.startalgorithm.set(1)
    cpx.parameters.simplex.limits.iterations.set(2000000000)

    cpx.parameters.simplex.tolerances.feasibility.set(1e-9)


    cpx.parameters.simplex.tolerances.optimality.set(1e-9)
    cpx.parameters.mip.pool.capacity.set(100)
    cpx.parameters.mip.pool.replace.set(2)
    cpx.parameters.mip.pool.intensity.set(4)
    cpx.parameters.mip.limits.populate.set(1000)

    cpx.solve()

    cpx.populate_solution_pool()

    soln = cpx.solution.pool.get_num()
    print("number of solutions in the solution pool: ", soln)

    resu = []
    varN = cpx.variables.get_names()
    parA = []
    for i in varN:
        match = pattern.search(i)
        if match.group(2) == None:
            parA.append((match.group(1),None))
        else:
            parA.append((match.group(1),(match.group(2))))

    for i in range(soln):
        inter = {}
        inter['lnc'] = {}
        obj = cpx.solution.pool.get_objective_value(i)
        varV= cpx.solution.pool.get_values(i)
        for i in range(len(varN)):
            na, iD = parA[i]
            if (not (na in inter.keys())) & (na != 'lnc'):
                inter1 = {}
                if iD == None:
                    inter1[0] = varV[i]
                else:
                    inter1[int(iD)] = varV[i]
                inter[na] = inter1
            elif (na != 'lnc'):
                inter[na][int(iD)] = varV[i]
            else:
                iD0, iD1 = iD.split('_')
                inter['lnc'][(int(iD0), int(iD1))] = varV[i]
        resu.append(Solu(obj, inter))

    return resu



