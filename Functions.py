import numpy as np
import cvxpy as cp
from tqdm.auto import tqdm

def GetPriceData(basePrice, nWeeks, weekdayCsvName, weekendCsvName):
    #basePrice : float
    #nWeeks : int
    #weekdayCsvName : string
    #weekendCsvName : string

    tSteps = int(7*24*nWeeks)

    normalizedWeekdayPrices = np.loadtxt(weekdayCsvName, delimiter=",")
    normalizedWeekendPrices = np.loadtxt(weekendCsvName, delimiter=",")

    prices = np.zeros(tSteps)
    pTemp = []
    for _ in range(nWeeks):
        for _ in range(5):
            pTemp.append(basePrice*normalizedWeekdayPrices[:, 1])
        for _ in range(2):
            pTemp.append(basePrice*normalizedWeekendPrices[:, 1])
    prices = np.array(pTemp).flatten()
    return prices

def GetDemandData(nHouses, nWeeks, noiseMean, noiseStd, batteryCapacities, weekdayCsvName, weekendCsvName):
    #nHouses : int
    #nWeeks : int
    #noiseMean : float
    #noiseStd : float
    # batteryCapacities : list of floats (n x 1)
    #weekdayCsvName : string
    #weekendCsvName : string

    tSteps = int(7*24*nWeeks)

    normalizedWeekdayDemand = np.loadtxt(weekdayCsvName, delimiter=",")
    normalizedWeekendDemand = np.loadtxt(weekendCsvName, delimiter=",")

    demand = np.zeros((tSteps, nHouses))
    for i in range(nHouses):
        baseDemand = batteryCapacities[i]
        dTemp = []
        for _ in range(nWeeks):
            for _ in range(5):
                dTemp.append(
                    baseDemand*normalizedWeekdayDemand[:, 1] + np.random.normal(noiseMean, noiseStd, 24))
            for _ in range(2):
                dTemp.append(
                    baseDemand*normalizedWeekendDemand[:, 1] + np.random.normal(noiseMean, noiseStd, 24))
        demand[:, i] = np.array(dTemp).flatten()
    return demand

def GetSolarData(nHouses, nWeeks, solarCsvName, solarCapacities):
    #n : int
    #W : int
    #solarCsvName : string

    nTimeSteps = int(7*24*nWeeks)

    normalizedSolarSupply = np.loadtxt(solarCsvName)
    solarPower = np.zeros((nTimeSteps, nHouses))

    for i in range(nHouses):
        base_power = solarCapacities[i]
        power = []
        for _ in range(nWeeks):
            for _ in range(7):
                temp_power = (base_power * normalizedSolarSupply[:,1])
                for j in range(24):
                    if temp_power[j] != 0:
                        temp_power[j] += 5*np.random.randn(1)
                        temp_power[j] = max(0, temp_power[j])
                power.append(temp_power)
        power = np.array(power)
        power = power.reshape((-1))
        solarPower[:, i] = power
        
    return solarPower

def ConvexProgram(nHouses, nTimeStepsMpc, demandMpc, solarMpc, prices, alpha1, alpha2, bInit, epsilon, batteryRateLimits,
                  batteryCapLower, batteryCapUpper, etaC, etaD):
    #nHouses : int
    #tStepsMpc : int
    #demandMpc : floats (tStepsMpc x n)
    #solarMpc : floats (tStepsMpc x n)
    #prices : floats (tStepsMpc x 1)
    #alpha : float
    #bInit : floats (n x 1)

    slope = np.minimum(etaC, 1/etaD)

    q_var = cp.Variable(nTimeStepsMpc)
    X_var = cp.Variable((nTimeStepsMpc, nHouses))
    Y_var = cp.Variable((nTimeStepsMpc, nHouses))
    Z_var = cp.Variable((nTimeStepsMpc, nHouses))
    B_var = cp.Variable((nTimeStepsMpc + 1, nHouses))

    objective = cp.Minimize(alpha1*cp.norm_inf(q_var) + alpha2*(prices@q_var))

    constraints = [B_var[0, :] == bInit, q_var == cp.sum(X_var, axis=1)]

    for i in range(nTimeStepsMpc):
        constraints = constraints + [X_var[i, :] >= 0,
                                     Z_var[i,:] >= 0,
#                                      Z_var[i, :] >= -epsilon, Z_var[i, :] <= epsilon,
                                     Y_var[i, :] >= -batteryRateLimits, Y_var[i, :] <= batteryRateLimits,
                                     B_var[i + 1, :] >= batteryCapLower, B_var[i + 1, :] <= batteryCapUpper,
                                     X_var[i, :] + Y_var[i, :] - Z_var[i, :] == demandMpc[i, :] - solarMpc[i, :],
                                     B_var[i + 1, :] == B_var[i, :] - slope*Y_var[i, :]]


    problem = cp.Problem(objective, constraints)
    value = problem.solve()
    if problem.status != 'optimal':
        raise RuntimeError('SCP solve failed. Problem status: ' + problem.status)

    return [q_var.value, X_var.value, Y_var.value, Z_var.value, B_var.value, value]

def MPC(demand, solar, prices, nHouses, nTimeSteps, nTimeStepsMpc, 
        alpha1, alpha2, epsilon, batteryRateLimits, batteryCapLower,
        batteryCapUpper, etaC, etaD):

    qMpc = np.zeros((nTimeSteps, nTimeStepsMpc))
    XMpc = np.zeros((nTimeSteps, nTimeStepsMpc, nHouses))
    YMpc = np.zeros((nTimeSteps, nTimeStepsMpc, nHouses))
    ZMpc = np.zeros((nTimeSteps, nTimeStepsMpc, nHouses))
    BMpc = np.zeros((nTimeSteps, nTimeStepsMpc + 1, nHouses))
    valueMpc = np.zeros(nTimeSteps)

    bInit = np.copy(batteryCapUpper)
    for t in tqdm(range(nTimeSteps - nTimeStepsMpc)):
        demandMpc = demand[t:t + nTimeStepsMpc, :]
        solarMpc = solar[t:t + nTimeStepsMpc, :]
        pricesMpc = prices[t:t + nTimeStepsMpc]


        [qMpc[t, :], XMpc[t, :, :], YMpc[t, :, :],
         ZMpc[t, :, :], BMpc[t, :, :], valueMpc[t]] = ConvexProgram(nHouses, nTimeStepsMpc, demandMpc, solarMpc, pricesMpc, 
                                                                    alpha1, alpha2, bInit, epsilon, batteryRateLimits, batteryCapLower,
                                                                    batteryCapUpper, etaC, etaD)

        # True battery state
        for i in range(nHouses):
            bInit[i] = bInit[i] - np.maximum(etaC*YMpc[t, 0, i], YMpc[t, 0, i]/etaD)
    
    qFinal = qMpc[:, 0]
    XFinal = XMpc[:, 0, :]
    YFinal = YMpc[:, 0, :]
    ZFinal = ZMpc[:, 0, :]
    BFinal = BMpc[:, 0, :]
    
    return qFinal, XFinal, YFinal, ZFinal, BFinal
    



