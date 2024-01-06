import numpy as np
import matplotlib.pyplot as plt
import Functions as f

# plot settings

plt.rcParams['figure.dpi'] = 200
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


if __name__ == "__main__":
    np.random.seed(1)

    nHouses = 5
    nWeeks = 2
    nTimeStepsMpc = 24
    nTimeSteps = 7*24*nWeeks

    basePrice = 0.2
    weekdayPriceCsvName = 'Data/normalized_prices_weekday.csv'
    weekendPriceCsvName = 'Data/normalized_prices_weekend.csv'
    solarCsvName = 'Data/normalized_solar_output.csv'

    demandNoiseMean = 0
    demandNoiseStd = 2
    standardBatteryCapacities = np.array([100, 300, 500])
    indices = np.random.choice(np.arange(len(standardBatteryCapacities)), nHouses)
    batteryCapacities = standardBatteryCapacities[indices]

    standard_solar_capacities = np.array([120, 320, 480]) # (kWh)
    solar_capacities = standard_solar_capacities[indices]
    batteryRateLimits = 0.3*batteryCapacities
    batteryCapLower = 0.2*batteryCapacities
    batteryCapUpper = 0.8*batteryCapacities

    prices = f.GetPriceData(basePrice, nWeeks, weekdayPriceCsvName, weekendPriceCsvName)
    demand = f.GetDemandData(nHouses, nWeeks, demandNoiseMean, demandNoiseStd, batteryCapacities,
                                weekdayPriceCsvName, weekendPriceCsvName)
    solar = f.GetSolarData(nHouses, nWeeks, solarCsvName, solar_capacities)
    
    epsilon = 0.5

    etaC = 0.9
    etaD = 0.8

    ## Varying parameters alpha1 and alpha2
    q = []

    alpha1 = 0
    alpha2 = 1
    qFinal, XFinal, YFinal, Zfinal, BFinal = f.MPC(demand, solar, prices, nHouses, nTimeSteps, nTimeStepsMpc, alpha1, alpha2, epsilon, batteryRateLimits, batteryCapLower, batteryCapUpper, etaC, etaD)
    q.append(qFinal)
    
    alpha1 = 1
    alpha2 = 1
    qFinal, XFinal, YFinal, Zfinal, BFinal = f.MPC(demand, solar, prices, nHouses, nTimeSteps, nTimeStepsMpc, alpha1, alpha2, epsilon, batteryRateLimits, batteryCapLower,batteryCapUpper, etaC, etaD)
    q.append(qFinal)

    alpha1 = 1
    alpha2 = 0
    qFinal, XFinal, YFinal, Zfinal, BFinal = f.MPC(demand, solar, prices, nHouses, nTimeSteps, nTimeStepsMpc, alpha1, alpha2, epsilon, batteryRateLimits, batteryCapLower, batteryCapUpper, etaC, etaD)
    q.append(qFinal)
    
    ## Plot : Varying alpha1 and alpha2
    plt.figure(figsize = (10, 4))
    for i in range(3):
        plt.step(np.linspace(0,7*24,7*24), q[i][:7*24])
    plt.legend([r'$\alpha_1$ = 0, $\alpha_2$ = 1',
                r'$\alpha_1$ = 1, $\alpha_2$ = 1',
                r'$\alpha_1$ = 1, $\alpha_2$ = 0'], loc = 'lower right')
    plt.xlabel('Time (hr)')
    plt.xlim([0, 7*24])
    plt.ylabel('Grid supply (kW)')
    plt.savefig('Assets/AlphaParameterVariations.png')
    
    ## Open Loop vs MPC comparison
    
    q = []

    alpha1 = 1
    alpha2 = 1
    
    # open-loop
    qFinal, XFinal, YFinal, ZFinal, BFinal = f.MPC(demand, solar, prices, nHouses, nTimeSteps, nTimeSteps-1, alpha1, alpha2, epsilon, batteryRateLimits, batteryCapLower, batteryCapUpper, etaC, etaD)
    q.append(qFinal)

    # mpc
    qFinal, XFinal, YFinal, ZFinal, BFinal = f.MPC(demand, solar, prices, nHouses, nTimeSteps, nTimeStepsMpc, alpha1, alpha2, epsilon, batteryRateLimits, batteryCapLower, batteryCapUpper, etaC, etaD)
    q.append(qFinal)
    
    ## Plot : MPC vs Open loop
    plt.figure(figsize = (10, 4))
    for i in range(2):
        plt.step(np.linspace(0,7*24,7*24), q[i][:7*24])
    plt.legend(['Open-loop', 'Closed-loop MPC'], loc = 'lower right')
    plt.xlabel('Time (hr)')
    plt.xlim([0, 7*24])
    plt.ylabel('Grid supply (kW)')
    plt.savefig('Assets/openLoopVsMPC.png')