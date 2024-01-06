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

    alpha1 = 1
    alpha2 = 1
    epsilon = 0.5

    etaC = 0.9
    etaD = 0.8
    
    qFinal, XFinal, YFinal, Zfinal, BFinal = f.MPC(demand, solar, prices, nHouses, nTimeSteps, nTimeStepsMpc, 
                                             alpha1, alpha2, epsilon, batteryRateLimits, batteryCapLower,
                                             batteryCapUpper, etaC, etaD)

    ### Plots

    ## Input Data
    fig = plt.figure(figsize = (10, 4), dpi = 200)
    ax = fig.add_subplot(111)
    t = np.linspace(1, nTimeSteps, nTimeSteps)
    for i in range(1):
        ax.step(t, demand[:,i], '-', color='tab:red', label = 'House #' + str(i+1), linewidth=2)

    ax2 = ax.twinx()
    ax2.step(t, prices, '--', color='blueviolet', label = 'Prices')

    ax.set_xlabel('Time (hr)')
    ax.set_ylabel('Demand (kW)', color = 'tab:red')
    ax2.set_ylabel('Prices ($/kWh)', color = 'blueviolet')
    ax.set_xlim([0, nTimeSteps/2])
    plt.savefig('Demand_Prices.png')
    
    ## House Energy Demands
    fig, axs = plt.subplots(2, 2, figsize = (10,5), dpi = 200)
    
    # During Weekdays
    start = 0
    end = 2*24

    t = np.linspace(start, end, end-start)
    i = 0 # House #1
    axs[0,0].step(t, demand[start:end,i], '-', color='orange', label = 'Demand')
    axs[0,0].step(t, solar[start:end,i], '--g', label = 'Solar power')
    axs[0,0].step(t, YFinal[start:end,i], '-.b', label = 'Battery power')
    axs[0,0].step(t, batteryRateLimits[i]*np.ones(len(t)), '--', color = 'gray', label = 'Battery charge/discharge limit')
    axs[0,0].step(t, -batteryRateLimits[i]*np.ones(len(t)), '--', color = 'gray',)
    axs[0,0].legend(ncol=4, bbox_to_anchor=(2.3, -1.5), loc='upper right')
    axs[0,0].set_xlim([start, end])

    i = 2 # House #3
    axs[1,0].step(t, demand[start:end,i], '-', color='orange', label = 'Demand')
    axs[1,0].step(t, solar[start:end,i], '--g', label = 'Solar power')
    axs[1,0].step(t, YFinal[start:end,i], '-.b', label = 'Battery level')
    axs[1,0].step(t, batteryRateLimits[i]*np.ones(len(t)), '--', color = 'gray', label = 'Battery charge/discharge limit')
    axs[1,0].step(t, -batteryRateLimits[i]*np.ones(len(t)), '--', color = 'gray',)
    axs[1,0].set_xlim([start, end])

    # ======================================================================
    # During Weekends
    start = 5*24 
    end = 7*24

    t = np.linspace(start, end, end-start)
    i = 0 # House #1
    axs[0,1].step(t, demand[start:end,i], '-', color='orange', label = 'Demand')
    axs[0,1].step(t, solar[start:end,i], '--g', label = 'Solar power')
    axs[0,1].step(t, YFinal[start:end,i], '-.b', label = 'Battery power')
    axs[0,1].step(t, batteryRateLimits[i]*np.ones(len(t)), '--', color = 'gray', label = 'Battery charge/discharge limit')
    axs[0,1].step(t, -batteryRateLimits[i]*np.ones(len(t)), '--', color = 'gray',)
    axs[0,1].set_xlim([start, end])

    i = 2 # House #3
    axs[1,1].step(t, demand[start:end,i], '-', color='orange', label = 'Demand')
    axs[1,1].step(t, solar[start:end,i], '--g', label = 'Solar power')
    axs[1,1].step(t, YFinal[start:end,i], '-.b', label = 'Battery level')
    axs[1,1].step(t, batteryRateLimits[i]*np.ones(len(t)), '--', color = 'gray', label = 'Battery charge/discharge limit')
    axs[1,1].step(t, -batteryRateLimits[i]*np.ones(len(t)), '--', color = 'gray',)
    axs[1,1].set_xlim([start, end])

    for ax in fig.get_axes():
        ax.label_outer()

    axs[1,0].set_xlabel('Time (hr)')
    axs[1,1].set_xlabel('Time (hr)')
    axs[0,0].set_ylabel('Power (kW)')
    axs[1,0].set_ylabel('Power (kW)')
    ax2.set_ylabel('Energy (kWh)')

    axs[0,0].set_title('Weekday, House #1')
    axs[0,1].set_title('Weekend, House #1')
    axs[1,0].set_title('Weekday, House #3')
    axs[1,1].set_title('Weekend, House #3')

    plt.savefig('HouseEnergyDemands.png')
    
    ## Battery Dynamics
    fig, axs = plt.subplots(2,2, figsize = (10,5), dpi = 200)

    # During Weekdays
    start = 0
    end = 2*24

    t = np.linspace(start, end, end-start)
    i = 0 # House #1
    axs[0,0].step(t, BFinal[start:end,i], '-m', label = 'Battery energy level')
    axs[0,0].step(t, batteryCapUpper[i]*np.ones(len(t)), '--', color = 'm', label = 'Battery limit', alpha = 0.2)
    axs[0,0].step(t, min(BFinal[start:end,i])*np.ones(len(t)), '--', color = 'm', alpha = 0.2)
    axs[0,0].set_xlim([start, end])
    ax2 = axs[0,0].twinx()
    ax2.step(t, YFinal[start:end,i], '-.b', label = 'Battery power')
    ax2.step(t, batteryRateLimits[i]*np.ones(len(t)), '--', color = 'b', label = 'Battery charge/discharge limit', alpha = 0.2)
    ax2.step(t, -batteryRateLimits[i]*np.ones(len(t)), '--', color = 'b', alpha = 0.2)
    ax2.set_ylim([-100, 300])

    i = 2 # House #3
    axs[1,0].step(t, BFinal[start:end,i], '-m', label = 'Battery energy level')
    axs[1,0].step(t, batteryCapUpper[i]*np.ones(len(t)), '--', color = 'm', label = 'Battery limit', alpha = 0.2)
    axs[1,0].step(t, min(BFinal[start:end,i])*np.ones(len(t)), '--', color = 'm', alpha = 0.2)
    axs[1,0].set_xlim([start, end])
    ax2 = axs[1,0].twinx()
    ax2.step(t, YFinal[start:end,i], '-.b', label = 'Battery power')
    ax2.step(t, batteryRateLimits[i]*np.ones(len(t)), '--', color = 'b', label = 'Battery charge/discharge limit', alpha = 0.2)
    ax2.step(t, -batteryRateLimits[i]*np.ones(len(t)), '--', color = 'b', alpha = 0.2)
    ax2.set_ylim([-50, 100])

    # ======================================================================
    # During Weekends
    start = 5*24
    end = 7*24

    t = np.linspace(start, end, end-start)
    i = 0 # House #1
    axs[0,1].step(t, BFinal[start:end,i], '-m', label = 'Battery energy level')
    axs[0,1].step(t, batteryCapUpper[i]*np.ones(len(t)), '--', color = 'm', label = 'Battery limit', alpha = 0.2)
    axs[0,1].step(t, min(BFinal[start:end,i])*np.ones(len(t)), '--', color = 'm', alpha = 0.2)
    axs[0,1].set_xlim([start, end])
    ax2 = axs[0,1].twinx()
    ax2.step(t, YFinal[start:end,i], '-.b', label = 'Battery power')
    ax2.step(t, batteryRateLimits[i]*np.ones(len(t)), '--', color = 'b', label = 'Battery charge/discharge limit', alpha = 0.2)
    ax2.step(t, -batteryRateLimits[i]*np.ones(len(t)), '--', color = 'b', alpha = 0.2)
    ax2.set_ylabel('Battery power (kW)', color='b')
    ax2.set_ylim([-100, 300])

    i = 2 # House #3
    axs[1,1].step(t, BFinal[start:end,i], '-m', label = 'Battery energy level')
    axs[1,1].step(t, batteryCapUpper[i]*np.ones(len(t)), '--', color = 'm', label = 'Battery limit', alpha = 0.2)
    axs[1,1].step(t, min(BFinal[start:end,i])*np.ones(len(t)), '--', color = 'm', alpha = 0.2)
    axs[1,1].set_xlim([start, end])
    ax2 = axs[1,1].twinx()
    ax2.step(t, YFinal[start:end,i], '-.b', label = 'Battery power')
    ax2.step(t, batteryRateLimits[i]*np.ones(len(t)), '--', color = 'b', label = 'Battery charge/discharge limit', alpha = 0.2)
    ax2.step(t, -batteryRateLimits[i]*np.ones(len(t)), '--', color = 'b', alpha = 0.2)
    ax2.set_ylabel('Battery power (kW)', color='b')
    ax2.set_ylim([-50, 100])

    for ax in fig.get_axes():
        ax.label_outer()

    axs[1,0].set_xlabel('Time (hr)')
    axs[1,1].set_xlabel('Time (hr)')
    axs[0,0].set_ylabel('Battery energy (kWh)', color='m')
    axs[1,0].set_ylabel('Battery energy (kWh)', color='m')

    axs[0,0].set_ylim([-100, 300])
    axs[0,1].set_ylim([-100, 300])
    axs[1,0].set_ylim([-100, 300])
    axs[1,1].set_ylim([-100, 300])

    axs[0,0].set_ylim([-100, 300])
    axs[0,1].set_ylim([-100, 300])
    axs[1,0].set_ylim([-50, 100])
    axs[1,1].set_ylim([-50, 100])

    axs[0,0].set_title('Weekday, House #1')
    axs[0,1].set_title('Weekend, House #1')
    axs[1,0].set_title('Weekday, House #3')
    axs[1,1].set_title('Weekend, House #3')

    plt.savefig('BatteryDynamics.png')