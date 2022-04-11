import numpy as np 
import pandas as pd

def create_simulation_parameters(folder, params):
    R0_i = params["beta0"]/params["gamma"]
    repeats = params["realisations"]
    ext_simulation_index = np.random.choice(repeats, repeats//2,
                                            replace = False)
    simulation_parameters = pd.DataFrame({"R0_i": R0_i*np.ones(repeats),
                                          "R0_f": R0_i*np.ones(repeats)})
    simulation_parameters.loc[ext_simulation_index, "R0_f"] = 0
    simulation_parameters.to_csv(folder + "/simulation_parameters.csv")
    
def betachange(time, params):
    betat = (1-params["p"]*(time - params["BurnTime"]))*params["beta0"]
    return betat

def betafix(time, params):
    betat = params["beta0"]
    return betat

def single_simulation(folder_output, params,simulation_params, simulation_number):
    print("simulaton:", simulation_number)
    if simulation_params.loc[simulation_number, 'R0_f'] == \
        simulation_params.loc[simulation_number, 'R0_i']:
        print("fixed simulation")
        gillespie_output = gillespieSIS(params = params,
                                        function = betafix)
    elif simulation_params.loc[simulation_number, 'R0_f']<1:
        print('elimination simulation')
        gillespie_output = gillespieSIS(params = params,
                                        function = betachange)
    gillespie_results_dict = linear_interpolate(gillespieOutput = gillespie_output,
                        params=params)
    
    df_delta_timestep = pd.DataFrame(gillespie_results_dict[0])
    df_delta_reindex = df_delta_timestep.set_index('time')
    df_daily = pd.DataFrame(gillespie_results_dict[1]).reindex(np.arange(0,\
        (params['Time']+params['BurnTime']),0.1), fill_value = 0)
    result = pd.merge(df_delta_reindex, df_daily, left_index = True, right_index = True)
    result['Time'] = np.arange(0, (params['Time']+params['BurnTime']), 0.1)
    result.to_csv(folder_output+"data_"+str(simulation_number)+".csv")
        
def gillespieSIS(params, function):
    np.random.seed()
    initial = [0.2*params["N"], 0.8*params["N"]]
    T = []
    pop = []
    N = sum(initial)
    pop.append(initial)
    newCase = np.zeros(params['Time']+params['BurnTime']+10)
    newCase[0] =0
    R0 = []
    R0.append(params['beta0']/params['gamma'])
    T.append(0)
    t = 0
    ind = 0
    state = np.zeros(shape= (2,2))
    rate = np.zeros(2)
    state[:,0] = [-1, 1]
    state[:,1] = [1, -1]
    R1 = params['beta0']*(pop[ind][0])*(pop[ind][1])/N
    R2 = params['gamma']*(pop[ind][1])
    rate[0] = R1
    rate[1] = R2
    while t <(params['Time']+params['BurnTime']):
        if t<params['BurnTime']:
            betat = params['beta0']
        else:
            betat = function(time = t, 
                             params = params)
            # print("beta:", betat)
        Rtotal = sum(rate)
        if Rtotal >0:
            delta_t= -np.log(np.random.uniform(0,1))/Rtotal

            P = np.random.uniform(0,1)*Rtotal
            t =t+ delta_t
            event = np.min(np.where(P<=np.cumsum(rate)))
            T.append(t)
            R0.append(betat/params['gamma'])
            if event == 0:
                newCase[int(np.ceil(t))] +=1 
            pop.append(pop[ind]+state[:,event])
            ind=ind+1
            rate[0] = betat*(pop[ind][0])*(pop[ind][1])/N
            rate[1] = params["gamma"]*(pop[ind][1])
        else: 
            t = (params['Time']+params['BurnTime'])
            T.append(t)
            pop.append(pop[ind])
            R0.append(betat/params["gamma"])
    return T, np.array(pop), newCase, R0


def linear_interpolate(gillespieOutput, params):
    t = gillespieOutput[0]
    s = gillespieOutput[1][:,0]
    i = gillespieOutput[1][:,1]
    NC = gillespieOutput[2][:(params['Time']+params['BurnTime'])]
    R0 = gillespieOutput[3]
    stept = []
    steps = []
    stepi = []
    stepr0 = []
    for ind, x in enumerate(t):
        if ind<len(t)-1:
            steps.append((s[ind], s[ind]))
            stepi.append((i[ind], i[ind]))
            stepr0.append((R0[ind], R0[ind]))
            stept.append((t[ind], t[ind+1]))
        else:
            steps.append((s[ind], s[ind]))
            stepi.append((i[ind], i[ind]))
            stept.append((t[ind], t[ind]))
            stepr0.append((R0[ind], R0[ind]))
            

    steps = np.array(steps).flatten()
    stepi = np.array(stepi).flatten()
    stept = np.array(stept).flatten()
    stepr0 = np.array(stepr0).flatten()
    
    
    #### linear interpolation
    def interpolation_method(time, var):
        inter_t = np.arange(0, round(max(time))+1, 0.1)
        inter_var = np.interp(inter_t, time, var)
        return inter_var[:(params['Time']+params['BurnTime'])*10]
    dict_return_cts = {'time': np.arange(0, params['Time']+params['BurnTime'], 0.1),
                       'S':interpolation_method(stept, steps),
                       'I': interpolation_method(stept, stepi),
                       'R0': interpolation_method(stept, stepr0)}
    
    return [dict_return_cts, {'cases': NC}]
