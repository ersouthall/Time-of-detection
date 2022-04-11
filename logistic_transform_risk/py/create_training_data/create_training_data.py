import sys
sys.path.append("..")
import helper as h
import gillespie_SIS as SIS
from multiprocessing import Pool

# Run Gillespie simulations. Use simulation_parameters.csv to give parameter choices for each realisation
# Repeat for realisations start to finish
start = int(sys.argv[1])
finish = int(sys.argv[2])
n_core = int(sys.argv[3]) #number of threads for parallel

print("running simulations from", str(start), "to", str(finish))
print("using", str(n_core), "cores")
f_root = "../../data"
f_output = "/storage/"
params = h.simulator_args()

# Either open or create simulation csv file (each row in file gives the initial R0 and final R0 for each simulation)
try:
    simulation_parameters = pd.read_csv(f_root+'/simulation_parameters.csv')
except:
    SIS.create_simulation_parameters(folder = f_root,
                                 params= h.simulator_args())
    
simulation_parameters = pd.read_csv(f_root + "/simulation_parameters.csv")


print("run simulations")
def parallel_process(run):
    SIS.single_simulation(
                      folder_output = f_output,
                      params = h.simulator_args(),
                      simulation_params = simulation_parameters,
                      simulation_number = run) 
num_threads = n_core
with Pool(num_threads) as pool:
    results = pool.map(parallel_process, range(start, finish))
    