# About

The training data for the logistic classifier follows an SIS process. Null datasets start and finish with the same $R_0$ value (endemic stochastic simulations). The bifurcating dataset is slowly approaches disease elimination at a rate p = 1/500. The transmission parameter $\beta$ is changed over time forcing $R_0$ from 5 to 0.

# How to use

To run simulations, run `create_training_data.py`. Below is an example bash script demonstrating how to run:

```
#!/bin/bash

n_threads=$SLURM_CPUS_PER_TASK
python create_training_data.py 0 10000 $n_threads

```
