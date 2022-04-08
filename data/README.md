# About

Python files to create the synthetic data for the simulation study.
- EXT: disease extinction, $R_0$ reduces from 5 to 0 by slowly decreasing $\beta$ at a rate $1/500$. This is an example of a bifurcating simulation.
- NEXT: disease <em>not</em> extinction (also called "fixchanging" in code). As in EXT, but $\beta$ stops changing when $R_0 = 1.3$. $R_0$ stays fixed at this value for the rest of the simulation. This is an example of a null simulation. 
- FIX: endemic disease. Parameters do not change in time and are fixed throughout the simulation. This is an example of a null simulation. 

Common parameters shared in all simulations:
| **_Parameter_** | **_Description_**                                                     | **_Value_** |
|-----------------|-----------------------------------------------------------------------|-------------|
| $\beta_0$       | Initial value of $\beta$                                              | 1           |
| $p$             | Rate of change of $\beta(t) = \beta_0 (1-pt)$ (for EXT and NEXT)      | 1/500       |
| $\gamma$        | Recovery Rate                                                         | 0.2         |
| (S(0), I(0))    | Initial conditions                                                    | (0.2, 0.8)  |
| N               | Total population size                                                 | 10,000      |
| BT              | Burn time (run models to initialize, without changing any parameters) | 300         |
| T               | Time period (how long each model is run for after BT)                 | 500         |

# How to use

To run the simulations and generate the synthetic data, run `.py`. Below is an example bash script demonstrating how to run:

```
#!/bin/bash

python .py

```