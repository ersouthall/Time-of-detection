# About

MATLAB files used to implement the maximum likelihood estimation (MLE)

# How to use

To run analysis, run `Run_MLE_changepoint.m`. Below is an example bash script demonstrating how to run:

```
#!/bin/bash

module load MATLAB/2017a
samplestep=10

realisationNUM=1
matlab -nodisplay -nodesktop -nosplash -r "Run_MLE_changepoint(${realisationNUM}, ${samplestep});exit;" 
```
