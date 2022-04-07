# About

MATLAB files used to implement the Quickest Detection method, introduced by Shiryaev & Roberts, 1961

** Need to add consecutive files
# How to use

To run analysis, run `quickest_detection.m`. Below is an example bash script demonstrating how to run:

```
#!/bin/bash

module load MATLAB/2017a
for STEPS in 2 5 10 25; do

    matlab -nodisplay -nodesktop -nosplash -r "quickest_detection(${STEPS});exit;" 
done
```


