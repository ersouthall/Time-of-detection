# About

Python files used to implement the changing p-value method, introduced by Harris, 2020

# How to use

To run analysis to get the p-value time-series, run `run_changing_pvalue.py`. Below is an example bash script demonstrating how to run:

```
#!/bin/bash

for TIME in 20 50 100 250; do
    python run_changing_pvalue.py  ${TIME}
done
```