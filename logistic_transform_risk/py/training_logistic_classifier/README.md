# About

# How to use

To run logistic classifier, run `run_log_classifier.py`. Below is an example bash script demonstrating how to run:

```
#!/bin/bash

python run_log_classifier.py

```

To run logistic classifier for each unique combination of EWSs (from a list of 500 in `data/training_signals.npy`), run `run_logistic_EWSscombination.py`. Below is an example bash script demonstrating how to run:
```
#!/bin/bash

for i in {1..500}; do 
    python run_logistic_EWSscombination.py $i
done 


```
