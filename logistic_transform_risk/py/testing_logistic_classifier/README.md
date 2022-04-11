# About
Takes the weights found from the logistic classifier using the training data, and tests with the testing data.

Reduces the total number of compositions of EWSs by taking composite EWSs which have an AUC > 0.6

Applies the consecutive point strategy with all EWSs in the reduced dataset. 

# How to use
Optional: reduced the total number of composite statistics by taking those which have an AUC>0.6 using `reducedIndicators=0.6`  in the file `logistic_with_consec.py`.

Then run the consecutive point method.  Below is an example bash script demonstrating how to run:
```
#!/bin/bash

reducedIndicators=0.6
for TIME in 20 50 100 250; do
    python logistic_with_consec.py ${TIME} ${reducedIndicators}
done 

```
