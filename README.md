# Time-of-detection
Github repository to accompany article "How early can an upcoming critical transition be detected?"

<h2 id="table-of-contents"> Table of Contents</h2>

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#abstract"> ➤ Abstract</a></li>
    <li><a href="#about-the-project"> ➤ About the project</a></li>
    <li><a href="#folder-structure"> ➤ Folder Structure</a></li>
    <li><a href="#dataset"> ➤ Sythentic Dataset</a></li>
    <li><a href="#results-and-discussion"> ➤ Results and Discussion</a></li>
    <li><a href="#references"> ➤ References</a></li>
  </ol>
</details>


<h2 id="abstract"> Abstract</h2>

The real-time application of early warning signals (EWSs) has often been overlooked; many studies show the presence of EWSs but do not detect when the trend becomes significant.  Knowing if the signal can be detected _early_ enough is of critical importance for the applicability of EWSs. Detection methods which present this analysis are sparse and are often developed for each individual study. Here, we provide a validation and summary of a range of currently available detection methods developed from EWSs. We include an additional constraint, which requires multiple time-series points to satisfy the algorithms' conditions before a detection of an approaching critical transition can be flagged.  We apply this procedure to a simulated study of an infectious disease system undergoing disease elimination. For each detection algorithm we select the hyper-parameter which minimises classification errors using ROC analysis. We consider the effect of time-series length on these results, finding that all algorithms become less accurate as the access to available data decreases.  We compare EWS detection methods with alternate algorithms found from the change-point analysis literature and assess the suitability of using change-point analysis to detect abrupt changes in a system's steady state. 

<h2 id="about-the-project"> About the project</h2>
EWSs are model-independent time-series methods for detecting when a system goes through a critical transition. Here we present the code for the following online detection algorithms: 
- 2-sigma method (Drake & Griffen, 2010)
- Changing p-value (Harris et al., 2020)
- Logistic Transform Risk (Brett & Rohani, 2020)
- Quickest Detection (Shiryaev, 1961)
- Maximum Likelihood Estimation

and investigate how early we can detect R_0 = 1, in order to inform the path towards disease elimination. We adapt all algorithms to include a consecutive point strategy and find the optimal number of consectuive points required to minimise classification errors. 
<h2 id="folder-structure"> Folder structure</h2>

[Create sythentic data (Ext, NExt and Fix)](./data/README.md)

    data
    ├── README.md
    ├── funcs_sim.py
    └── gillespie_run_SIS_model.py

[Run the 2-sigma method (Drake & Griffen, 2010)](./two_sigma_method/README.md)

    two_sigma_method
    ├── README.md
    ├── funcs_twosigma.py
    └── run_two_sigma_with_consec.py

[Run the changing p-value method (Harris, 2020)](./changing_p_value/README.md)

    changing_p_value
    ├── README.md
    ├── funcs_pvalue.py
    ├── pvalue_consecutive.py
    └── run_changing_pvalue.py

[Run the logistic transform risk method (Brett & Rohani, 2021)](./logistic_transform_risk/py/README.md)

    logistic_transform_risk
    ├── data
    │   └── simulation_parameters.csv
    ├── py
    │   ├── helper.py
    │   ├── create_training_data
    │   │   ├── README.md
    │   │   ├── create_training_data.py
    │   │   └── gillespie_SIS.py
    │   ├── testing_logistic_classifier
    │   │   ├── README.md
    │   │   ├── funcs_logs.py
    │   │   └── logistic_with_consec.py
    │   ├── training_logistic_classifier
    │   │   ├── README.md
    │   │   ├── classifier_training.py
    │   │   ├── cross_validation.py
    │   │   ├── ews_logistic_regression.py
    │   │   ├── gillespie_SIS.py
    │   │   ├── latin_hypercube_simulator.py
    │   │   ├── run_log_classifier.py
    │   └── └── run_logistic_EWSscombination.py
    ├── py_earlywarnings
    │   ├── README.md
    │   ├── LICENSE
    │   ├── setup.py
    │   ├── requirements.txt
    │   ├── ews
    │   │   ├── __init__.pu
    │   │   ├── entropy.py
    │   │   ├── ews.py
    └── └── └── kolmogorov_complexity.pyx

[Run the Quickest Detection method (Shiryaev, 1961)](./quickest_detection/README.md)

    quickest_detection
    ├── README.md
    ├── funcs_qd.py
    ├── QD_consecutive.py
    ├── shiryaev_roberts_stat.m
    └── quickest_detection.m

[Run the Maximum Likelihood Estimation detection method](./mle/README.md)

    mle
    ├── README.md
    ├── Likelihood_of_changepoint.m
    ├── LogLikelihoodNormal.m
    ├── LogLikelihoodNormalsigma_tau.m
    ├── MLE_alternate_hypothesis_tau.m
    ├── Run_MLE_changepoint.m
    ├── central_difference_2nd.m
    └── confidence_interval_smoothing.m

<h2 id="dataset"> Sythetic Dataset</h2>

Stochastic simulations of the testing data is made using the Gillespie Algorithm [see further information and parameter choices](./data/README.md)

We test each online detection algorithm with data that is bifurcating (R_0 goes through 1), known as:
- EXT: disease extinction, $R_0$ reduces from 5 to 0 by slowly decreasing $\beta$ at a rate $1/500$. This is an example of a bifurcating simulation.
and with null datasets which do not undego a bifurcation: 
- NEXT: disease <em>not</em> extinction (also called "fixchanging" in code). As in EXT, but $\beta$ stops changing when $R_0 = 1.3$. $R_0$ stays fixed at this value for the rest of the simulation. This is an example of a null simulation. 
- FIX: endemic disease. Parameters do not change in time and are fixed throughout the simulation. This is an example of a null simulation. 

A true positive detection occurs when we correctly detection disease elimination with Ext data. 
A true negative detection occurs when we (correctly) do not detection disease elimination with the null datasets. 

<h2 id="references"> References</h2>

<ul>
  <li>
    <p>
    Southall E, Brett TS, Tildesley MJ, Dyson L. Early warning signals of infectious disease transitions: a review. Journal of the Royal society of Interface. 2021.
    </p>
  </li>
  <li>
    <p>
    Drake JM, Griffen BD. Early warning signals of extinction in deteriorating environments. Nature. 2010.
    </p>
  </li>
  <li>
    <p>
      Harris MJ, Hay SI, Drake JM. Early warning signals of malaria resurgence in Kericho, Kenya. Biology letters. 2020.
    </p>
  </li>
  <li>
    <p>
      Brett TS, Rohani P. Dynamical footprints enable detection of disease emergence. PLoS biology. 2020.
    </p>
  </li>
  <li>
    <p>
      Shiryaev AN. Problem of most rapid detection of a disturbance in stationary processes. Doklady Alademii Nauk SSSR. 1961.
    </p>
  </li>
</ul>