# Time-of-detection
Github repository to accompany article "How early can an upcoming critical transition be detected?"

<h2 id="table-of-contents"> Table of Contents</h2>

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project"> ➤ Abstract</a></li>
    <li><a href="#folder-structure"> ➤ Folder Structure</a></li>
    <li><a href="#dataset"> ➤ Dataset</a></li>
    <li><a href="#roadmap"> ➤ Roadmap</a></li>
    <li>
      <a href="#preprocessing"> ➤ Preprocessing</a>
      <ul>
        <li><a href="#preprocessed-data">Pre-processed data</a></li>
        <li><a href="#statistical-feature">Statistical feature</a></li>
        <li><a href="#topological-feature">Topological feature</a></li>
      </ul>
    </li>
    <!--<li><a href="#experiments">Experiments</a></li>-->
    <li><a href="#results-and-discussion"> ➤ Results and Discussion</a></li>
    <li><a href="#references"> ➤ References</a></li>
    <li><a href="#contributors"> ➤ Contributors</a></li>
  </ol>
</details>


<h2 id="about-the-project"> Abstract</h2>

The real-time application of early warning signals (EWSs) has often been overlooked; many studies show the presence of EWSs but do not detect when the trend becomes significant.  Knowing if the signal can be detected _early_ enough is of critical importance for the applicability of EWSs. Detection methods which present this analysis are sparse and are often developed for each individual study. Here, we provide a validation and summary of a range of currently available detection methods developed from EWSs. We include an additional constraint, which requires multiple time-series points to satisfy the algorithms' conditions before a detection of an approaching critical transition can be flagged.  We apply this procedure to a simulated study of an infectious disease system undergoing disease elimination. For each detection algorithm we select the hyper-parameter which minimises classification errors using ROC analysis. We consider the effect of time-series length on these results, finding that all algorithms become less accurate as the access to available data decreases.  We compare EWS detection methods with alternate algorithms found from the change-point analysis literature and assess the suitability of using change-point analysis to detect abrupt changes in a system's steady state. 
<h2 id="folder-structure"> Folder Structure</h2>

[Create sythentic data (Ext, NExt and Fix)](./data/README.md)

    data
    ├── README.md
    ├── funcs_sim.py
    ├── gillespie_run_SIS_model.py

[Run the 2-sigma method (Drake & Griffen, 2010)](./two_sigma_method/README.md)

    two_sigma_method
    ├── README.md
    ├── funcs_twosigma.py
    ├── run_two_sigma_with_consec.py

[Run the changing p-value method (Harris, 2020)](./changing_p_value/README.md)

    changing_p_value
    ├── README.md
    ├── funcs_pvalue.py
    ├── pvalue_consecutive.py
    ├── run_changing_pvalue.py

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
    │   │   └── run_logistic_EWSscombination.py
    ├── py_earlywarnings
