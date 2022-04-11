from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

# Random seed for logistic regression
r_state = 42


def ews_logistic_regression(ews_data, signals,
                            standardise=False, do_pca=False, pca_components=7,
                            method="LR", print_output=False, **kwargs):
    '''
    Implements Lasso regression with regularization strength to prevent overfitting 
    
    INPUT
    :param ews_data: pandas dataframe, training models 
    :param signals: array of strings, EWS signals used in analysis, array from helper.signals()
    :param standardise: boolean, default False. Transforms ews data using ...
    ... sklearn.preprocessing.StandardScaler
    :param do_pca: boolean, default False. Transforms ews data using ...
    ... sklearn.decomposition.PCA with n_components 
    :param pca_components: int, default 7. required for sklearn.decomposition.PCA
    :param methods: str, default "LR" (logistic regression). Options "LR" or "SGD"
    :param print_output: boolean, default False
    :param kwargs: handle named arguments: solver, penalty, C
    :param solver: "liblinear" supports both L1 and L2 regularization wtih dual formulation only for the L2 penlaty
    :param penalty: "l1" for L1-logsitic regression
    :param C: inverse of regularization strength, positive float, smaller values specify stronger regularization
    
    RETURNS
    : lr_clf: instance of LogisticRegression or SGDClassifier defined and fitted 
    : coefs: dictionary of a weight (coefs from lr_clf) and intercept (intercept from lr_clf)

    
    '''
    # select matrix of EWS signals (shape: repeats by ews) and y as binary array of emerging (1)...
    # ... and non emerging (0)
    x, y = ews_data[signals].values, ews_data["is_test"].values
    if type(signals) is str:
        x = x.reshape(-1, 1)

    scaler = None
    pca = None

    if standardise:        
        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)
    if do_pca:
        pca = PCA(n_components=pca_components)
        pca.fit(x)
        x = pca.transform(x)

    if method == "LR":
        kwargs.setdefault('solver', "liblinear")
        # creates an instance of logisticregression model and binds its references to the variable lr_clf
        lr_clf = LogisticRegression(random_state=r_state, **kwargs)
                                    #,solver="liblinear")
    elif method == "SGD":
        lr_clf = SGDClassifier(loss="log", random_state=r_state, **kwargs)
    else:
        print("Invalid method")
        return None
    #once the model is created, need to fit it to observations y
    lr_clf.fit(x, y)

    if do_pca:
        coefs = pca.inverse_transform(lr_clf.coef_.reshape(-1))
    else:
        coefs = lr_clf.coef_.reshape(-1)


    if standardise:
        #get coefficients from sklearn
        w = lr_clf.coef_.reshape(-1)
        #get intercept from sklearn and standardise 
        w0 = lr_clf.intercept_[0] - np.sum(w * scaler.mean_ / scaler.scale_)
        coefs = w/scaler.scale_
        coefs = dict(zip(signals, coefs))
        coefs["intercept"] = w0
        if print_output:
            print(scaler.scale_)
            print(scaler.mean_)
            print(w)
            print(coefs)

    else:
        coefs = dict(zip(signals, coefs))
        coefs["intercept"] = lr_clf.intercept_[0]
        if print_output:
            print(coefs)


    return coefs, lr_clf
