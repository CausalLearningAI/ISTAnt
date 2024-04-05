from sklearn.linear_model import LogisticRegression, LinearRegression
#from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import torch
import numpy as np
from econml.dml import CausalForestDML
from econml.metalearners import TLearner, SLearner, XLearner
from econml.dr import DRLearner

def compute_ate(Y, T, X, method="AIPW", color="preselected", model_propensity = LogisticRegression(), model_outcome = None, verbose=True):
    if color=="yellow":
        Y = Y[:,0]
    elif color=="blue":
        Y = Y[:,1]
    elif color=="preselected":
        pass
    else:
        raise ValueError(f"Color {color} not defined. Please select between 'yellow', 'blue' or 'preselected'.")

    if model_outcome is None:
        if Y.dtype == torch.int32 or Y.dtype == torch.int64:
            model_outcome = LogisticRegression()
        else:
            model_outcome = LinearRegression()

    if method=="ead":
        model = EAD()
    elif method=="aipw":
        model = AIPW(model_propensity = model_propensity, 
                     model_outcome = model_outcome)
    elif method == 'slearner':
        model = SLearner(overall_model = model_outcome)
    elif method == 'tlearner':
        model = TLearner(models = model_outcome)
    elif method == 'xlearner':
        model = XLearner(models = model_outcome,
                         propensity_model = model_propensity,
                         #cate_models = model_outcome
                         )
    elif method == 'drlearner':
        model = DRLearner(discrete_outcome = True,
                          #model_propensity = model_propensity,
                          #model_regression = model_outcome,
                          #model_final = LinearRegression()
                          )
    elif method == 'causalforest':
        model = CausalForestDML(discrete_outcome = True,
                                discrete_treatment = True,
                                #model_t = model_propensity, 
                                #model_y = model_outcome
                                )
    else:
        raise ValueError(f"'{method}' method for ATE estimation not implemented.")
    model.fit(Y=Y[T!=2].int(), T=T[T!=2].int(), X=X[T!=2])
    ATE_B = model.effect(X[T!=2]).mean()
    model.fit(Y=Y[T!=1].int(), T=(T[T!=1]/2).int(), X=X[T!=1])
    ATE_inf = model.effect(X[T!=1]).mean()
    if verbose: 
        print(f"Color: {color}; Method: {method}")
        print(f"    ATE_B: {ATE_B:.3f}; ATE_inf: {ATE_inf:.3f}")
    return ATE_B, ATE_inf

class EAD:
    def __init__(self):
        self.name = "EAD"

    def fit(self, Y, T, X):
        E_Y_control = Y[T==0].sum() / len(Y[T==0])
        E_Y_treated = Y[T==1].sum() / len(Y[T==1])
        ead = E_Y_treated - E_Y_control
        self.ate = ead.item()

    def effect(self, _):
        return np.array(self.ate)

class AIPW:
    def __init__(self, model_propensity, model_outcome):
        self.name = "AIPW"
        self.model_propensity = model_propensity
        self.model_outcome = model_outcome

    def fit(self, Y, T, X):
        N = len(Y)
        self.model_propensity.fit(X = X, y = T)
        self.model_outcome.fit(X = torch.cat((X, T.reshape(N, 1)), dim=1), y = Y)
        mu0 = self.model_outcome.predict(torch.cat((X, torch.zeros(N, 1)), dim=1))
        mu1 = self.model_outcome.predict(torch.cat((X, torch.ones(N, 1)), dim=1))
        ps = self.model_propensity.predict_proba(X)[:, 1]
        self.ite = mu1-mu0 + T.numpy() * (Y.numpy()-mu1) / (ps) - (1-T.numpy()) * (Y.numpy()-mu0) / (1-ps) 
        #self.ate = ite.mean().item()

    def effect(self, _):
        return self.ite