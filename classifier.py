
import optuna
import pandas as pd
import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

class objective(object):
    def __init__(self, df, classifier_name=None):
        self.df = df
        self.classifier_name = classifier_name
    
    def __call__(self, trial):
        if self.classifier_name==None:
            self.i=True
        if self.i:
            self.classifier_name = trial.suggest_categorical("classifier", ["SVC", "rf","knn",'tree','logistic'])
        if self.classifier_name == "SVC":
            svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
            self.classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")
        elif self.classifier_name=='rf':
            rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
            self.classifier_obj = sklearn.ensemble.RandomForestClassifier(
                max_depth=rf_max_depth, n_estimators=10
            )
        elif self.classifier_name=='knn':
            n_neighbors = trial.suggest_int('n_neighbours',5,7,log=True)
            weights = trial.suggest_categorical('weights',['uniform','distance'])
            self.classifier_obj = KNeighborsClassifier(
                n_neighbors = n_neighbors,
                weights = weights
            )
        elif self.classifier_name=='tree':
            max_depth = trial.suggest_int('max_depth',10,20,log=True)
            criterion = trial.suggest_categorical('criterion',['gini', 'entropy'])
            self.classifier_obj = DecisionTreeClassifier(
                criterion=criterion,
                max_depth=max_depth,
                random_state=42
            )

        elif self.classifier_name=='logistic':
            C = trial.suggest_int('c',1,100,log=True)
            self.classifier_obj = sklearn.linear_model.LogisticRegression(
            C=C, random_state=42)


        score = sklearn.model_selection.cross_val_score(self.classifier_obj, x, y, n_jobs=-1, cv=3)
        accuracy = score.mean()
        return accuracy

class Classifier():

    def __init__(self,df, classifier_name):
        self.df = df
        self.classifier_name = classifier_name

    

    def classify(self):
        global x, y
        x, y = df.iloc[:,:-1], df.iloc[:,-1]
        study = optuna.create_study(direction="maximize")
        study.optimize(objective(df), n_trials=100)
        return study

df = pd.read_csv('cleaned/processed.csv')
clf_name = 'SVC'
clf = Classifier(df, clf_name)
study = clf.classify()
trail = study.best_trial
print(trail.params)
