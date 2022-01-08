
from classifier import Classifier
from regressor import Regressor


def processor(df, model):
    model = Classifier(df, model)
    study, best_model = model.classify()
    return study, best_model