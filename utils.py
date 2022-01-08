
from classifier import Classifier
from regressor import Regressor


def processor(df, c, model):

    if c == "clf":
        model = Classifier(df, model)
        study, best_model = model.classify()
        return study, best_model

    if c == "reg":
        model = Regressor(df, model)
        study, best_model = model.classify()
        return study, best_model
