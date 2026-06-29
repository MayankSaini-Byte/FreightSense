from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report



def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    model = XGBClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model



def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    acc     = round(accuracy_score(y_test, y_pred), 4)
    roc_auc = round(roc_auc_score(y_test, y_pred), 4)
    print(f"{model_name:<25} Accuracy: {acc}  ROC-AUC: {roc_auc}")
    return {'model_name': model_name, 'accuracy': acc, 'roc_auc': roc_auc}


def print_classification_report(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Flagged']))