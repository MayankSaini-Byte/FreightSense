from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error



def train_ridge(X_train, y_train):
    model = Ridge()
    model.fit(X_train, y_train)
    return model

def train_lasso(X_train, y_train):
    model = Lasso()
    model.fit(X_train, y_train)
    return model

def train_elasticnet(X_train, y_train):
    model = ElasticNet()
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_adaboost(X_train, y_train):
    model = AdaBoostRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    return model

def train_knn(X_train, y_train):
    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train, y_train)
    return model

def train_svr(X_train, y_train):
    model = SVR(kernel='rbf')
    model.fit(X_train, y_train)
    return model



def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    r2  = round(r2_score(y_test, y_pred), 4)
    mae = round(mean_absolute_error(y_test, y_pred), 4)
    print(f"{model_name:<25} R²: {r2}  MAE: {mae}")
    return {'model_name': model_name, 'r2': r2, 'mae': mae}