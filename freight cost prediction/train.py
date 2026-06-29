import joblib
from pathlib import Path

from data_preprocessing import (
    load_vendor_invoice_data,
    engineer_features,
    prepare_features,
    split_data
)
from model_evaluation import (
    train_ridge, train_lasso, train_elasticnet,
    train_decision_tree, train_random_forest,
    train_gradient_boosting, train_adaboost,
    train_xgboost, train_knn, train_svr,
    evaluate_model
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def main():
    print("=" * 50)
    print("  FreightSense — Freight Cost Predictor")
    print("=" * 50)

    df           = load_vendor_invoice_data()
    df           = engineer_features(df)
    X, y         = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\nTraining models...")
    models = {
        "Ridge"            : train_ridge(X_train, y_train),
        "Lasso"            : train_lasso(X_train, y_train),
        "ElasticNet"       : train_elasticnet(X_train, y_train),
        "DecisionTree"     : train_decision_tree(X_train, y_train),
        "RandomForest"     : train_random_forest(X_train, y_train),
        "GradientBoosting" : train_gradient_boosting(X_train, y_train),
        "AdaBoost"         : train_adaboost(X_train, y_train),
        "XGBoost"          : train_xgboost(X_train, y_train),
        "KNN"              : train_knn(X_train, y_train),
        "SVR"              : train_svr(X_train, y_train),
    }

    print("\nEvaluation Results:")
    print("-" * 45)
    results = [
        evaluate_model(model, X_test, y_test, name)
        for name, model in models.items()
    ]

    best_info  = max(results, key=lambda x: x['r2'])
    best_name  = best_info['model_name']
    best_model = models[best_name]

    print(f"\n✅ Best Model : {best_name}")
    print(f"   R²         : {best_info['r2']}")
    print(f"   MAE        : {best_info['mae']}")

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "freight_predictor_gb.pkl"
    joblib.dump(best_model, model_path)
    print(f"\n Model saved → {model_path}")


if __name__ == "__main__":
    main()