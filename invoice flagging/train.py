import joblib
from pathlib import Path

from data_preprocessing import (
    load_data,
    aggregate_purchases,
    merge_tables,
    engineer_features,
    create_flag,
    prepare_features,
    split_data
)
from model_evaluation import (
    train_random_forest,
    train_gradient_boosting,
    train_xgboost,
    evaluate_model,
    print_classification_report
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def main():
    print("=" * 50)
    print("  Invoice Intelligence — Invoice Flagger")
    print("=" * 50)

    purchases, purchase_prices, vendor_invoice = load_data()
    purchases_agg = aggregate_purchases(purchases)
    df            = merge_tables(vendor_invoice, purchases_agg, purchase_prices)
    df            = engineer_features(df)
    df            = create_flag(df)
    X, y          = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    models = {
        "RandomForest"     : train_random_forest(X_train, y_train),
        "GradientBoosting" : train_gradient_boosting(X_train, y_train),
        "XGBoost"          : train_xgboost(X_train, y_train),
    }

    print("\nEvaluation Results:")
    print("-" * 50)
    results = [
        evaluate_model(model, X_test, y_test, name)
        for name, model in models.items()
    ]

    best_info  = max(results, key=lambda x: x['roc_auc'])
    best_name  = best_info['model_name']
    best_model = models[best_name]

    print_classification_report(best_model, X_test, y_test)

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "invoice_flagger.pkl"
    joblib.dump(best_model, model_path)
    print(f"\n💾 Model saved → {model_path}")


if __name__ == "__main__":
    main()