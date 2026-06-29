import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_vendor_invoice_data():
    df = pd.read_csv(DATA_DIR / "vendor_invoice.csv")
    print(f"Loaded vendor_invoice: {df.shape}")
    return df


def engineer_features(df):
    df = df.copy()

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['PODate']      = pd.to_datetime(df['PODate'])
    df['PayDate']     = pd.to_datetime(df['PayDate'])

    df['invoice_to_pay'] = (df['PayDate']     - df['InvoiceDate']).dt.days
    df['po_to_invoice']  = (df['InvoiceDate'] - df['PODate']).dt.days
    df['po_to_pay']      = (df['PayDate']     - df['PODate']).dt.days

    df['Freight_per_unit'] = df['Freight'] / df['Quantity']

    return df


def prepare_features(df):
    drop_cols = [
        'InvoiceDate', 'PODate', 'PayDate',
        'VendorName', 'Approval',
        'VendorNumber', 'PONumber',
        'Quantity',
        'Freight_per_unit',
        'Freight'
    ]

    X = df.drop(columns=drop_cols)
    y = np.log1p(df['Freight'])

    print(f"Features : {X.columns.tolist()}")
    print(f"X shape  : {X.shape} | y shape: {y.shape}")
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test