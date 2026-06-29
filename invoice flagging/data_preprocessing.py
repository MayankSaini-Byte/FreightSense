
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_data():
    purchases       = pd.read_csv(DATA_DIR / "purchases.csv")
    purchase_prices = pd.read_csv(DATA_DIR / "purchase_prices.csv")
    vendor_invoice  = pd.read_csv(DATA_DIR / "vendor_invoice.csv")

    print(f"purchases       : {purchases.shape}")
    print(f"purchase_prices : {purchase_prices.shape}")
    print(f"vendor_invoice  : {vendor_invoice.shape}")

    return purchases, purchase_prices, vendor_invoice


def aggregate_purchases(purchases):
    purchases_agg = purchases.groupby(['PONumber', 'VendorNumber']).agg(
        Dollars_po    = ('Dollars',       'sum'),
        Quantity_po   = ('Quantity',      'sum'),
        PurchasePrice = ('PurchasePrice', 'mean')
    ).reset_index()

    print(f"purchases_agg : {purchases_agg.shape}")
    return purchases_agg


def merge_tables(vendor_invoice, purchases_agg, purchase_prices):

    df = vendor_invoice.merge(
        purchases_agg,
        on=['PONumber', 'VendorNumber'],
        how='left'
    )
    df = df.rename(columns={
        'Quantity' : 'Quantity_inv',
        'Dollars'  : 'Dollars_inv'
    })

   
    vendor_avg_price = (
        purchase_prices
        .groupby('VendorNumber')['PurchasePrice']
        .mean()
        .reset_index()
    )
    vendor_avg_price.columns = ['VendorNumber', 'PurchasePrice_catalog']

    df = df.merge(vendor_avg_price, on='VendorNumber', how='left')

    print(f"Merged df : {df.shape}")
    return df


def engineer_features(df):
    df = df.copy()

    df['price_deviation'] = df['Dollars_inv'] - df['Dollars_po']
    df['price_ratio']     = df['Dollars_inv'] / (df['Dollars_po'] + 1)
    df['freight_ratio']   = df['Freight']     / (df['Dollars_inv'] + 1)

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['PODate']      = pd.to_datetime(df['PODate'])
    df['PayDate']     = pd.to_datetime(df['PayDate'])

    df['invoice_delay_days'] = (df['InvoiceDate'] - df['PODate']).dt.days
    df['payment_delay_days'] = (df['PayDate']     - df['InvoiceDate']).dt.days

    df['qty_mismatch'] = df['Quantity_inv'] - df['Quantity_po']

    return df


def create_flag(df):
    df = df.copy()
    df['flag'] = (
        (df['freight_ratio']      > 0.05) |
        (df['invoice_delay_days'] > 19)   |
        (df['payment_delay_days'] > 40)
    ).astype(int)

    print(f"Flag distribution:\n{df['flag'].value_counts()}")
    print(f"Flagged: {df['flag'].mean()*100:.1f}%")
    return df


def prepare_features(df):
    features = [
        'Dollars_inv', 'Freight', 'Quantity_inv',
        'PurchasePrice', 'invoice_delay_days', 'payment_delay_days'
    ]
    X = df[features].fillna(0)
    y = df['flag']

    print(f"X shape: {X.shape} | y shape: {y.shape}")
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test