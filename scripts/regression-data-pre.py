#!/usr/bin/env python3
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    X = df.drop('Lifetime_years', axis=1)
    y = df['Lifetime_years']

    categorical_features = [
        'Base_Resin','Environment',
        'Primary_AO','Secondary_AO',
        'Carbon_Black_Type','Wax_Type'
    ]
    numeric_to_scale = [
        'SDR','Service_Pressure_bar',
        'Service_Temperature_C',
        'CB_Content_%','Wax_Content_%'
    ]
    numeric_to_keep = [
        'Primary_AO_ppm','Secondary_AO_ppm'
    ]

    numeric_transformer = StandardScaler()
    preprocessor = ColumnTransformer(transformers=[
        ('num_scale', numeric_transformer, numeric_to_scale),
        ('num_keep', 'passthrough', numeric_to_keep),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    X_processed = preprocessor.fit_transform(X)
    cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = numeric_to_scale + numeric_to_keep + list(cat_names)
    X_processed = pd.DataFrame(X_processed, columns=feature_names)

    return X_processed, y, preprocessor

if __name__ == "__main__":
    input_path = '/Users/hj/MLAdditive/data/iso9080_lifetime_dataset.csv'
    output_path = '/Users/hj/MLAdditive/data/preprocessed.csv'

    df = load_data(input_path)
    X_proc, y, _ = preprocess(df)
    df_out = X_proc.copy()
    df_out['Lifetime_years'] = y
    df_out.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")
