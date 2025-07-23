# #!/usr/bin/env python3
# import pandas as pd
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder, StandardScaler

# def load_data(path):
#     return pd.read_csv(path)

# def preprocess(df):
#     X = df.drop('Lifetime_years', axis=1)
#     y = df['Lifetime_years']

#     categorical_features = [
#         'Base_Resin','Environment',
#         'Primary_AO','Secondary_AO',
#         'Carbon_Black_Type','Wax_Type'
#     ]
#     numeric_to_scale = [
#         'SDR','Service_Pressure_bar',
#         'Service_Temperature_C',
#         'CB_Content_%','Wax_Content_%'
#     ]
#     numeric_to_keep = [
#         'Primary_AO_ppm','Secondary_AO_ppm'
#     ]

#     numeric_transformer = StandardScaler()
#     preprocessor = ColumnTransformer(transformers=[
#         ('num_scale', numeric_transformer, numeric_to_scale),
#         ('num_keep', 'passthrough', numeric_to_keep),
#         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
#     ])

#     X_processed = preprocessor.fit_transform(X)
#     cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
#     feature_names = numeric_to_scale + numeric_to_keep + list(cat_names)
#     X_processed = pd.DataFrame(X_processed, columns=feature_names)

#     return X_processed, y, preprocessor

# if __name__ == "__main__":
#     input_path = '/Users/hj/MLAdditive/data/iso9080_lifetime_dataset.csv'
#     output_path = '/Users/hj/MLAdditive/data/preprocessed.csv'

#     df = load_data(input_path)
#     X_proc, y, _ = preprocess(df)
#     df_out = X_proc.copy()
#     df_out['Lifetime_years'] = y
#     df_out.to_csv(output_path, index=False)
#     print(f"Preprocessed data saved to {output_path}")
#!/usr/bin/env python3
# #!/usr/bin/env python3
# import pandas as pd
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder, StandardScaler

# def load_data(path):
#     return pd.read_csv(path)

# def preprocess(df):
#     X = df.drop('Lifetime_years', axis=1)
#     y = df['Lifetime_years']

#     categorical_features = [
#         'Base_Resin','Environment',
#         'Primary_AO','Secondary_AO',
#         'Carbon_Black_Type','Wax_Type'
#     ]
#     numeric_to_scale = [
#         'SDR','Service_Pressure_bar',
#         'Service_Temperature_C',
#         'CB_Content_%','Wax_Content_%'
#     ]
#     numeric_to_keep = [
#         'Primary_AO_ppm','Secondary_AO_ppm'
#     ]

#     numeric_transformer = StandardScaler()
#     preprocessor = ColumnTransformer(transformers=[
#         ('num_scale', numeric_transformer, numeric_to_scale),
#         ('num_keep', 'passthrough', numeric_to_keep),
#         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
#     ])

#     X_processed = preprocessor.fit_transform(X)
#     cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
#     feature_names = numeric_to_scale + numeric_to_keep + list(cat_names)
#     X_processed = pd.DataFrame(X_processed, columns=feature_names)

#     return X_processed, y, preprocessor

# if __name__ == "__main__":
#     input_path = '/Users/hj/MLAdditive/data/iso9080_lifetime_dataset.csv'
#     output_path = '/Users/hj/MLAdditive/data/preprocessed.csv'

#     df = load_data(input_path)
#     X_proc, y, _ = preprocess(df)
#     df_out = X_proc.copy()
#     df_out['Lifetime_years'] = y
#     df_out.to_csv(output_path, index=False)
#     print(f"Preprocessed data saved to {output_path}")
#!/usr/bin/env python3
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import numpy as np

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    X = df.drop('Lifetime_years', axis=1)
    y = df['Lifetime_years']

    categorical_features = [
        'Base_Resin', 'Environment',
        'Primary_AO', 'Secondary_AO',
        'Carbon_Black_Type', 'Wax_Type'
    ]
    numeric_to_scale = [
        'SDR', 'Service_Pressure_bar',
        'Service_Temperature_C',
        'CB_Content_%', 'Wax_Content_%'
    ]
    numeric_to_keep = [
        'Primary_AO_ppm', 'Secondary_AO_ppm'
    ]

    # Preprocessor definition
    numeric_transformer = StandardScaler()
    preprocessor = ColumnTransformer(transformers=[
        ('num_scale', numeric_transformer, numeric_to_scale),
        ('num_keep', 'passthrough', numeric_to_keep),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    # Transform
    X_processed = preprocessor.fit_transform(X)
    cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = numeric_to_scale + numeric_to_keep + list(cat_names)
    X_processed = pd.DataFrame(X_processed, columns=feature_names)


    return X_processed, y, preprocessor

def bin_target(y, bins=3):
    return pd.qcut(y, q=bins, labels=False, duplicates='drop')

def apply_smote(X, y_binned):
    smote = SMOTE(random_state=42)
    X_over, y_binned_over = smote.fit_resample(X, y_binned)
    return X_over, y_binned_over

if __name__ == "__main__":
    input_path = '/Users/hj/MLAdditive/data/iso9080_lifetime_dataset.csv'
    output_path = '/Users/hj/MLAdditive/data/preprocessed_smote.csv'

    # Load data
    df = load_data(input_path)

    # Preprocess
    X_proc, y_true, _ = preprocess(df)

    # Bin for SMOTE + stratified proxy
    y_binned = bin_target(y_true, bins=3)

    # Apply SMOTE
    X_over, y_binned_over = apply_smote(X_proc, y_binned)

    # Map binned targets back to original regression values
    # Works because y_true is a pandas Series
    y_true_over = [y_true[y_binned == b].sample(1, random_state=42).iloc[0] for b in y_binned_over]
    y_true_over = pd.Series(y_true_over, name='Lifetime_years')

    # Save output
    df_out = pd.DataFrame(X_over, columns=X_proc.columns)
    df_out['Lifetime_years'] = y_true_over
    df_out.to_csv(output_path, index=False)

    print(f"✅ SMOTE-balanced and preprocessed data saved to: {output_path}")