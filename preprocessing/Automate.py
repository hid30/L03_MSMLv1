import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_loan_data(path, test_size=0.2, random_state=42):
    """
    Fungsi otomatis untuk memproses data peminjaman.
    
    Parameters:
        path (str): Path ke file CSV
        test_size (float): Proporsi data test
        random_state (int): Nilai random state untuk replikasi
        
    Returns:
        X_train_processed (pd.DataFrame)
        X_test_processed (pd.DataFrame)
        y_train (pd.Series)
        y_test (pd.Series)
    """
    # Load dan bersihkan data
    df = pd.read_csv(path)
    df = df.dropna().drop_duplicates()
    
    # Pisahkan fitur dan target
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # Tentukan kolom
    numeric_cols = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
                    'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
                    'credit_score']
    nominal_cols = ['person_education', 'person_home_ownership', 'loan_intent']
    biner_cols = ['person_gender', 'previous_loan_defaults_on_file']

    # Filter kolom yang tersedia
    all_cols = X_train.columns.tolist()
    numeric_cols = [col for col in numeric_cols if col in all_cols]
    nominal_cols = [col for col in nominal_cols if col in all_cols]
    biner_cols = [col for col in biner_cols if col in all_cols]
    
    # Label encoding kolom biner
    for col in biner_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
    
    # Pipeline transformasi
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    nominal_transformer = Pipeline(steps=[('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('nom', nominal_transformer, nominal_cols)
        ],
        remainder='passthrough'  # biner tetap
    )
    
    # Fit-transform pada data train, transform pada test
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Ambil nama kolom akhir
    ohe_features = preprocessor.named_transformers_['nom']['onehot'].get_feature_names_out(nominal_cols)
    final_columns = numeric_cols + list(ohe_features) + biner_cols
    
    # Kembalikan sebagai DataFrame
    X_train_processed = pd.DataFrame(X_train_processed, columns=final_columns)
    X_test_processed = pd.DataFrame(X_test_processed, columns=final_columns)
    
    return X_train_processed, X_test_processed, y_train, y_test

def main():
    # Path ke dataset mentah
    raw_dataset_path = "loandata_raw/loan_data.csv"
    
    if not os.path.exists(raw_dataset_path):
        raise FileNotFoundError(f"File tidak ditemukan di path: {raw_dataset_path}")
    
    # Path untuk menyimpan hasil preprocessing
    train_output_path = "preprocessing/X_train_processed.csv"
    test_output_path = "preprocessing/X_test_processed.csv"
    y_train_path = "preprocessing/y_train.csv"
    y_test_path = "preprocessing/y_test.csv"
    
    # Buat folder output jika belum ada
    os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
    
    print("Memulai proses preprocessing dataset peminjaman...")
    X_train_processed, X_test_processed, y_train, y_test = preprocess_loan_data(raw_dataset_path)
    
    # Simpan hasil preprocessing
    X_train_processed.to_csv(train_output_path, index=False)
    X_test_processed.to_csv(test_output_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    y_test.to_csv(y_test_path, index=False)

    print("Preprocessing selesai.")
    print(f"X_train disimpan di: {train_output_path}")
    print(f"X_test disimpan di: {test_output_path}")
    
    # Tampilkan hasil
    print("\nContoh X_train:")
    print(X_train_processed.head())

if __name__ == "__main__":
    main()