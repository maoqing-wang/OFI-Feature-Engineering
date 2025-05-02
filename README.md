# OFI-Feature-Engineering
git init
git add ofi_feature_engineering.py
git commit -m "Initial commit: OFI feature functions"
git remote add origin https://github.com/your-username/OFI-Feature-Engineering.git
git push -u origin main
def load_and_clean_data(path):
    df = pd.read_csv(path)
    df['ts_event'] = pd.to_datetime(df['ts_event'], utc=True)
    df.sort_values('ts_event', inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def compute_best_level_ofi(df):
    df['OFI_best_level'] = df['bid_sz_00'].diff().fillna(0) - df['ask_sz_00'].diff().fillna(0)
    return df

def compute_multi_level_ofi(df, depth=10):
    delta_bid = [df[f'bid_sz_0{i}'].diff().fillna(0) for i in range(depth)]
    delta_ask = [df[f'ask_sz_0{i}'].diff().fillna(0) for i in range(depth)]
    df['OFI_multi_level'] = sum(delta_bid) - sum(delta_ask)
    return df

def compute_integrated_ofi(df, depth=10):
    features = [df[f'bid_sz_0{i}'].diff().fillna(0) - df[f'ask_sz_0{i}'].diff().fillna(0) for i in range(depth)]
    ofi_matrix = pd.concat(features, axis=1)
    ofi_matrix.columns = [f'OFI_L{i}' for i in range(depth)]
    pca = PCA(n_components=1)
    df['OFI_integrated'] = pca.fit_transform(ofi_matrix)
    return df

def add_lagged_features(df):
    df['OFI_best_level_lag1'] = df['OFI_best_level'].shift(1)
    df['OFI_multi_level_lag1'] = df['OFI_multi_level'].shift(1)
    df['OFI_integrated_lag1'] = df['OFI_integrated'].shift(1)
    return df
