from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def prepare_data(self, df: pd.DataFrame):
        # Feature Engineering: Adding a 'temp_vibration_ratio'
        df['stress_index'] = df['temperature'] * df['vibration']
        
        # Scaling numerical values
        cols_to_scale = ['temperature', 'vibration', 'pressure', 'stress_index']
        df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
        
        return df
