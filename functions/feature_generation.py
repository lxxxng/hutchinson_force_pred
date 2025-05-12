import pandas as pd
import numpy as np

class FeatureGenerator:
    def __init__(self, window_size=1000, sampling_rate=None):
        self.window_size = window_size
        self.sampling_rate = sampling_rate

        # Add overall statistics across all selected features
    def add_global_statistics(self, df, feature_cols):
        df['global_mean'] = df[feature_cols].mean(axis=1)  # Row-wise mean
        df['global_std'] = df[feature_cols].std(axis=1)    # Row-wise standard deviation
        df['global_sum'] = df[feature_cols].sum(axis=1)    # Row-wise sum
        
        return df

    # Add mean and std for color-based sensor groups (red, blue, yellow)
    def add_color_statistics(self, df, feature_cols):
        color_groups = {'red': [], 'blue': [], 'yellow': []}
        
        for col in feature_cols:
            for color in color_groups:
                if f'_{color}' in col:
                    color_groups[color].append(col)  # Group columns by color
                    
        for color, cols in color_groups.items():
            df[f'{color}_mean'] = df[cols].mean(axis=1)  # Row-wise mean for each color group
            df[f'{color}_std'] = df[cols].std(axis=1)    # Row-wise std for each color group
            
        return df

    # Add rolling window mean and std for each feature column
    def add_rolling_features(self, df, feature_cols):
        for col in feature_cols:
            df[f'{col}_rolling_mean'] = df[col].rolling(window=self.window_size, center=True, min_periods=1).mean()
            df[f'{col}_rolling_std'] = df[col].rolling(window=self.window_size, center=True, min_periods=1).std()
            
        return df

    # Add temporal change (difference between consecutive values) for each feature
    def add_temporal_changes(self, df, feature_cols):
        for col in feature_cols:
            df[f'{col}_diff'] = df[col].diff().fillna(0)  # Difference from previous row
            
        return df

    # Add mean and std for sensor blocks 1-3 and 4-6
    def add_sensor_block_aggregations(self, df, feature_cols):
        
        block_1_3 = [col for col in feature_cols if col.startswith(('1_', '2_', '3_'))]  # Group 1-3
        block_4_6 = [col for col in feature_cols if col.startswith(('4_', '5_', '6_'))]  # Group 4-6
        
        df['block_1_3_mean'] = df[block_1_3].mean(axis=1)  # Mean of block 1-3
        df['block_4_6_mean'] = df[block_4_6].mean(axis=1)  # Mean of block 4-6
        df['block_1_3_std'] = df[block_1_3].std(axis=1)    # Std of block 1-3
        df['block_4_6_std'] = df[block_4_6].std(axis=1)    # Std of block 4-6
        return df


    ## try adding this if model does not perform well
    # def add_fourier_features(self, df, feature_cols):
    #     # Estimate sampling rate if not provided
    #     if self.sampling_rate is None and 'Time' in df.columns:
    #         time_diffs = df['Time'].diff().dropna()
    #         avg_interval = time_diffs.mean()
    #         self.sampling_rate = 1 / avg_interval
    #         print(f"Estimated Sampling Rate: {self.sampling_rate:.2f} Hz")

    #     # Create frequency bins
    #     freqs = np.round(np.fft.rfftfreq(self.window_size) * self.sampling_rate, 3)

    #     # Prepare summary columns only
    #     new_columns = {}
    #     for col in feature_cols:
    #         new_columns[f'{col}_max_freq'] = [np.nan] * len(df)
    #         new_columns[f'{col}_freq_weighted'] = [np.nan] * len(df)
    #         new_columns[f'{col}_pse'] = [np.nan] * len(df)

    #     # Compute FFT-based summary features
    #     for i in range(self.window_size, len(df)):
    #         for col in feature_cols:
    #             window_data = df[col].iloc[i - self.window_size:i].values
    #             real_ampl = np.abs(np.fft.rfft(window_data, self.window_size))

    #             # Max Frequency
    #             max_freq_index = np.argmax(real_ampl)
    #             new_columns[f'{col}_max_freq'][i] = freqs[max_freq_index]

    #             # Weighted Average Frequency
    #             new_columns[f'{col}_freq_weighted'][i] = np.sum(freqs * real_ampl) / np.sum(real_ampl)

    #             # Power Spectral Entropy (PSE)
    #             psd = np.square(real_ampl) / len(real_ampl)
    #             psd_pdf = psd / np.sum(psd)
    #             new_columns[f'{col}_pse'][i] = -np.sum(np.log(psd_pdf + 1e-10) * psd_pdf)  # Stability term

    #     # Combine new summary features into DataFrame
    #     df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    #     return df


    def generate_all(self, df):
        
        feature_cols = [col for col in df.columns if col != 'Time']
        
        df = self.add_global_statistics(df, feature_cols)
        df = self.add_color_statistics(df, feature_cols)
        df = self.add_rolling_features(df, feature_cols)
        df = self.add_temporal_changes(df, feature_cols)
        df = self.add_sensor_block_aggregations(df, feature_cols)
        # df = self.add_fourier_features(df, feature_cols)
        
        return df
