import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

class RegressionAlgorithms:
    def forward_selection(self, max_features, X_train, y_train):
        """
        Perform forward feature selection using DecisionTreeRegressor based on R2 score.
        
        Parameters:
        - max_features (int): Maximum number of features to select.
        - X_train (pd.DataFrame): Training features.
        - y_train (pd.Series or pd.DataFrame): Target values.

        Returns:
        - selected_features (list): Final selected features.
        - ordered_features (list): Ordered selection process.
        - ordered_scores (list): Corresponding R2 scores.
        """
        selected_features = []
        ordered_features = []
        ordered_scores = []
        prev_best_perf = float("-inf")

        for i in range(max_features):
            features_left = list(set(X_train.columns) - set(selected_features))
            best_perf = float("-inf")
            best_feature = None

            for f in features_left:
                current_features = selected_features + [f]
                model = DecisionTreeRegressor()
                model.fit(X_train[current_features], y_train)
                y_pred = model.predict(X_train[current_features])
                perf = r2_score(y_train, y_pred)

                if perf > best_perf:
                    best_perf = perf
                    best_feature = f

            if best_feature is not None:
                selected_features.append(best_feature)
                ordered_features.append(best_feature)
                ordered_scores.append(best_perf)

        return selected_features, ordered_features, ordered_scores
