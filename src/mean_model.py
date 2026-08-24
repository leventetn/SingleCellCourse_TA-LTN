from anndata import AnnData
import pandas as pd
import scanpy as sc
import numpy as np
from pandas import DataFrame


class MeanModel:
    """Predicts the mean of the training data."""

    def __init__(self):
        self.mean_FC = None


    def train(
            self,
            train_X: AnnData,
            train_y: DataFrame,
    ):
        """Computes the mean log2FC of each gene in the training data
        train_X (unused): AnnData object containing the training features
        train_y: DataFrame object from scanpy.get.rank_genes_groups_df containing the training log2FCs"""



        self.mean_FC = train_y.groupby('names').logfoldchanges.mean()

        return self


    def predict(
            self,
            X: list[str],
    ):
        """Predicts the mean log2FC of each gene in the test data
        X: AnnData object containing the test features"""

        dfs = []

        for perturbation in X:
            df = DataFrame(self.mean_FC)
            df.insert(0, 'perturbation', perturbation)
            df.reset_index(drop=False, inplace=True)
            dfs.append(df)

        predictions = pd.concat(dfs, ignore_index=True)

        return predictions


    def get_RMSE(self, predictions, y):
        """Computes the root mean squared error between the predicted and actual values"""

        RMSE = np.sqrt(np.mean(((predictions.sort_values(by=['perturbation','names']).logfoldchanges.values
               - y.sort_values(by=['group','names']).logfoldchanges.values) ** 2)))

        return RMSE.astype('float64')
