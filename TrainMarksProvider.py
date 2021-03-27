import pandas as pd

from ResourceProviders.TrainPathProvider import TrainPathProvider


class TrainMarksProvider:
    def __init__(self):
        self.df = pd.read_csv(TrainPathProvider().get_train_df_path())

    def _get_max_rank(self):
        return max(self.df['Label'].str.split('|').str.len().values)

    def _get_by_rank(self, rank):
        mask = self.df['Label'].str.split('|').str.len() == rank
        return self.df[mask].index.values
