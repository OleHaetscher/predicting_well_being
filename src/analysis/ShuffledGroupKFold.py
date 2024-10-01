import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.utils import check_random_state
from sklearn.utils import shuffle
from sklearn.model_selection._split import BaseCrossValidator
import numpy as np

class ShuffledGroupKFold(BaseCrossValidator):
    """
    Custom GroupKFold class with shuffling and random state support.

    This class extends GroupKFold by adding the ability to shuffle the data
    before performing the group-based cross-validation. It takes `random_state`
    and `shuffle` arguments to control the shuffling.

    Attributes:
        n_splits: int, number of splits.
        shuffle: bool, whether to shuffle data before splitting.
        random_state: int or None, seed for shuffling.
    """

    def __init__(self, n_splits, random_state, shuffle=True):
        """
        Constructor method for ShuffledGroupKFold.

        Args:
            n_splits: Number of folds for cross-validation.
            shuffle: Whether to shuffle the data before splitting.
            random_state: Seed for the random number generator for shuffling.
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        Args:
            X: DataFrame or array-like, shape (n_samples, n_features).
            y: Array-like, shape (n_samples,) - Optional, used for compatibility.
            groups: Array-like, shape (n_samples,), group labels for the samples.

        Returns:
            Iterator: The training and testing indices for each fold.
        """
        groups = pd.Series(groups)
        ix = np.arange(len(groups))
        unique = np.unique(groups)
        np.random.RandomState(self.random_state).shuffle(unique)
        result = []

        for split in np.array_split(unique, self.n_splits):
            mask = groups.isin(split)
            train, test = ix[~mask], ix[mask]
            result.append((train, test))

        return result