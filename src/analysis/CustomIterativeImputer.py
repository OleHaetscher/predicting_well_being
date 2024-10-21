import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.utils._mask import _get_mask
from sklearn.base import clone
from scipy import stats
import pandas as pd


class CustomIterativeImputer(IterativeImputer):
    def __init__(self,  # default parameters, copied
                 estimator=None,
                 missing_values=np.nan,
                 sample_posterior=False,
                 max_iter=10,
                 tol=1e-3,
                 n_nearest_features=None,
                 initial_strategy="mean",
                 imputation_order="ascending",
                 skip_complete=False,
                 min_value=-np.inf,
                 max_value=np.inf,
                 verbose=0,
                 random_state=None,
                 add_indicator=False,
                 keep_empty_features=False,
                 categorical_idx=None,
                 pmm_k=5):
        # Pass all relevant parameters to the parent class (IterativeImputer)
        super().__init__(
            estimator=estimator,
            missing_values=missing_values,
            sample_posterior=sample_posterior,
            max_iter=max_iter,
            tol=tol,
            n_nearest_features=n_nearest_features,
            initial_strategy=initial_strategy,
            imputation_order=imputation_order,
            skip_complete=skip_complete,
            min_value=min_value,
            max_value=max_value,
            verbose=verbose,
            random_state=random_state,
            add_indicator=add_indicator,
            keep_empty_features=keep_empty_features
        )
        # Initialize custom parameter
        self.categorical_idx = categorical_idx
        self.pmm_k = pmm_k

    def _impute_one_feature(
        self,
        X_filled,
        mask_missing_values,
        feat_idx,
        neighbor_feat_idx,
        estimator=None,
        fit_mode=True,
        params=None,
    ):
        """Impute a single feature from the others provided.

        This function predicts the missing values of one of the features using
        the current estimates of all the other features.

        Parameters
        ----------
        X_filled : ndarray or DataFrame
            Input data with the most recent imputations.

        mask_missing_values : ndarray
            Input data's missing indicator matrix.

        feat_idx : int
            Index of the feature currently being imputed.

        neighbor_feat_idx : ndarray
            Indices of the features to be used in imputing `feat_idx`.

        estimator : object
            The estimator to use at this step of the round-robin imputation.
            If None, it will be cloned from self._estimator.

        fit_mode : boolean, default=True
            Whether to fit and predict with the estimator or just predict.

        params : dict
            Additional params routed to the individual estimator.

        Returns
        -------
        X_filled : ndarray or DataFrame
            Input data with `X_filled[missing_row_mask, feat_idx]` updated.

        estimator : estimator with sklearn API
            The fitted estimator used to impute
            `X_filled[missing_row_mask, feat_idx]`.
        """
        if estimator is None and fit_mode is False:
            raise ValueError(
                "If fit_mode is False, then an already-fitted "
                "estimator should be passed in."
            )

        if estimator is None:
            estimator = clone(self._estimator)

        missing_row_mask = mask_missing_values[:, feat_idx]
        if fit_mode:
            X_train = safe_indexing(
                safe_indexing(X_filled, neighbor_feat_idx, axis=1),
                ~missing_row_mask,
                axis=0,
            )
            y_train = safe_indexing(
                safe_indexing(X_filled, feat_idx, axis=1),
                ~missing_row_mask,
                axis=0,
            )

            # Passing feat_idx and neighbor_feat_idx to the estimator
            if hasattr(estimator, "set_params"):
                estimator.set_params(
                    feat_idx=feat_idx,
                    neighbor_feat_idx=neighbor_feat_idx,
                )
            if params is None:
                estimator.fit(X_train, y_train)
            else:
                estimator.fit(X_train, y_train, **params)

        else:
            # In case fit_mode=False, we still need X_train and y_train (observed values)  # TODO remove if condition?
            X_train = safe_indexing(
                safe_indexing(X_filled, neighbor_feat_idx, axis=1),
                ~missing_row_mask,
                axis=0,
            )

            y_train = safe_indexing(
                safe_indexing(X_filled, feat_idx, axis=1),
                ~missing_row_mask,
                axis=0,
            )

        # If no missing values, don't predict
        if np.sum(missing_row_mask) == 0:
            return X_filled, estimator

        X_test = safe_indexing(
            safe_indexing(X_filled, neighbor_feat_idx, axis=1),
            missing_row_mask,
            axis=0,
        )
        # Get posterior samples (this does not work with the models we currently use)
        if self.sample_posterior:
            mus, sigmas = estimator.predict(X_test, return_std=True)
            imputed_values = np.zeros(mus.shape, dtype=X_filled.dtype)
            # Handle non-positive sigmas and out-of-bounds mus
            positive_sigmas = sigmas > 0
            imputed_values[~positive_sigmas] = mus[~positive_sigmas]
            mus_too_low = mus < self._min_value[feat_idx]
            imputed_values[mus_too_low] = self._min_value[feat_idx]
            mus_too_high = mus > self._max_value[feat_idx]
            imputed_values[mus_too_high] = self._max_value[feat_idx]
            # Sample from truncated normal distribution
            inrange_mask = positive_sigmas & ~mus_too_low & ~mus_too_high
            mus = mus[inrange_mask]
            sigmas = sigmas[inrange_mask]
            a = (self._min_value[feat_idx] - mus) / sigmas
            b = (self._max_value[feat_idx] - mus) / sigmas

            truncated_normal = stats.truncnorm(a=a, b=b, loc=mus, scale=sigmas)
            imputed_values[inrange_mask] = truncated_normal.rvs(
                random_state=self.random_state_
            )

            # Post-process for binary features
            if feat_idx in self.categorical_idx:
                # Threshold the imputed values to 0 or 1
                imputed_values = (imputed_values >= 0.5).astype(int)

        # If not sample posterior, use predictive mean matching (this works with all models)
        else:
            is_binary = np.array_equal(np.unique(y_train), [0, 1])
            # If dealing with binary variables, use predict_proba
            if is_binary:
                # For binary variables, predict_proba gives probabilities
                imputed_probs = estimator.predict_proba(X_test)[:, 1]  # Get probabilities for class 1
                y_pred_obs = estimator.predict_proba(X_train)[:, 1]  # Observed probabilities for class 1
                y_pred_mis = imputed_probs
            else:
                # For continuous variables, use regular predict
                imputed_values = estimator.predict(X_test, return_std=False)
                y_pred_obs = estimator.predict(X_train, return_std=False)
                y_pred_mis = imputed_values

            # Initialize array for PMM-imputed values
            imputed_values_pmm = np.empty_like(y_pred_mis)

            # Use the class-level random_state for reproducibility
            rng = np.random.RandomState(self.random_state)

            for i, y_pred in enumerate(y_pred_mis):
                # Compute distances between y_pred and all y_pred_obs
                distances = np.abs(y_pred_obs - y_pred)
                # Find indices of k nearest neighbors
                nn_indices = np.argsort(distances)[:self.pmm_k]
                # Randomly select one of the k nearest observed values
                imputed_value = rng.choice(y_train[nn_indices])

                if is_binary:
                    # Get the binary value (0 or 1) from y_train corresponding to the selected neighbor
                    imputed_value = (imputed_value >= 0.5).astype(int)  # Ensure it's 0 or 1 based on threshold

                imputed_values_pmm[i] = imputed_value

            # Replace imputed_values with PMM-imputed values
            imputed_values = imputed_values_pmm

        # Update the feature
        safe_assign(
            X_filled,
            imputed_values,
            row_indexer=missing_row_mask,
            column_indexer=feat_idx,
        )
        return X_filled, estimator


def safe_indexing(X, indices, axis=0):
    """Safely index X along the specified axis for both 1D and 2D arrays or DataFrames."""
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        if axis == 0:
            return X.iloc[indices]
        elif axis == 1:
            return X.iloc[:, indices]
        else:
            raise ValueError(f"Invalid axis {axis} for pandas object.")

    elif isinstance(X, np.ndarray):
        X = np.asarray(X)  # Ensure it's a NumPy array

        if X.ndim == 1:
            # If 1D, we can only index along axis 0 (rows)
            if axis != 0:
                raise IndexError("Axis out of bounds for 1D array.")
            return X[indices]

        elif X.ndim == 2:
            # If 2D, index appropriately based on the axis
            if axis == 0:
                return X[indices, :]  # Index rows
            elif axis == 1:
                return X[:, indices]  # Index columns
            else:
                raise ValueError(f"Invalid axis {axis} for 2D array.")

    elif isinstance(X, list):
        # For lists, we can only index by rows (axis=0)
        if axis != 0:
            raise ValueError("safe_indexing only supports axis=0 for lists.")
        return [X[i] for i in indices]

    else:
        raise ValueError(f"Unsupported data type: {type(X)}")


def safe_assign(X, values, row_indexer=None, column_indexer=None):
    """Safely assign values to X at specified indices."""
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        if row_indexer is not None and column_indexer is not None:
            X.iloc[row_indexer, column_indexer] = values
        elif row_indexer is not None:
            X.iloc[row_indexer] = values
        elif column_indexer is not None:
            X.iloc[:, column_indexer] = values
        else:
            raise ValueError("Either row_indexer or column_indexer must be provided.")

    elif isinstance(X, np.ndarray):
        if row_indexer is not None and column_indexer is not None:
            # Fix: Ensure values match the shape of the slice
            if np.isscalar(column_indexer):
                # Reshape the values if assigning to a single column
                values = np.asarray(values).reshape(-1, 1)
            X[np.ix_(row_indexer, [column_indexer])] = values
        elif row_indexer is not None:
            X[row_indexer, :] = values
        elif column_indexer is not None:
            X[:, column_indexer] = values
        else:
            raise ValueError("Either row_indexer or column_indexer must be provided.")

    else:
        raise ValueError("Unsupported data type for safe_assign")