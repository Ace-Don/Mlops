# Apache Software License 2.0
# 
# Copyright (c) ZenML GmbH 2026. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

from typing import Sequence, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NADropper(TransformerMixin, BaseEstimator):
    """Support class to drop NA values in sklearn Pipeline."""

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series]):
        return X.dropna()


class ColumnsDropper(TransformerMixin, BaseEstimator):
    """Support class to drop specific columns in sklearn Pipeline."""

    def __init__(self, columns: Sequence[str]):
        self.columns = list(columns)

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series]):
        return X.drop(columns=self.columns)


class DataFrameCaster(TransformerMixin, BaseEstimator):
    """Support class to cast type back to pd.DataFrame in sklearn Pipeline."""

    def __init__(self, columns: Sequence[str]):
        self.columns = list(columns)

    def fit(self, X, y=None):
        # Set fitted attributes so sklearn can recognize this transformer as fitted.
        # (newer sklearn calls check_is_fitted on the Pipeline's final step)
        self.n_features_in_ = X.shape[1] if hasattr(X, "shape") else None
        self.is_fitted_ = True
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.columns)


"""
NADropper
The simplest of the three. 
Its only job is to drop any rows containing null/NaN values from the data. '
It has no configuration — it always does the same thing no matter what data you give it, which is why there's no __init__. '
You'd use this at the start of a pipeline to clean dirty data before any model sees it.

ColumnsDropper
Does exactly what the name says — drops specific columns from the DataFrame. 
Unlike NADropper it needs to know which columns to drop, so __init__ takes a list of column names and stores them. 
You'd use this to remove columns that aren't useful for your model — things like ID fields, redundant features, or anything you identified during EDA as having no predictive value.

DataFrameCaster
This one solves a specific sklearn problem. 
When data passes through sklearn transformers internally, it often gets converted to a numpy array — losing all the column names in the process. 
DataFrameCaster takes that raw array and converts it back into a proper pd.DataFrame with the correct column names restored. 
You'd typically put this at the end of a pipeline or after any step that strips column names, so downstream steps that rely on named columns still work correctly.

"""