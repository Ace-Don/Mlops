import pandas as pd
from typing_extensions import Annotated
from zenml import step
import mlflow

def last_n_days_transaction_count(df_group, n):
    temp = pd.Series(df_group.index, index=df_group.date).sort_index()
    count_n_days = temp.rolling(f'{n}d').count() - 1
    count_n_days.index = temp.values
    df_group[f'count_{n}_days'] = count_n_days.reindex(df_group.index)
    return df_group

@step
def feature_engineer(dataset: pd.DataFrame) -> Annotated[pd.DataFrame, "engineered_dataset"]:
    """Feature engineering step before data splitting.
    
    Creates rolling window transaction counts.
    """
    mlflow.set_experiment("feature_engineering")
    mlflow.start_run(run_name="feature_engineer")
    
    if 'zipMerchant' in dataset.columns:
        dataset.drop(['zipMerchant', 'zipcodeOri'], axis=1, inplace=True)
        
    dataset['step'] = 1577836800 + dataset['step'] * 3600 * 24
    dataset['step'] = pd.to_datetime(dataset['step'], unit='s')
    dataset.rename(columns={'step': 'date'}, inplace=True)
    
    dataset = dataset.sort_values(['customer', 'date'])
    # In older pandas, apply works, in newer it might raise a deprecation warning about groupings.
    # To fix, we can include_groups=False or just ignore.
    try:
        dataset = dataset.groupby('customer', group_keys=False).apply(lambda x: last_n_days_transaction_count(x, 1)).reset_index(drop=True)
        dataset = dataset.groupby('customer', group_keys=False).apply(lambda x: last_n_days_transaction_count(x, 7)).reset_index(drop=True)
        dataset = dataset.groupby('customer', group_keys=False).apply(lambda x: last_n_days_transaction_count(x, 30)).reset_index(drop=True)
    except TypeError:
        # Fallback for older pandas where group_keys=False acts differently
        dataset = dataset.groupby('customer').apply(lambda x: last_n_days_transaction_count(x, 1)).reset_index(drop=True)
        dataset = dataset.groupby('customer').apply(lambda x: last_n_days_transaction_count(x, 7)).reset_index(drop=True)
        dataset = dataset.groupby('customer').apply(lambda x: last_n_days_transaction_count(x, 30)).reset_index(drop=True)
        
    dataset.drop(['customer', 'merchant', 'date'], axis=1, inplace=True)
    
    mlflow.log_param("engineered_columns", list(dataset.columns))
    mlflow.end_run()
    
    return dataset
