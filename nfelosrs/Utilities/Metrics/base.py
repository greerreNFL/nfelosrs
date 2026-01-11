## base class for metrics calculations ##

from abc import ABC, abstractmethod
import pandas as pd
import statsmodels.api as sm

from ..constants import SRS_RATING_COLUMNS


class MetricCalculator(ABC):
    '''Base class for SRS metric calculations'''
    
    def __init__(self, rating_columns=None):
        self.rating_columns = rating_columns or SRS_RATING_COLUMNS
    
    @abstractmethod
    def calculate(self, *args, **kwargs):
        '''Calculate the metric - must be implemented by subclasses'''
        pass


def grouped_rsq(grouped_df, rating_columns=None):
    '''
    Calculates rsq for each measure from a grouped df. Call with apply.
    
    Parameters:
        grouped_df: DataFrame group from groupby operation
        rating_columns: list of rating column names to calculate rsq for
        
    Returns:
        pd.Series with rsq values for each rating column
    '''
    columns = rating_columns or SRS_RATING_COLUMNS
    output = {}
    for rating in columns:
        model = sm.OLS(
            grouped_df['future_mov'],
            grouped_df[[rating, 'const']],
            hasconst=True
        ).fit()
        output['{0}_rsq'.format(rating)] = model.rsquared
    return pd.Series(output)
