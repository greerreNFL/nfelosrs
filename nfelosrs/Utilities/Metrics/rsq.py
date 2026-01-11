## rsq calculation utilities ##

import pandas as pd
import numpy

from .base import grouped_rsq


def calc_future_margin(srs_ratings):
    '''
    Add a col for future margin of victory
    '''
    srs_ratings['future_mov'] = (
        ## delta in total margin
        (
            ## total mov at end of year ##
            srs_ratings.groupby(['team', 'season'])['gp'].transform('last') *
            srs_ratings.groupby(['team', 'season'])['avg_mov'].transform('last')
        ) -
        ## less current total mov
        srs_ratings['gp'] * srs_ratings['avg_mov'] ## total current margin ##
    ) / (
        ## delta in GP to get an average ##
        srs_ratings.groupby(['team', 'season'])['gp'].transform('last') -
        srs_ratings['gp']
    )
    return srs_ratings


def calc_rsq(srs_rating_df):
    '''
    Wrapper to calculate the RSQ to future mov for each week and rating type
    '''
    temp = srs_rating_df[
        srs_rating_df['gp'] < 16
    ].copy()
    ## add future mov ##
    temp = calc_future_margin(temp)
    temp = temp[
        ~pd.isnull(temp['future_mov'])
    ].copy()
    temp['const'] = 1
    ## get the aggregation ##
    agg = temp.groupby(['gp']).apply(grouped_rsq)
    agg = agg.reset_index()
    agg = agg.sort_values(
        by=['gp'],
        ascending=[True]
    ).reset_index(drop=True)
    ## return ##
    return agg
