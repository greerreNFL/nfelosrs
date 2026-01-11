## rmse calculation utilities ##

import pandas as pd
import numpy

from ..constants import SRS_RATING_COLUMNS


def calc_rmse(games, srs_file):
    '''
    Calcs an RMSE to margin by week
    '''
    ## output structure ##
    output = {}
    ## srs ratings are through the week, so the rating
    ## to join is the one from the previous week. Add 1
    srs_file['week'] = srs_file['week'] + 1
    ## join and calc for each ##
    for rating in SRS_RATING_COLUMNS:
        ## home ##
        temp = pd.merge(
            games[
                (games['week'] > 1) &
                (games['week'] < 19)
            ],
            srs_file[[
                'season', 'week', 'team', rating
            ]].rename(columns={
                'team' : 'home_team',
                rating : 'home_rating'
            }),
            on=['season', 'week', 'home_team']
        )
        ## away ##
        temp = pd.merge(
            temp,
            srs_file[[
                'season', 'week', 'team', rating
            ]].rename(columns={
                'team' : 'away_team',
                rating : 'away_rating'
            }),
            on=['season', 'week', 'away_team']
        )
        ## calc the margin ##
        temp['margin_pred'] = (
            temp['home_rating'] +
            temp['modeled_hfa'] -
            temp['away_rating']
        )
        ## calc SE ##
        temp['se'] = (temp['result'] - temp['margin_pred']) ** 2
        ## MSE ##
        temp = temp.groupby(['week']).agg(
            mse = ('se', 'mean')
        ).reset_index()
        ## iter and add to output ##
        for index, row in temp.iterrows():
            ## add week to the output if necessary ##
            if 'week' in output.keys():
                if row['week'] not in output['week']:
                    output['week'].append(row['week'])
            else:
                output['week'] = [row['week']]
            ## init rating if necessary ##
            if rating not in output.keys():
                output[rating] = [None] * 17
            ## populate ##
            output[rating][output['week'].index(row['week'])] = row['mse'] ** (1/2)
    ## create a df ##
    return pd.DataFrame(output).sort_values(
        by=['week'],
        ascending=[True]
    ).reset_index(drop=True)
