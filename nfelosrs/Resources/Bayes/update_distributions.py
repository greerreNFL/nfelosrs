import pandas as pd
import numpy
import pathlib
import json

def calc_margin_distributions(games):
    '''
    Determines the standard deviation of NFL game margins against
    spreads to estimate single game uncertainty
    '''
    ## only played games ##
    games = games[
        ~pd.isnull(games['result'])
    ].copy()
    ## calc error ##
    games['spread_error'] = games['result'] - games['spread_line']
    ## return std ##
    return games['spread_error'].std()


def calc_ranking_distributions(qbs):
    '''
    Determines the standard deviation of preseason rankings using 
    the qbelo to estiamte pre season ranking uncertainty
    '''
    ## only take played games in teh same time period as the games file ##
    qbs = qbs[
        (qbs['season'] >= 1999) &
        (~pd.isnull(qbs['score1']))
    ].copy()
    ## flatten teams by season ##
    qbs = pd.concat([
        qbs[[
            'season', 'date', 'team1', 'qbelo1_pre',
            'qbelo1_post', 'qb1_adj'
        ]].rename(columns={
            'team1' : 'team',
            'qbelo1_pre' : 'qbelo_pre',
            'qbelo1_post' : 'qbelo_post',
            'qb1_adj' : 'qb_adj'
        }),
        qbs[[
            'season', 'date', 'team2', 'qbelo2_pre',
            'qbelo2_post', 'qb2_adj'
        ]].rename(columns={
            'team2' : 'team',
            'qbelo2_pre' : 'qbelo_pre',
            'qbelo2_post' : 'qbelo_post',
            'qb2_adj' : 'qb_adj'
        })
    ])
    ## chrono sort ##
    qbs = qbs.sort_values(
        by=['date', 'team'],
        ascending=[True, True]
    ).reset_index(drop=True)
    ## since we are measuring ranking uncertainty for rankings set as points
    ## against and average, we do not want to include QB variance and we need to rescale
    ## the values and add back the QB adj ##
    qbs['qbelo_pre'] = ((qbs['qbelo_pre'] + qbs['qb_adj']) - 1505) / 25
    qbs['qbelo_post'] = ((qbs['qbelo_post'] + qbs['qb_adj']) - 1505) / 25
    ## get the starting and ending values for the seaosn plus games played ##
    seasons = qbs.groupby(['season', 'team']).agg(
        gp = ('season', 'count'),
        pre_season_rank=('qbelo_pre', lambda x: x.head(1)),
        end_of_season_rank=('qbelo_post', lambda x: x.tail(1)),
    ).reset_index(drop=True)
    ## drop incomplete seasons ##
    seasons = seasons[
        seasons['gp']>14
    ].copy()
    ## return the std of the delta between actual and expected ##
    return (seasons['end_of_season_rank']-seasons['pre_season_rank']).std()



def update_distributions(games, qbs):
    '''
    Wrapper that updates the config file
    '''
    ## get values ##
    rankings = calc_ranking_distributions(qbs)
    margins = calc_margin_distributions(games)
    ## create new dict ##
    ## note numpy values need to be converted to python native ##
    ## for json serialization, so use item() ##
    distros = {
        'rankings' : rankings.item(),
        'margins' : margins.item()
    }
    ## write to package ##
    folder = pathlib.Path(__file__).parent.resolve()
    with open('{0}/distributions.json'.format(folder), 'w') as fp:
        json.dump(distros, fp, indent=2)
