import pandas as pd
import numpy
import pathlib

import nfelodcm as dcm

class DataLoader():
    ## this class loads, formats, and merges, necessary data ##
    def __init__(self):
        ## package path ##
        self.package_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
        ## states ##
        self.current_season, self.current_week = dcm.get_season_state()
        ## data frames ##
        self.db = dcm.load(['games', 'qbelo'])
        self.wts = None ## win total lines ##
        self.games = None ## fastr game file ##
        self.qbs = None ## nfeloqb file rankings ##
        self.wt_ratings = None ## win total ratings ##
        self.seasonal_srs = None ## season 
        self.weekly_srs = None ## where the weekly srs will be outputed ##
        ## repl dicts ##
        self.team_repl = {
            'LA': 'LAR',
            'LV': 'OAK',
            'STL': 'LAR',
            'SD': 'LAC',
            'WSH' : 'WAS'
        }
        ## init ##
        self.load_dfs()
        self.compute_simple_hfa()
    
    def load_dfs(self):
        ## load and format all initial dfs ##
        ## win totals ##
        self.wts = pd.read_csv(
            '{0}/nfelosrs/Manual Data/win_totals.csv'.format(self.package_dir),
            index_col=0
        )
        self.wts['team'] = self.wts['team'].replace(self.team_repl)
        ## games ##
        self.games = self.db['games'].copy()
        self.qbs = self.db['qbelo'].copy()
        ## repl qb names ##
        self.qbs['team1'] = self.qbs['team1'].replace(self.team_repl)
        self.qbs['team2'] = self.qbs['team2'].replace(self.team_repl)
        ## existing wts ##
        try:
            self.wt_ratings = pd.read_csv(
                '{0}/wt_ratings.csv'.format(self.package_dir),
                index_col=0
            )
        except:
            pass
        ## seasonal srs ##
        try:
            self.seasonal_srs = pd.read_csv(
                '{0}/seasonal_srs.csv'.format(self.package_dir),
                index_col=0
            )
        except:
            pass
        ## weekly srs ##
        try:
            self.weekly_srs = pd.read_csv(
                '{0}/weekly_srs.csv'.format(self.package_dir),
                index_col=0
            )
        except:
            pass
    
    def compute_simple_hfa(self):
        '''
        Calculates a simple rolling homefield advantage expecation
        '''
        ## average margin by season ##
        seasonal_margin = self.games[
            self.games['game_type'] == 'REG'
        ].groupby(['season']).agg(
            avg_margin = ('result', 'mean')
        ).reset_index()
        ## shift forward so no forward data is used ##
        seasonal_margin['last_margin'] = seasonal_margin['avg_margin'].shift()
        ## drop 2020 ##
        seasonal_margin = seasonal_margin[
            seasonal_margin['season']!=2020
        ].copy()
        ## calc a trailing rolling average ##
        seasonal_margin['rolling_hfa'] = seasonal_margin['last_margin'].ewm(span=5).mean()
        ## add 2020 ##
        seasonal_margin = pd.concat([
            seasonal_margin,
            pd.DataFrame([{
                'season' : 2020,
                'avg_margin' : numpy.nan,
                'last_margin' : numpy.nan,
                'rolling_hfa' : 0,
            }])
        ])
        ## add to games ##
        self.games = pd.merge(
            self.games,
            seasonal_margin[[
                'season', 'rolling_hfa'
            ]].rename(columns={
                'rolling_hfa' : 'modeled_hfa'
            }),
            on=['season'],
            how='left'
        )
