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
        self.db = dcm.load(['games'])
        self.wts = None ## win total lines ##
        self.games = None
        self.wt_ratings = None ## win total ratings ##
        self.seasonal_srs = None ## season 
        self.weekly_srs = None ## where the weekly srs will be outputed ##
        ## repl dicts ##
        self.team_repl = {
            'LA': 'LAR',
            'LV': 'OAK',
            'STL': 'LAR',
            'SD': 'LAC',
        }
        ## init ##
        self.load_dfs()
    
    def load_dfs(self):
        ## load and format all initial dfs ##
        ## win totals ##
        self.wts = pd.read_csv(
            '{0}/nfelosrs/Manual Data/win_totals.csv'.format(self.package_dir),
            index_col=0
        )
        self.wts['team'] = self.wts['team'].replace(self.team_repl)
        ## games ##
        self.games = self.db['games']
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

