import pandas as pd
import numpy
import pathlib


class DataLoader():
    ## this class loads, formats, and merges, necessary data ##
    def __init__(self):
        ## package path ##
        self.package_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
        ## states ##
        self.current_season = None ## season of last fully played week ##
        self.current_week = None ## week of last fully played week ##
        ## data frames ##
        self.wts = None ## win total lines ##
        self.games = None ## nflfastR game file##
        self.wt_ratings = None ## win total ratings ##
        self.seasonal_srs = None ## season 
        self.weekly_srs = None ## where the weekly srs will be outputed ##
        ## locations of external data ##
        self.game_data_url = 'https://github.com/nflverse/nfldata/raw/master/data/games.csv'
        ## repl dicts ##
        self.team_repl = {
            'LA': 'LAR',
            'LV': 'OAK',
            'STL': 'LAR',
            'SD': 'LAC',
        }
        ## init ##
        self.load_dfs()
        self.set_season_state()
    
    def load_dfs(self):
        ## load and format all initial dfs ##
        ## win totals ##
        self.wts = pd.read_csv(
            '{0}/nfelosrs/Manual Data/win_totals.csv'.format(self.package_dir),
            index_col=0
        )
        self.wts['team'] = self.wts['team'].replace(self.team_repl)
        ## games ##
        self.games = pd.read_csv(
            self.game_data_url
        )
        self.games['home_team'] = self.games['home_team'].replace(self.team_repl)
        self.games['away_team'] = self.games['away_team'].replace(self.team_repl)
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
    
    def set_season_state(self):
        ## set the current season and week for the last fully played week ##
        ## determine which games have been played ##
        self.games['game_played'] = numpy.where(
            pd.isnull(self.games['result']),
            1,
            0
        )
        ## define which weeks have no unplayed games ##
        weeks = self.games.groupby(['season', 'week']).agg(
            completed = ('game_played', 'min')
        ).reset_index()
        ## ensure sorting is correct ##
        weeks = weeks.sort_values(
            by=['season', 'week'],
            ascending=[True, True]
        ).reset_index(drop = True)
        ## get last record with no unplayed games ##
        last_week = weeks[weeks['completed'] == 1].tail(1)
        ## set season and week ##
        self.current_season = last_week.iloc[0]['season']
        self.current_week = last_week.iloc[0]['week']
