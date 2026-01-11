import pandas as pd
import numpy

from ... import Utilities as utils
from ...Utilities import get_package_dir, add_line_rating, ELO_CENTER


class WTRatings():
    ## model class for the wt model ##
    def __init__(self, wts, games, wt_ratings, rebuild=False, config_override=None):
        ## package path ##
        self.package_dir = get_package_dir()
        ## state ##
        self.wts_season = None
        self.wt_ratings_season = None
        ## config ##
        self.rebuild = rebuild
        self.config = utils.load_config('config.json', ['wt_ratings'])
        if config_override:
            self.config.update(config_override)
        ## dfs ##
        self.wts = wts
        self.games = games
        self.wt_ratings = wt_ratings
        ## init ##
        self.set_states()
        self.format_wts()
        
    ## SETUP FUNCS ##
    def set_states(self):
        ## set states to determine how many season to build win total ratings for ##
        if self.rebuild or self.wt_ratings is None:
            self.wt_ratings_season = self.wts['season'].min() - 1
        else:
            self.wt_ratings_season = self.wt_ratings['season'].max()
        self.wts_season = self.wts['season'].max()
    
    def format_wts(self):
        ## format win total lines ##
        self.wts = utils.add_odds_and_line_adj(self.wts, self.config['over_prob_logit_coef'])
        ## add line_rating (normalized to 16-game season, centered at 0) ##
        self.wts = add_line_rating(self.wts)
    
    def calc_ratings(self):
        ## vectorized calculation of wt_rating and wt_rating_elo ##
        ## build adjustment map with int keys ##
        adj_map = {int(k): v for k, v in self.config.get('wt_rating_adjustments', {}).items()}
        ## map season to adjustment coefficient, fillna(-0.40) for missing seasons ##
        adjustments = self.wts['season'].map(adj_map).fillna(-0.40)
        ## wt_rating = line_rating + line_rating * adjustment ##
        self.wts['wt_rating'] = self.wts['line_rating'] + self.wts['line_rating'] * adjustments
        ## wt_rating_elo = wt_rating * elo_scale + ELO_CENTER ##
        elo_scale = self.config.get('elo_scale', 56.0573)
        self.wts['wt_rating_elo'] = self.wts['wt_rating'] * elo_scale + ELO_CENTER
    
    def calc_sos(self, season):
        ## calculate SOS using line_ratings ##
        season_games = self.games[
            (self.games['season'] == season) &
            (self.games['game_type'] == 'REG')
        ].copy()
        season_wts = self.wts[self.wts['season'] == season][['team', 'line_rating']].copy()
        ## add ratings to games ##
        season_games = pd.merge(
            season_games,
            season_wts.rename(columns={'team': 'home_team', 'line_rating': 'home_rating'}),
            on='home_team',
            how='left'
        )
        season_games = pd.merge(
            season_games,
            season_wts.rename(columns={'team': 'away_team', 'line_rating': 'away_rating'}),
            on='away_team',
            how='left'
        )
        ## flatten ##
        flat = pd.concat([
            season_games[['home_team', 'away_rating']].rename(columns={
                'home_team': 'team',
                'away_rating': 'opponent_rating'
            }),
            season_games[['away_team', 'home_rating']].rename(columns={
                'away_team': 'team',
                'home_rating': 'opponent_rating'
            })
        ])
        ## group by team and calc sos ##
        sos_df = flat.groupby(['team']).agg(
            sos = ('opponent_rating', 'mean'),
        ).reset_index()
        return sos_df
    
    def update(self):
        print('Updating Win Total Ratings...')
        if self.rebuild:
            print('     Rebuild set to True. Will rebuild from first available win total season...')
        if self.wts_season == self.wt_ratings_season:
            print('     Win Totals and Win Total Ratings are both current through {0}'.format(self.wts_season))
            return
        ## calculate ratings vectorized ##
        self.calc_ratings()
        ## get seasons to process ##
        seasons_to_process = [
            s for s in self.wts['season'].unique()
            if s > self.wt_ratings_season
        ]
        ## collect new data ##
        new_data = []
        for season in seasons_to_process:
            print('     Processing Season {0}...'.format(season))
            season_wts = self.wts[self.wts['season'] == season].copy()
            sos_df = self.calc_sos(season)
            for _, row in season_wts.iterrows():
                sos_val = sos_df[sos_df['team'] == row['team']]['sos'].values
                new_data.append({
                    'team': row['team'],
                    'season': season,
                    'line_rating': row['line_rating'],
                    'wt_rating': row['wt_rating'],
                    'wt_rating_elo': row['wt_rating_elo'],
                    'sos': sos_val[0] if len(sos_val) > 0 else numpy.nan
                })
        ## create new df and merge with wts data ##
        if len(new_data) > 0:
            new_df = pd.DataFrame(new_data)
            new_df = pd.merge(
                new_df,
                self.wts.rename(columns={
                    'over_prob_vf': 'over_probability',
                    'under_prob_vf': 'under_probability'
                }).drop(columns=['logit_over_prob_vf', 'line_rating', 'wt_rating', 'wt_rating_elo']),
                on=['season', 'team'],
                how='left'
            )
            ## round ratings ##
            for col in ['line_rating', 'wt_rating', 'wt_rating_elo', 'sos', 'hold', 'over_probability', 'under_probability', 'line_adj']:
                if col in new_df.columns:
                    new_df[col] = new_df[col].round(4)
            ## add to existing or replace ##
            if self.wt_ratings is None or self.rebuild:
                self.wt_ratings = new_df
            else:
                self.wt_ratings = pd.concat([self.wt_ratings, new_df])
                self.wt_ratings = self.wt_ratings.reset_index(drop=True)
            ## save ##
            self.wt_ratings.to_csv('{0}/wt_ratings.csv'.format(self.package_dir))
