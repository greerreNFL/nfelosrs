import pandas as pd
import numpy
import json
import statsmodels.api as sm

import nfelodcm as dcm

from ... import Utilities as utils
from ...Utilities import (
    get_package_dir, add_line_rating, 
    TEAM_REPLACEMENTS, ELO_CENTER, ELO_TO_POINTS_DIVISOR
)


class WTRatingsTrainer():
    ## trains regression coefficients for win total ratings ##
    def __init__(self):
        ## package path ##
        self.package_dir = get_package_dir()
        ## config ##
        self.config = utils.load_config('config.json', ['wt_ratings'])
        ## data ##
        self.db = dcm.load(['games', 'qbelo'])
        self.games = None
        self.qbelo_spine = None
        self.wts = None
        ## trained params ##
        self.wt_rating_adjustments = {}
        self.elo_scale = None
        ## init ##
        self.load_data()
    
    def load_data(self):
        ## games with week info ##
        self.games = self.db['games'][['game_id', 'season', 'week']].copy()
        ## qbelo data - flatten and create spine with forward fill ##
        qbelo_raw = self.db['qbelo'].copy()
        self.qbelo_spine = self._create_qbelo_spine(qbelo_raw)
        ## win totals ##
        self.wts = pd.read_csv(
            '{0}/nfelosrs/Manual Data/win_totals.csv'.format(self.package_dir),
            index_col=0
        )
        self.wts['team'] = self.wts['team'].replace(TEAM_REPLACEMENTS)
        ## add line_adj and line_rating to win totals ##
        self.wts = utils.add_odds_and_line_adj(self.wts, self.config['over_prob_logit_coef'])
        self.wts = add_line_rating(self.wts)
    
    def _create_qbelo_spine(self, qbelo_raw):
        '''
        Creates a complete season x week x team spine with qbelo values.
        Forward fills then backfills to handle bye weeks and missing early weeks.
        '''
        ## only use games that have been played ##
        qbelo_raw = qbelo_raw[~pd.isnull(qbelo_raw['score1'])].copy()
        ## flatten to team level ##
        flat = pd.concat([
            qbelo_raw[[
                'game_id', 'season', 'team1', 'qbelo1_pre', 'qbelo1_post'
            ]].rename(columns={
                'team1': 'team',
                'qbelo1_pre': 'qbelo_pre',
                'qbelo1_post': 'qbelo_post'
            }),
            qbelo_raw[[
                'game_id', 'season', 'team2', 'qbelo2_pre', 'qbelo2_post'
            ]].rename(columns={
                'team2': 'team',
                'qbelo2_pre': 'qbelo_pre',
                'qbelo2_post': 'qbelo_post'
            })
        ])
        flat['team'] = flat['team'].replace(TEAM_REPLACEMENTS)
        ## join with games to get week ##
        flat = pd.merge(
            flat,
            self.games[['game_id', 'week']],
            on='game_id',
            how='left'
        )
        ## convert qbelo to points scale ##
        flat['qbelo_post_pts'] = (flat['qbelo_post'] - ELO_CENTER) / ELO_TO_POINTS_DIVISOR
        flat['qbelo_pre_pts'] = (flat['qbelo_pre'] - ELO_CENTER) / ELO_TO_POINTS_DIVISOR
        ## create complete spine of season x week combinations ##
        seasons = flat['season'].unique()
        weeks = list(range(1, 19))
        teams = flat['team'].unique()
        spine = pd.DataFrame([
            {'season': s, 'week': w, 'team': t}
            for s in seasons for w in weeks for t in teams
        ])
        ## join qbelo values to spine ##
        spine = pd.merge(
            spine,
            flat[['season', 'week', 'team', 'qbelo_pre', 'qbelo_post', 'qbelo_pre_pts', 'qbelo_post_pts']],
            on=['season', 'week', 'team'],
            how='left'
        )
        ## forward fill then backfill within each team-season ##
        spine = spine.sort_values(by=['season', 'team', 'week'])
        for col in ['qbelo_pre', 'qbelo_post', 'qbelo_pre_pts', 'qbelo_post_pts']:
            spine[col] = spine.groupby(['season', 'team'])[col].ffill()
            spine[col] = spine.groupby(['season', 'team'])[col].bfill()
        return spine
    
    def train_mean_reversion_coefficient(self, target_season):
        ## train mean reversion coefficient using 6-year rolling window ##
        training_seasons = list(range(target_season - 6, target_season))
        ## filter line_ratings to applicable seasons ##
        wts_filtered = self.wts[self.wts['season'].isin(training_seasons)][
            ['season', 'team', 'line_rating']
        ].copy()
        ## filter qbelos (in points) for week 4 ##
        qbelo_w4 = self.qbelo_spine[
            (self.qbelo_spine['season'].isin(training_seasons)) &
            (self.qbelo_spine['week'] == 4)
        ][['season', 'team', 'qbelo_post_pts']].copy()
        ## merge ##
        merged = pd.merge(
            wts_filtered,
            qbelo_w4,
            on=['season', 'team'],
            how='inner'
        )
        ## validate no nans ##
        if merged[['line_rating', 'qbelo_post_pts']].isna().any().any():
            merged = merged.dropna()
        if len(merged) < 50:
            return None
        ## regression 1: line_rating -> qbelo_pts to get scale ##
        model1 = sm.OLS(merged['qbelo_post_pts'], merged['line_rating']).fit()
        coef1 = model1.params.iloc[0]
        ## calculate error: preseason rating minus actual qbelo ##
        merged['error'] = merged['line_rating'] - merged['qbelo_post_pts']
        ## regression 2: line_rating -> error ##
        model2 = sm.OLS(merged['error'], merged['line_rating']).fit()
        coef2 = model2.params.iloc[0]
        ## descale ##
        adjustment = coef2 / coef1 if coef1 > 0.001 else coef2
        return adjustment
    
    def train_elo_scale(self, debug=False):
        ## train global elo_scale using all historical data ##
        ## uses qbelo_pre (preseason elo) to scale adjusted wt_rating ##
        available_wt_seasons = set(self.wts['season'].unique())
        available_qbelo_seasons = set(self.qbelo_spine['season'].unique())
        all_seasons = list(available_wt_seasons & available_qbelo_seasons)
        if debug:
            print('    All seasons: {0}'.format(len(all_seasons)))
        ## filter qbelo_pre for week 1 ##
        qbelo_w1 = self.qbelo_spine[
            (self.qbelo_spine['season'].isin(all_seasons)) &
            (self.qbelo_spine['week'] == 1)
        ][['season', 'team', 'qbelo_pre']].copy()
        if debug:
            print('    Qbelo W1 rows: {0}'.format(len(qbelo_w1)))
            print('    Qbelo W1 NaNs: {0}'.format(qbelo_w1['qbelo_pre'].isna().sum()))
        ## filter wt_ratings (adjusted line_ratings) ##
        wts_all = self.wts[self.wts['season'].isin(all_seasons)][
            ['season', 'team', 'wt_rating']
        ].copy()
        if debug:
            print('    WTs rows: {0}'.format(len(wts_all)))
        ## merge ##
        merged = pd.merge(
            wts_all,
            qbelo_w1,
            on=['season', 'team'],
            how='inner'
        )
        if debug:
            print('    Merged rows: {0}'.format(len(merged)))
            print('    Merged NaNs in qbelo_pre: {0}'.format(merged['qbelo_pre'].isna().sum()))
            print('    Merged NaNs in wt_rating: {0}'.format(merged['wt_rating'].isna().sum()))
        ## regression: (qbelo_pre - ELO_CENTER) = elo_scale * wt_rating (no constant) ##
        merged['qbelo_centered'] = merged['qbelo_pre'] - ELO_CENTER
        if debug:
            print('    qbelo_centered: min={0:.2f}, max={1:.2f}'.format(
                merged['qbelo_centered'].min(), merged['qbelo_centered'].max()
            ))
            print('    wt_rating: min={0:.2f}, max={1:.2f}'.format(
                merged['wt_rating'].min(), merged['wt_rating'].max()
            ))
        model = sm.OLS(merged['qbelo_centered'], merged['wt_rating']).fit()
        elo_scale = model.params.iloc[0]
        if debug:
            print('    elo_scale: {0}'.format(elo_scale))
        return round(elo_scale, 4)
    
    def train_all(self, start_season=2009, end_season=None, debug=False):
        if end_season is None:
            end_season = self.wts['season'].max()
        print('Training WTRatings coefficients...')
        ## train mean reversion coefficients ##
        print('  Training mean reversion coefficients...')
        for season in range(start_season, end_season + 1):
            print('  Season {0}:'.format(season))
            adj = self.train_mean_reversion_coefficient(season)
            if adj is not None:
                self.wt_rating_adjustments[str(season)] = adj
                print('    adjustment = {0:.6f}'.format(adj))
            else:
                print('    insufficient data, skipping')
        ## apply adjustments to compute wt_rating ##
        coef_map = {int(k): v for k, v in self.wt_rating_adjustments.items()}
        self.wts['wt_rating'] = (
            self.wts['line_rating'] + 
            self.wts['line_rating'] * self.wts['season'].map(coef_map).fillna(0)
        )
        ## train elo scale using adjusted wt_rating ##
        print('  Training elo_scale...')
        self.elo_scale = self.train_elo_scale(debug=debug)
        print('    elo_scale = {0}'.format(self.elo_scale))
        return self.wt_rating_adjustments, self.elo_scale
    
    def save_to_config(self):
        ## load full config ##
        config_path = '{0}/config.json'.format(self.package_dir)
        with open(config_path, 'r') as fp:
            full_config = json.load(fp)
        ## round adjustments for storage ##
        rounded_adjustments = {k: round(v, 6) for k, v in self.wt_rating_adjustments.items()}
        ## update wt_ratings section ##
        full_config['wt_ratings']['wt_rating_adjustments'] = rounded_adjustments
        full_config['wt_ratings']['elo_scale'] = self.elo_scale
        ## save ##
        with open(config_path, 'w') as fp:
            json.dump(full_config, fp, indent=2)
        print('Saved coefficients to {0}'.format(config_path))
    
    def run(self, start_season=2009, end_season=None, save=True):
        self.train_all(start_season, end_season)
        if save:
            self.save_to_config()
        return self.wt_rating_adjustments, self.elo_scale
