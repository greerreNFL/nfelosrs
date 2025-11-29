import pandas as pd
import numpy
import pathlib
from scipy.optimize import minimize

from .. import Utilities as utils

class WTRatings():
    ## model class for the wt model ##
    def __init__(self, wts, games, wt_ratings, rebuild=False, config_override=None):
        ## package path ##
        self.package_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
        ## state ##
        self.wts_season = None ## last season with win total lines ##
        self.wt_ratings_season = None ## last season with win total ratings ##
        ## config ##
        self.rebuild = rebuild
        self.config = utils.load_config('config.json', ['wt_ratings'])
        if config_override:
            self.config.update(config_override)
        ## dfs ##
        self.wts = wts
        self.games = games
        self.wt_ratings = wt_ratings
        ## new data ##
        self.new_wt_ratings_data = []
        ## init ##
        self.set_states()
        self.format_wts()
        
    ## SETUP FUNCS ##
    def set_states(self):
        ## set states to determine how many season to build win total ratings for ##
        ## Win Total Ratings ##
        if self.rebuild or self.wt_ratings is None:
            ## if we are doing a full rebuilt, set the init season to the first with win totals ##
            self.wt_ratings_season = self.wts['season'].min() - 1
        else:
            ## otherwise, set the init season to the last season with ratings ##
            self.wt_ratings_season = self.wt_ratings['season'].max()
        ## Win Total Lines ##
        self.wts_season = self.wts['season'].max()
        
    
    def format_wts(self):
        ## format win total lines ##
        ## this vig free over and under odds, and a probability adjusted win expectency called line_adj
        self.wts = utils.add_odds_and_line_adj(self.wts, self.config['over_prob_logit_coef'])
    
    ## OPTI HELPERS ##
    def apply_wins(self, x, var_keys, df, is_elo=False):
        ## estimates margin based on team ratings, then calculates
        ## win probability from margin, which is summed to create season
        ## win total estimation ##
        temp = df.copy()
        ## first construct dictionary of values to do vectorized apply ##
        val_dict = {}
        for index, value in enumerate(x):
            val_dict[
                var_keys[index]
            ] = value
        ## then apply to frame with a replace ##
        temp['hfa'] = x[0]
        temp['home_rating'] = temp['home_team'].map(val_dict)
        temp['away_rating'] = temp['away_team'].map(val_dict)
        ## get rating delta ##
        rating_delta = (
            temp['home_rating'] +
            temp['hfa'] -
            temp['away_rating']
        )
        ## calc margin ##
        if is_elo:
            temp['expected_elo_dif'] = rating_delta
            ## calc win probs from margin ##
            temp['home_win'] = 1.0 / (numpy.power(10.0, (-1 * temp['expected_elo_dif']/400)) + 1.0)
            temp['away_win'] = 1 - temp['home_win']
        else:
            temp['expected_margin'] = rating_delta
            ## calc win probs from margin ##
            elo_divisor = self.config.get('elo_divisor', 400)
            temp['home_win'] = utils.spread_to_prob(temp['expected_margin'], divisor=elo_divisor)
            temp['away_win'] = 1 - temp['home_win']
        ## calc binary outcome ##
        home_win_binary = numpy.where(
            temp['home_win'] > 0.5,
            numpy.where(
                temp['home_win'] == 0.5,
                0.5,
                1
            ),
            0
        )
        away_win_binary = 1 - home_win_binary
        ## blend into margin ##
        temp['home_win'] = (
            home_win_binary * self.config['binary_weight'] +
            temp['home_win'] * (1 - self.config['binary_weight'])
        )
        temp['away_win'] = (
            away_win_binary * self.config['binary_weight'] +
            temp['away_win'] * (1 - self.config['binary_weight'])
        )
        ## retrun ##
        return temp
        
    
    def score_opti(self, applied_df, lines):
        ## score optimization with rmse ##
        ## flatten ##
        temp = pd.concat([
            ## home ##
            applied_df[[
                'home_team', 'home_win'
            ]].rename(columns={
                'home_team' : 'team',
                'home_win' : 'win'
            }),
            ## away ##
            applied_df[[
                'away_team', 'away_win'
            ]].rename(columns={
                'away_team' : 'team',
                'away_win' : 'win'
            })
        ])
        ## aggregate ##
        agg = temp.groupby(['team']).agg(
            wins = ('win', 'sum'),
        ).reset_index()
        ## add scores ##
        agg = pd.merge(
            agg,
            lines,
            on=['team'],
            how='left'
        )
        ## calc error ##
        agg['se'] = (
            agg['wins'] -
            agg['line_adj']
        ) ** 2
        rmse = agg['se'].mean() ** (1/2)
        ## return ##
        return rmse
    
    
    def obj_func(self, x, lines, games, var_keys, is_elo=False):
        ## actual function that optimizes ##
        ## apply wins ##
        applied = self.apply_wins(x, var_keys, games, is_elo=is_elo)
        ## score ##
        rmse = self.score_opti(
            applied, lines
        )
        ## return rmse as obj func ##
        return rmse
    
    
    def create_opti_params(self, season_wts, is_elo=False):
        ## helper to create params necessary for optimization ##
        ## elo and spread have slightly different setups ##
        if is_elo:
            ## best guesses, variable keys, bounds ##
            best_guesses = [37.5] ## init with hfa of 1.5 ##
            var_keys = ['HFA'] ## HFA Key ##
            bounds_list = [(0, 70)] ## hfa bounds ##
            ## add teams ##
            season_wts = season_wts.sort_values(
                by=['team'],
                ascending=[True]
            ).reset_index(drop=True)
            for index, row in season_wts.iterrows():
                best_guesses.append((row['line_adj'] - 8) / 7 * 200 + 1505)
                var_keys.append(row['team'])
                bounds_list.append((1100,1800)) 
        else:
            ## best guesses, variable keys, bounds ##
            best_guesses = [1.5] ## init with hfa of 1.5 ##
            var_keys = ['HFA'] ## HFA Key ##
            bounds_list = [(-5, 5)] ## hfa bounds ##
            ## add teams ##
            season_wts = season_wts.sort_values(
                by=['team'],
                ascending=[True]
            ).reset_index(drop=True)
            for index, row in season_wts.iterrows():
                best_guesses.append(row['line_adj'] - 8)
                var_keys.append(row['team'])
                bounds_list.append((-15,15))
        ## return ##
        return best_guesses, var_keys, tuple(bounds_list)
    
    
    ## OPTI RUNNER ##
    def optimize_season(self, season, is_elo=False):
        ## wrapper that calculates spread based win total ratings for a season ##
        ## get just the current seasons win total lines ##
        season_wts = self.wts[self.wts['season'] == season].copy()
        season_games = self.games[
            (self.games['season'] == season) &
            (self.games['game_type'] == 'REG')
        ].copy()
        ## get params ##
        best_guesses, var_keys, bounds = self.create_opti_params(season_wts, is_elo)
        ## run opti ##
        opti = minimize(
            self.obj_func,
            best_guesses,
            args=(
                season_wts,
                self.games[self.games['season'] == season].copy(),
                var_keys,
                is_elo
            ),
            bounds=bounds,
            method='SLSQP'
        )
        ## return ##
        return opti, var_keys

    ## SOS ##
    def calc_sos(self, x, var_keys, season):
        ## uses wt_ratings and sched to calculte sos ##
        ## apply ratings to schedule ##
        temp = self.apply_wins(
            x,
            var_keys,
            self.games[self.games['season'] == season].copy()
        )
        ## flatten ##
        flat = pd.concat([
            temp[[
                'home_team', 'away_rating'
            ]].rename(columns={
                'home_team' : 'team',
                'away_rating' : 'opponent_rating'
            }),
            temp[[
                'away_team', 'home_rating'
            ]].rename(columns={
                'away_team' : 'team',
                'home_rating' : 'opponent_rating'
            })
        ])
        ## group by team and calc sos ##
        sos_df = flat.groupby(['team']).agg(
            sos = ('opponent_rating', 'mean'),
        ).reset_index()
        ## return ##
        return sos_df
        
    
    ## MAIN ##
    def run_season(self, season):
        ## wrapper that optimizes season and outputs to new data ##
        ## run optis ##
        opti, var_keys = self.optimize_season(season)
        opti_elo, var_keys_elo = self.optimize_season(season, is_elo=True)
        ## SOS ##
        sos = self.calc_sos(opti.x, var_keys, season)
        ## write to new data ##
        for team in var_keys:
            if team == 'HFA':
                continue
            self.new_wt_ratings_data.append({
                'team' : team,
                'season' : season,
                'wt_rating' : opti.x[var_keys.index(team)],
                'wt_rating_elo' : opti_elo.x[var_keys_elo.index(team)],
                'sos' : sos[sos['team'] == team].iloc[0]['sos']
            })

    def update(self):
        print('Updating Win Total Ratings...')
        if self.rebuild:
            print('     Rebuild set to True. Will rebuild from first available win total season...')
        if self.wts_season == self.wt_ratings_season:
            print('     Win Totals and Win Total Ratings are both current through {0}'.format(self.wts_season))
        ## optimize all seasons as needed and add to existing ##
        ## run optis ##
        while self.wts_season > self.wt_ratings_season:
            print('     Running Season {0}...'.format(self.wt_ratings_season + 1))
            self.run_season(self.wt_ratings_season + 1)
            self.wt_ratings_season += 1
        ## determine if new data has been added ##
        if len(self.new_wt_ratings_data) > 0:
            ## if new data avail, create DF, and add select data from games ##
            new_df = pd.DataFrame(self.new_wt_ratings_data)
            new_df = pd.merge(
                new_df,
                self.wts.rename(columns={
                    'over_prob_vf' : 'over_probability',
                    'under_prob_vf' : 'under_probability'
                }).drop(columns=['logit_over_prob_vf']),
                on=['season','team'],
                how='left'
            )
            ## round ratings ##
            for col in [
                'wt_rating', 'wt_rating_elo', 'sos',
                'hold', 'over_probability', 'under_probability',
                'line_adj'
            ]:
                new_df[col] = new_df[col].round(4)
            ## add to existing ##
            if self.wt_ratings is None or self.rebuild:
                self.wt_ratings = new_df
                self.wt_ratings.to_csv(
                    '{0}/wt_ratings.csv'.format(self.package_dir)
                )
            else:
                self.wt_ratings = pd.concat([
                    self.wt_ratings,
                    new_df
                ])
                self.wt_ratings = self.wt_ratings.reset_index(drop=True)
                self.wt_ratings.to_csv(
                    '{0}/wt_ratings.csv'.format(self.package_dir)
                )
