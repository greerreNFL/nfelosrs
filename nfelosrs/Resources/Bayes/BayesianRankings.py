import pandas as pd
import numpy
import pathlib
import json
from scipy.stats import invgamma

class BayesianRankings:
    '''
    Creates a DF of rankings by week to use as priors for SRS
    This starts with the WT Rankings and then updates them based on
    each weekly results

    This is an adaptive bayesian model
    '''
    def __init__(self, games_w_qb_adj, season, week):
        self.games = games_w_qb_adj
        self.season = season
        self.week = week
        self.package_dir = pathlib.Path(__file__).parent.parent.parent.parent.resolve()
        self.distributions = self.load_distributions()
        ## structure for bayesian updated ##
        self.current = self.initialize_rankings()
        self.weekly = []
    
    def load_wt_rankings(self):
        '''
        Loads the wt rankigns and filters for the seasons passed
        '''
        df = pd.read_csv(
            '{0}/wt_ratings.csv'.format(self.package_dir),
            index_col=0
        )
        return df[
            df['season'] == self.season
        ].copy()

    def load_distributions(self):
        '''
        Loads the observed stanrdard deviations for rankigns and results
        '''
        with open(
            '{0}/nfelosrs/Resources/Bayes/distributions.json'.format(self.package_dir),
            'r'
        ) as fp:
            return json.load(fp)
        
    def initialize_rankings(self):
        '''
        Creates the initial structure for the bayesian inference
        This is a dictionary where each team's current mean and stdev is stored
        '''
        ## load wt_ratings ##
        wt_ratings = self.load_wt_rankings()
        ## add distribution ##
        wt_ratings['ranking_stdev'] = self.distributions['rankings']
        ## return as dictionary ##
        return wt_ratings[[
            'team', 'wt_rating', 'ranking_stdev'
        ]].rename(columns={
            'wt_rating' : 'ranking_mean'
        }).set_index('team').to_dict('index')

    def likelihood(self, row):
        '''
        Takes a game row from the games df and calculates updated priors
        based on result likelihood
        '''
        ## get priors ##
        home_priors = self.current[row['home_team']]
        away_priors = self.current[row['away_team']]
        ## create team, QB, and HFA adjusted results ##
        home_result = (
            ## actual score ##
            row['result'] +
            ## adjust for opponent
            (
                away_priors['ranking_mean'] + row['away_qb_adj']
            ) -
            ## adjust for teams own qb ##
            row['home_qb_adj'] -
            ## adjust for HFA ##
            row['modeled_hfa']
        )
        away_result = (
            ## actual score ##
            -1 * row['result'] +
            ## adjust for opponent
            (
                home_priors['ranking_mean'] + row['home_qb_adj']
            ) -
            ## adjust for teams own qb ##
            row['away_qb_adj'] -
            ## adjust for HFA ##
            -1 * row['modeled_hfa']
        )
        ## calcualte new mean and stdev from results ##
        updated_home_mean = (
            (
                home_priors['ranking_mean'] / home_priors['ranking_stdev']**2 +
                home_result / self.distributions['margins']**2
            ) / (
                1 / home_priors['ranking_stdev']**2 +
                1 / self.distributions['margins']**2
            )
        )
        updated_home_st_dev = numpy.sqrt(
            1 / (1 / home_priors['ranking_stdev']**2 +
            1 / self.distributions['margins']**2)
        )
        updated_away_mean = (
            (
                away_priors['ranking_mean'] / home_priors['ranking_stdev']**2 +
                away_result / self.distributions['margins']**2
            ) / (
                1 / away_priors['ranking_stdev']**2 +
                1 / self.distributions['margins']**2
            )
        )
        updated_away_st_dev = numpy.sqrt(
            1 / (1 / away_priors['ranking_stdev']**2 +
            1 / self.distributions['margins']**2)
        )
        ## return updated priors ##
        return updated_home_mean, updated_home_st_dev, updated_away_mean, updated_away_st_dev

    def update_priors(self):
        '''
        Runs through the games file and updated priors for each game and week
        '''
        for index, row in self.games[
            self.games['week'] <= self.week
        ].iterrows():
            ## create a structure for the output for the home and away teams ##
            home_rec = {
                'game_id' : row['game_id'],
                'season' : row['season'],
                'week' : row['week'],
                'opponent' : row['away_team'],
                'result' : row['result'],
                'bayesian_ranking_pre' : self.current[row['home_team']]['ranking_mean'],
                'bayesian_stdev_pre' : self.current[row['home_team']]['ranking_stdev'],
                'qb_adj' : row['home_qb_adj']
            }
            away_rec = {
                'game_id' : row['game_id'],
                'season' : row['season'],
                'week' : row['week'],
                'opponent' : row['home_team'],
                'result' : -1 * row['result'],
                'bayesian_ranking_pre' : self.current[row['away_team']]['ranking_mean'],
                'bayesian_stdev_pre' : self.current[row['away_team']]['ranking_stdev'],
                'qb_adj' : row['away_qb_adj']
            }
            ## update the model ##
            updated_home_mean, updated_home_st_dev, updated_away_mean, updated_away_st_dev = self.likelihood(
                row
            )
            ## update the records ##
            home_rec['bayesian_ranking_post'] = updated_home_mean
            home_rec['bayesian_stdev_post'] = updated_home_st_dev
            away_rec['bayesian_ranking_post'] = updated_away_mean
            away_rec['bayesian_stdev_post'] = updated_away_st_dev
            ## write records to weekly ##
            self.weekly.append(home_rec)
            self.weekly.append(away_rec)
            ## update current ##
            self.current[row['home_team']] = {
                'ranking_mean' : updated_home_mean,
                'ranking_stdev' : updated_home_st_dev
            }
            self.current[row['away_team']] = {
                'ranking_mean' : updated_away_mean,
                'ranking_stdev' : updated_away_st_dev
            }

    ## utility functions ##
    def return_updated_priors(self):
        '''
        Returns the current rankings without the standard deviation
        so it can be easily mapped
        '''
        just_means = {}
        for k,v in self.current.items():
            just_means[k] = v['ranking_mean']
        ## return ##
        return just_means

    def return_updated_deviations(self):
        '''
        Returns the current standard deviations without the rankings
        so it can be easily mapped
        '''
        just_stdevs = {}
        for k,v in self.current.items():
            just_stdevs[k] = v['ranking_stdev']
        ## return ##
        return just_stdevs
    
    def return_wt_ratings(self):
        '''
        Returns a dictionary of the wt rankings for the season passed
        '''
        wt_rankings = self.load_wt_rankings()
        mapping = {}
        for index, row in wt_rankings.iterrows():
            mapping[row['team']] = row['wt_rating']
        return mapping