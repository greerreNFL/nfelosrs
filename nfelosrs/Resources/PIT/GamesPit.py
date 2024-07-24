import pandas as pd
import numpy

from ..Bayes import BayesianRankings

class GamesPit:
    '''
    Create a "Point in Time" snapshot of games for a given season and week
    For the matrix to create a connected graph, it is helpful to have results for all
    scheduled games. Additionally, SRS models suffer from vol in early weeks when
    not enough games have been played. 
    
    To combat both, the default behavior for GamesPit is to fill games not yet played
    at the time of the snapshot with the spread between the home and away team's pre-seaosn
    WT Ranking.

    This, in effect, creates priors that are slowly removed from the model as
    the season progresses. However, these priors also become stale as the season progresses,
    making them a noiser plug value for future games. 

    To combat this, the wt_rankings are updated with a bayesian approach each week so that they
    stay as relevant as possible.

    '''
    
    def __init__(self, games, qb_adjs):
        self.games = games.copy()
        self.qb_adjs = qb_adjs
        ## infer season and week from where qb_adjs cuts off ##
        self.season = qb_adjs['season'].max()
        self.week = qb_adjs['week'].max()
        ## filter games to passed season ##
        self.games = self.games[self.games['season']==self.season].copy()
        ## add qbs to games ##
        self.add_qb_adjs()
        ## create bayesian rankings ##
        self.current_rankings, self.current_stdevs, self.wt_ratings = self.get_bayesian_rankings()
        ## add them as results to games ##
        self.construct_synthetic_results()

    def add_qb_adjs(self):
        '''
        Adds qb adjustments to the games file
        '''
        self.games = pd.merge(
            self.games,
            self.qb_adjs[[
                'game_id', 'team', 'qb_adj'
            ]].rename(columns={
                'team' : 'home_team',
                'qb_adj' : 'home_qb_adj'
            }),
            on=['game_id', 'home_team'],
            how='left'
        )
        self.games = pd.merge(
            self.games,
            self.qb_adjs[[
                'game_id', 'team', 'qb_adj'
            ]].rename(columns={
                'team' : 'away_team',
                'qb_adj' : 'away_qb_adj'
            }),
            on=['game_id', 'away_team'],
            how='left'
        )
        ## fill na for future weeks ##
        self.games['home_qb_adj'] = self.games['home_qb_adj'].fillna(0)
        self.games['away_qb_adj'] = self.games['away_qb_adj'].fillna(0)
    
    def get_bayesian_rankings(self):
        '''
        Calculates bayesian rankings and returns the most current through
        the week
        '''
        ## init a rankigns obj ##
        br = BayesianRankings(self.games, self.season, self.week)
        br.update_priors()
        return br.return_updated_priors(), br.return_updated_deviations(), br.return_wt_ratings()
    
    def construct_synthetic_results(self):
        '''
        Constructs synthetic results using real results from any
        game already played through the week passed, and the spread
        between the most recent rankings for any game after
        '''
        ## add last rankings for each team ##
        self.games['home_team_current_prior'] = self.games['home_team'].map(self.current_rankings)
        self.games['away_team_current_prior'] = self.games['away_team'].map(self.current_rankings)
        self.games['home_team_current_prior_stdev'] = self.games['home_team'].map(self.current_stdevs)
        self.games['away_team_current_prior_stdev'] = self.games['away_team'].map(self.current_stdevs)
        ## generate prior result ##
        ## note, we do not use qb adjustments here as ##
        ## that adjustment (which would be 0 for a prior based result) ##
        ## is factored in the SRS step ##
        self.games['prior_based_result'] = (
            self.games['home_team_current_prior'] +
            self.games['modeled_hfa'] -
            self.games['away_team_current_prior']
        )
        ## conditionally select the result to use based on week passed ##
        self.games['results_with_rankings'] = numpy.where(
            self.games['week'] > self.week,
            self.games['prior_based_result'],
            self.games['result']
        )
