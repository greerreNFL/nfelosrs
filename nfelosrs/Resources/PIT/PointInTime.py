import pandas as pd
import numpy

from . import QBPit, GamesPit

class PointInTime:
    '''
    Point In Time creates a snapshot of NFL games in a given season
    through a given week.

    It calculates QB Adjustments as they would exist at that point in time,
    uses real results that have occured through that week, and then calculated bayesian
    updated priors to use as filler for results after the week where the snapshot was taken.

    This game file represents all info available through the week in question, and
    no information beyond it, so we can then use these game results to calcualte
    point in time, prior informed SRS rankings for each week
    '''

    def __init__(self, qb_df, games, season, week):
        self.qb_pit = QBPit.QBPit(qb_df, games, season, week)
        self.games_pit = GamesPit.GamesPit(games, self.qb_pit.weekly_qb_adjustments)
        ## unpack some data for convenience in the SRS class
        self.games = self.games_pit.games
        self.current_bayesian_ratings = self.games_pit.current_rankings
        self.current_bayesian_stdevs = self.games_pit.current_stdevs
        self.current_qb_adjs = self.qb_pit.get_last_qb_adjs()
        self.wt_ratings = self.games_pit.wt_ratings
