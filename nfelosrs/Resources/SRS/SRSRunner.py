import pandas as pd
import numpy

from ...Utilities import calc_rsq_by_week, calc_rmse_by_week, get_package_dir
from .SRS import SRS


class SRSRunner:
    '''
    A wrapper for SRSs. Takes an existing SRS file, and the current season state
    to determine which weeks need to be updated.
    '''
    def __init__(self, games, qbs, most_recent_season, most_recent_week, rebuild=False):
        ## load data ##
        self.package_dir = get_package_dir()
        self.games = games
        self.qbs = qbs
        ## state for tracking data freshness
        self.rebuild = rebuild
        self.most_recent_season = most_recent_season
        self.most_recent_week = most_recent_week
        self.week_list = self.games[
            ## unique weeks after the first win totals are available
            ## but on or before the most recent full week of play
            (self.games['season'] >= 2003) &
            ((
                (self.games['week'] <= self.most_recent_week) &
                (self.games['season'] == self.most_recent_season)
            ) | (
                self.games['season'] < self.most_recent_season
            ))
        ].sort_values(
            by=['season', 'week'],
            ascending=[True, True]
        ).reset_index(drop=True).groupby(['season', 'week']).head(1)[[
            'season', 'week'
        ]].values.tolist()
        self.existing_ratings, self.current_week_index = self.load_existing()

    def load_existing(self):
        '''
        Loads existing srs ratings and sets state
        '''
        try:
            existing = pd.read_csv(
                '{0}/srs_ratings.csv'.format(self.package_dir),
                index_col=0
            )
            ## most recent ##
            mr_season = existing['season'].max()
            mr_week = existing[existing['season']==mr_season]['week'].max()
            ## get index in week list ##
            mr_index = self.week_list.index([mr_season, mr_week])
            ## return ##
            return existing, mr_index
        except FileNotFoundError:
            ## if no file exists, return none and start the index at 0
            return None, 0
    
    def run(self):
        '''
        Determines what needs to be run and adds new data to storage
        '''
        if self.rebuild:
            ## if we are rebuilding, force the index back to 0 ##
            self.current_week_index = 0
            self.existing_ratings = None
        if self.current_week_index < len(self.week_list) -1:
            print('SRS Ratings are not up to date. Updating...')
            ## Only if the current week index is less than the index of the last week
            ## do we have fresh weeks to pull
            ## storage for each run ##
            new_dfs = []
            ## run for each ##
            for season_week_array in self.week_list[self.current_week_index+1:]:
                print('     On week {0}, {1}'.format(
                    season_week_array[1],
                    season_week_array[0]
                ))
                srs_ = SRS(
                    self.games,
                    self.qbs,
                    season_week_array[0],
                    season_week_array[1]
                )
                new_dfs.append(pd.DataFrame(srs_.records))
            ## combine and write to local as necessary ##
            ## get a df of new data
            new_df = pd.concat(new_dfs)
            if self.existing_ratings is not None:
                ## if existing data exists, combine
                new_df = pd.concat([self.existing_ratings, new_df])
            ## sort accordingly ##
            new_df = new_df.sort_values(
                by=['season', 'team', 'week'],
                ascending=[True, True, True]
            ).reset_index(drop=True)
            ## save ##
            new_df.to_csv(
                '{0}/srs_ratings.csv'.format(self.package_dir)
            )
            ## calc rsq ##
            rsq = calc_rsq_by_week(new_df)
            ## save
            rsq.to_csv(
                '{0}/srs_rating_rsqs.csv'.format(self.package_dir)
            )
            ## calc rsme ##
            rmse = calc_rmse_by_week(self.games, new_df)
            rmse.to_csv(
                '{0}/srs_rating_rmse.csv'.format(self.package_dir)
            )
