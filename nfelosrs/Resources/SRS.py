import pandas as pd
import numpy
import pathlib

from .PIT import PointInTime

from .Bayes import update_distributions

from .DataLoader import DataLoader

def update_bayesian_distributions():
    db = DataLoader()
    update_distributions(db.games, db.db['qbelo'])
    

class SRS:
    '''
    A class for managing the calculation and return of a point in time
    SRS
    '''

    def __init__(self, games, qbs, season, week):
        ## data and meta ##
        self.season = season
        self.week = week
        self.PointInTime = PointInTime(qbs.copy(), games.copy(), season, week)
        self.games = self.PointInTime.games
        self.avg_margins = self.calc_margins()
        ## set up some structure for the SRS ##
        self.teams = self.games[['home_team', 'away_team']].stack().unique().tolist()
        self.team_to_index = {team : i for i, team in enumerate(self.teams)}
        self.coefficients = numpy.zeros((len(self.teams), len(self.teams)))
        self.constants = numpy.zeros(len(self.teams))
        self.records = []
        ## actions ##
        self.games_adjustments()
        self.populate_srs()
        self.solve_srs()
    
    def games_adjustments(self):
        '''
        Make adjustments to the games file -- filter out post season, and
        create results adjusted for HFA and QBs
        '''
        ## regular season only ##
        self.games = self.games[
            self.games['game_type'] == 'REG'
        ].copy()
        ## adjusted result ##
        self.games['adjusted_result'] = (
            ## start with the home margin, which here uses
            ## real results and prior based results
            self.games['results_with_rankings'] -
            ## subtract homefield advantage ##
            self.games['modeled_hfa'] -
            ## subtract the home based QB adj, but add the
            ## the away adjustment since this result is
            ## with respect to the home team
            self.games['home_qb_adj'] +
            self.games['away_qb_adj']
        )
    
    def calc_margins(self):
        '''
        Flatten the games df and calculate a teams average MoV and opponent
        avg MoV
        '''
        games_ = self.games[
            self.games['week']<=self.week
        ].copy()
        games_['away_result'] = games_['result'] * -1
        ## create a flat file of results by team ##
        flat = pd.concat([
            games_[['home_team', 'away_team','result']].rename(columns={
                'home_team' : 'team',
                'away_team' : 'opponent',
                'result' : 'mov'
            }),
            games_[['away_team', 'home_team','away_result']].rename(columns={
                'away_team' : 'team',
                'home_team' : 'opponent',
                'away_result' : 'mov'
            })
        ])
        ## calc an average margin ##
        avg_mov = flat.groupby(['team']).agg(
            avg_mov = ('mov', 'mean')
        ).reset_index()
        ## calc opp margins, filtered for other teams only ##
        avg_mov_against_records = []
        for index, row in flat.iterrows():
            ## filter flat for opp games where their opp
            ## was not the team ##
            flat_ = flat[
                (flat['team'] == row['opponent']) &
                (flat['opponent'] != row['team'])
            ].copy()
            if len(flat_) >= 0:
                avg_mov_against_records.append({
                    'team' : row['team'],
                    'opp_avg_mov' : flat_['mov'].mean()
                })
        ## merege
        avg_mov_against = pd.DataFrame(avg_mov_against_records)
        avg_mov = pd.merge(
            avg_mov,
            avg_mov_against.groupby(['team']).agg(
                avg_mov_of_opponents = ('opp_avg_mov', 'mean')
            ).reset_index(),
            on=['team'],
            how='left'
        )
        return avg_mov.set_index('team').to_dict('index')

    def populate_srs(self):
        '''
        Populated the coefficient matrix and constants vector based on
        games

        This is done with fully vectorized operations for speed. Refer to
        comments for whats going on
        '''
        ## add indicies of the teams within the SRS structures as columns
        ## in the games file
        home_teams = self.games['home_team'].map(self.team_to_index).values
        away_teams = self.games['away_team'].map(self.team_to_index).values
        adjusted_results = self.games['adjusted_result'].values
        ## populate the coefficients matrix using add.at ##
        numpy.add.at(self.coefficients, (home_teams, home_teams), 1)
        numpy.add.at(self.coefficients, (away_teams, away_teams), 1)
        numpy.add.at(self.coefficients, (home_teams, away_teams), -1)
        numpy.add.at(self.coefficients, (away_teams, home_teams), -1)
        ## populate the constants
        numpy.add.at(self.constants, home_teams, adjusted_results)
        numpy.add.at(self.constants, away_teams, -1 * adjusted_results)
        ## normailze the constants based on games played (ie avg margin) which 
        ## are the units we want this expressed in ##
        ## Initialize a game counts array to count the number of games each team plays ##
        game_counts = numpy.zeros(len(self.teams))
        numpy.add.at(game_counts, home_teams, 1)
        numpy.add.at(game_counts, away_teams, 1)
        ## normalize ##
        ## constants ##
        self.constants = self.constants / game_counts
        ## coefs ##
        for i in range(len(self.teams)):
            self.coefficients[i, :] /= game_counts[i]
    
    def solve_srs(self):
        '''
        Solves the populated coefficient matrix and constants vector
        '''
        ## solve the system ##
        try:
            srs_ratings = numpy.linalg.solve(
                self.coefficients,
                self.constants
            )
        except Exception as e:
            print('Linalg could not be solved. Will use least squares approx')
            srs_ratings = numpy.linalg.lstsq(
                self.coefficients,
                self.constants,
                rcond=None
            )[0]
        ## normalize around 0
        median_srs = numpy.median(srs_ratings)
        srs_ratings -= median_srs
        ## normalize to be on same scale as the bayesian ##
        max_bayes = 0
        min_bayes = 0
        for team in self.teams:
            val = self.PointInTime.current_bayesian_ratings[team]
            max_bayes = val if val > max_bayes else max_bayes
            min_bayes = val if val < min_bayes else min_bayes
        scaler = (
            (max_bayes - min_bayes) / 
            (numpy.max(srs_ratings) - numpy.min(srs_ratings))
        )
        srs_ratings_norm = srs_ratings * scaler
        ## populate records ##
        for team, rating, rating_norm in zip(self.teams, srs_ratings, srs_ratings_norm):
            self.records.append({
                'season' : self.season,
                'week' : self.week,
                'team' : team,
                'avg_mov' : round(self.avg_margins[team]['avg_mov'] if team in self.avg_margins else numpy.nan, 2),
                'avg_mov_of_opponents' : round(self.avg_margins[team]['avg_mov_of_opponents'] if team in self.avg_margins else numpy.nan, 2),
                ## ratings ##
                'srs_rating' : round(rating,2),
                'srs_rating_normalized' : round(rating_norm,2),
                'bayesian_rating' : round(self.PointInTime.current_bayesian_ratings[team],2),
                'bayesian_stdev' : round(self.PointInTime.current_bayesian_stdevs[team],2),
                'pre_season_wt_rating' : round(self.PointInTime.wt_ratings[team],2),
                ## qb adjusted ratings ##
                'qb_adjustment' : round(self.PointInTime.current_qb_adjs.get(team, 0),2),
                'srs_rating_w_qb_adj' : round(rating + self.PointInTime.current_qb_adjs.get(team, 0),2),
                'srs_rating_normalized_w_qb_adj' : round(rating_norm + self.PointInTime.current_qb_adjs.get(team, 0),2),
                'bayesian_rating_w_qb_adj' : round(self.PointInTime.current_bayesian_ratings[team] + self.PointInTime.current_qb_adjs.get(team, 0),2),
                'pre_season_wt_rating_w_qb_adj' : round(self.PointInTime.wt_ratings[team] + self.PointInTime.current_qb_adjs.get(team, 0),2),
            })

class SRSRunner:
    '''
    A wrapper for SRSs. Takes an existing SRS file, and the current season state
    to determine which weeks need to be updated.
    '''
    def __init__(self, games, qbs, most_recent_season, most_recent_week, rebuild=False):
        ## load data ##
        self.package_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
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
            mr_index = self.week_list.index([mr_season, mr_week]) + 1
            ## return ##
            return existing, mr_index
        except:
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
            for season_week_array in self.week_list[self.current_week_index:]:
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


                               