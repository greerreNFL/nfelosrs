import pandas as pd
import numpy

class QBPit:
    '''
    Create a "Point in Time" snapshot of QBs for a given season and week
    A team's QB baseline is the best QB that has played that season through the PIT week
    All other QBs that play for the team are represented as a negative Adj for the week in
    which they play

    Since win totals are used as the prior, we make the assumption that the best QB is the one
    that is baked in to that number. This obviously fails for early weeks if that QB is not
    the Week 1 starter due to suspension or injury. These are accepted as edge cases
    '''
    
    def __init__(self, qb_df, games, season, week):
        self.qb_df = qb_df.copy()
        self.games = games
        self.season = season
        self.week = week
        ## format and transform ##
        self.add_game_id()
        self.filter_to_season_week()
        self.qb_df_flat = self.flatten_qbs()
        self.recent_rankings = self.get_most_recent_rankings()
        ## calculate ##
        self.weekly_qb_adjustments = self.calc_adjs()
    
    def add_game_id(self):
        '''
        Adds the nflfastr game ids to the qbelo file
        '''
        ## create a datetime series from the date ##
        self.qb_df['date_time'] = pd.to_datetime(self.qb_df['date'])
        ## mondays are new weeks, so subtract a day and then trunc to get an NFL week ##
        self.qb_df['date_time'] = self.qb_df['date_time'] - pd.Timedelta(days=1)
        self.qb_df['week_of'] = self.qb_df['date_time'].dt.isocalendar().week
        self.qb_df = self.qb_df.sort_values(
            by=['date_time'],
            ascending=[True]
        ).reset_index(drop=True)
        ## get unique weeks ##
        unique_week_df = self.qb_df.copy()[['season','week_of']].drop_duplicates()
        unique_week_df['week'] = unique_week_df.groupby(['season']).cumcount() + 1
        self.qb_df = pd.merge(
            self.qb_df,
            unique_week_df,
            on=['season','week_of'],
            how='left'
        )
        ## create nflfastR style game id ##
        self.qb_df = pd.merge(
            self.qb_df.rename(columns={
                'team1' : 'home_team',
                'team2' : 'away_team'
            }),
            self.games[['season', 'week', 'home_team', 'away_team', 'game_id']],
            on=['season', 'week', 'home_team', 'away_team'],
            how='left'
        )

    def filter_to_season_week(self):
        '''
        Filters the QB file down to the most recent week
        '''
        self.qb_df = self.qb_df[
            (self.qb_df['season'] == self.season) &
            (self.qb_df['week'] <= self.week)
        ].copy()

    def flatten_qbs(self):
        '''
        Flattens the qb df, whose records are games into individual team<>qb<>week records 
        '''
        return pd.concat([
            self.qb_df[[
                'game_id', 'season', 'week', 'home_team',
                'qb1', 'qb1_value_pre'
            ]].rename(columns={
                'home_team' : 'team',
                'qb1' : 'qb',
                'qb1_value_pre' : 'qb_value'
            }),
            self.qb_df[[
                'game_id', 'season', 'week', 'away_team',
                'qb2', 'qb2_value_pre'
            ]].rename(columns={
                'away_team' : 'team',
                'qb2' : 'qb',
                'qb2_value_pre' : 'qb_value'
            }),
        ]).sort_values(
            by=['team', 'season', 'week'],
            ascending=[True, True, True]
        ).reset_index(drop=True)

    def get_most_recent_rankings(self):
        '''
        Determines the last ranking available for all QBs that have started
        We are assuming that the most recent ranking available is the most accurate
        representation of the QBs performance for all weeks in the season

        The qbelo model has strong off-seaosn mean reversion and strong in season adjustments,
        which reflects the belief that QBs produce at a level above or below their true ability
        due to seasonal context (scheme, injury, etc) and that the direction of the variance (not noise!)
        is not knowable going in to the seaosn. Thus the QBs most recent ranking, which utilizes the most
        data possible, is the best representation of their true *production* even in earlier weeks
        '''
        ## get most recent ##
        recent_rankings = self.qb_df_flat.groupby(['team', 'qb']).tail(1).copy()
        ## translate value to a point value ##
        recent_rankings['point_value'] = recent_rankings['qb_value'] / 25
        ## calc the adj vs max for team ##
        recent_rankings['qb_adj'] = (
            recent_rankings['point_value'] -
            recent_rankings.groupby(['team'])['point_value'].transform(lambda x: x.max())
        )
        ## return ##
        return recent_rankings
    
    def calc_adjs(self):
        '''
        Calculates the weekly adjustments
        '''
        return pd.merge(
            self.qb_df_flat,
            self.recent_rankings[[
                'team', 'qb', 'qb_adj'
            ]],
            on=['team', 'qb'],
            how='left'
        )
    
    def get_last_qb_adjs(self):
        '''
        returns the most recent QB rating adjustment
        '''
        last_ratings = self.weekly_qb_adjustments.groupby(['team']).tail(1).copy()
        mapping = {}
        for index, row in last_ratings.iterrows():
            mapping[row['team']] = row['qb_adj']
        return mapping


