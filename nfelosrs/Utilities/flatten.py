## flatten utilities for home/away game data ##

import pandas as pd


def flatten_home_away(df, home_team_col, away_team_col, value_mappings):
    '''
    Generic flattening of home/away game data to team-level records.
    
    Parameters:
        df: DataFrame with game-level data
        home_team_col: column name for home team
        away_team_col: column name for away team
        value_mappings: dict mapping new_col_name -> (home_col, away_col)
            Example: {'result': ('result', 'away_result')}
            The team columns are handled automatically.
            
    Returns:
        DataFrame with team-level records (2 rows per game)
    '''
    ## build rename mappings ##
    home_rename = {home_team_col: 'team'}
    away_rename = {away_team_col: 'team'}
    
    for new_col, (home_col, away_col) in value_mappings.items():
        home_rename[home_col] = new_col
        away_rename[away_col] = new_col
    
    ## build column lists ##
    home_cols = [home_team_col] + [m[0] for m in value_mappings.values()]
    away_cols = [away_team_col] + [m[1] for m in value_mappings.values()]
    
    ## concatenate flattened data ##
    return pd.concat([
        df[home_cols].rename(columns=home_rename),
        df[away_cols].rename(columns=away_rename)
    ])
