## line rating calculation utility ##

from .constants import STANDARD_NFL_GAMES, AVERAGE_WINS

def add_line_rating(df):
    '''
    Adds line_rating column to a dataframe containing line_adj values.
    Line rating is effectively projected wins over 8, normalized for 
    16 vs 17 game seasons, then centered at 0.
    
    Parameters:
        df: DataFrame with 'season' and 'line_adj' columns
        
    Returns:
        DataFrame with 'line_rating' column added
    '''
    STANDARD_TOTAL_WINS = STANDARD_NFL_GAMES * 16
    df['line_rating'] = (
        df['line_adj'] *
        (STANDARD_TOTAL_WINS / df.groupby('season')['line_adj'].transform('sum'))
    ) - AVERAGE_WINS
    return df
