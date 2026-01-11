## shared constants for nfelosrs ##

## team name replacements for standardization ##
TEAM_REPLACEMENTS = {
    'LA': 'LAR',
    'LV': 'OAK',
    'STL': 'LAR',
    'SD': 'LAC',
    'WSH': 'WAS'
}

## elo and rating constants ##
ELO_CENTER = 1505
ELO_TO_POINTS_DIVISOR = 25

## NFL game constants ##
STANDARD_NFL_GAMES = 16
AVERAGE_WINS = 8

## SRS rating columns used in metrics calculations ##
SRS_RATING_COLUMNS = [
    'avg_mov',
    'srs_rating',
    'srs_rating_normalized',
    'bayesian_rating',
    'pre_season_wt_rating',
    'srs_rating_w_qb_adj',
    'srs_rating_normalized_w_qb_adj',
    'bayesian_rating_w_qb_adj',
    'pre_season_wt_rating_w_qb_adj'
]
