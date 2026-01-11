import pandas as pd
import pathlib
import sys

package_dir = pathlib.Path(__file__).parent.parent.resolve()

EXPECTED_TEAMS = 32

def test_wt_ratings_completeness():
    '''
    Ensures we have wt_ratings for all teams in all seasons.
    Checks for:
        - 32 teams per season
        - No duplicate teams
        - No missing wt_rating or line_rating values
    '''
    wt_ratings = pd.read_csv('{0}/wt_ratings.csv'.format(package_dir), index_col=0)
    seasons = sorted(wt_ratings['season'].unique())
    all_passed = True
    for season in seasons:
        season_data = wt_ratings[wt_ratings['season'] == season]
        team_count = season_data['team'].nunique()
        has_duplicates = season_data.duplicated(subset=['team']).any()
        has_missing = season_data[['wt_rating', 'line_rating']].isna().any().any()
        passed = team_count == EXPECTED_TEAMS and not has_duplicates and not has_missing
        status = 'PASS' if passed else 'FAIL'
        print('  {0}: {1} teams={2}'.format(season, status, team_count))
        if not passed:
            all_passed = False
    return all_passed

if __name__ == '__main__':
    print('Testing WT Ratings Completeness...')
    passed = test_wt_ratings_completeness()
    print('Result: {0}'.format('PASS' if passed else 'FAIL'))
    sys.exit(0 if passed else 1)
