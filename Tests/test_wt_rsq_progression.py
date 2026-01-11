import pandas as pd
import pathlib
import sys

package_dir = pathlib.Path(__file__).parent.parent.resolve()

def test_rsq_progression():
    '''
    Ensures wt_rating has higher R² to SRS than line_adj for weeks 8-17.
    This validates that the mean reversion adjustment improves predictive power.
    '''
    wt_ratings = pd.read_csv('{0}/wt_ratings.csv'.format(package_dir), index_col=0)
    srs = pd.read_csv('{0}/srs_ratings.csv'.format(package_dir), index_col=0)
    predictors = ['line', 'line_adj', 'line_rating', 'wt_rating']
    all_passed = True
    for week in range(8, 18):
        srs_week = srs[srs['week'] == week][['season', 'team', 'srs_rating']]
        merged = pd.merge(
            wt_ratings[['season', 'team'] + predictors],
            srs_week,
            on=['season', 'team'],
            how='inner'
        )
        rsq_line_adj = merged['line_adj'].corr(merged['srs_rating']) ** 2
        rsq_wt = merged['wt_rating'].corr(merged['srs_rating']) ** 2
        passed = rsq_wt > rsq_line_adj
        status = 'PASS' if passed else 'FAIL'
        print('  Week {0}: {1} - line_adj={2:.4f}, wt_rating={3:.4f}'.format(
            week, status, rsq_line_adj, rsq_wt
        ))
        if not passed:
            all_passed = False
    return all_passed

if __name__ == '__main__':
    print('Testing WT Ratings R² Progression...')
    passed = test_rsq_progression()
    print('Result: {0}'.format('PASS' if passed else 'FAIL'))
    sys.exit(0 if passed else 1)
