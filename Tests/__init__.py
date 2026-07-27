from .test_wt_completeness import test_wt_ratings_completeness
from .test_wt_rsq_progression import test_rsq_progression


def run_tests():
    '''
    Run package WT rating validation tests.
    Returns True if all tests pass.
    '''
    print('Testing WT Ratings Completeness...')
    completeness_passed = test_wt_ratings_completeness()
    print('Result: {0}'.format('PASS' if completeness_passed else 'FAIL'))
    print('Testing WT Ratings R² Progression...')
    rsq_passed = test_rsq_progression()
    print('Result: {0}'.format('PASS' if rsq_passed else 'FAIL'))
    return completeness_passed and rsq_passed
