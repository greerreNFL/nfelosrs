from .Resources import *

def run(rebuild=False, with_date_return=False):
    ## wrapper to run and update all models ##
    ## load data ##
    data = DataLoader()
    ## update win totals ##
    wt_ratings = WTRatings(
        data.wts,
        data.games,
        data.wt_ratings,
        rebuild
    )
    wt_ratings.update()
    ## update srs
    srs_runner = SRSRunner(
        data.games, data.qbs,
        data.current_season, data.current_week,
        rebuild
    )
    srs_runner.run()
    if with_date_return:
        ## if flagged, will return the season and week
        ## the ratings are through. This is done to give the 
        ## downstream repo updating script contextual info for commit msg
        return data.current_season, data.current_week

def create_bayesian_distributions():
    '''
    wrapper for the bayesian distribution workflow
    '''
    update_distributions()
