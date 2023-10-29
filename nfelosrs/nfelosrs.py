from .Resources import *

def run(rebuild=False):
    ## wrapper to run and update all models ##
    data = DataLoader()
    wt_ratings = WTRatings(
        data.wts,
        data.games,
        data.wt_ratings,
        rebuild
    )
    wt_ratings.update()
