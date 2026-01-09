import pandas as pd
import numpy


def american_to_prob(series):
    ## convert ameriacn odds to a probability ##
    return numpy.where(
        series < 100,
        (-1 * series) / (100 - series),
        100 / (100 + series)
    )

def calc_vf_over_prob(over_prob, under_prob):
    ## calculates a vig free probability ##
    return (
        over_prob /
        (over_prob + under_prob)
    )

def spread_to_prob(spread, divisor=400):
    ## hacky method for converting spreads to win probabilities ##
    ## using the elo formula ##
    ## convert to elo dif ##
    dif = spread * 25
    ## convert to prob ##
    prob = (
        1 /
        (
            10 ** (-dif / divisor) +
            1
        )
    )
    return prob

def calc_probs_and_hold(over, under):
    ## takes over and under as american and returns ##
    ## vig free prob and hold ##
    over_prob = american_to_prob(over)
    under_prob = american_to_prob(under)
    vf_over_prob = calc_vf_over_prob(over_prob, under_prob)
    vf_under_prob = 1 - vf_over_prob
    hold = (over_prob + under_prob) - 1
    return vf_over_prob, vf_under_prob, hold

def add_odds_and_line_adj(wts, over_prob_logit_coef):
    ## translates over and under odds probabilities and an adjusted line ##
    ## translate american odds to probabilities ##
    wts['over_prob'] = american_to_prob(wts['over_odds'])
    wts['under_prob'] = american_to_prob(wts['under_odds'])
    wts['hold'] = wts['over_prob'] + wts['under_prob'] - 1
    ## get vig free probabilities ##
    wts['over_prob_vf'] = calc_vf_over_prob(wts['over_prob'], wts['under_prob'])
    wts['under_prob_vf'] = 1 - wts['over_prob_vf']
    ## take the logit of the over prob for regression ##
    wts['logit_over_prob_vf'] = numpy.log(
        wts['over_prob_vf'] /
        (1 - wts['over_prob_vf'])
    )
    ## adjust total line by using regression ##
    wts['line_adj'] = (
        wts['line'] +
        wts['logit_over_prob_vf'] * over_prob_logit_coef
    )
    ## drop ##
    wts = wts.drop(columns=['over_prob', 'under_prob'])
    return wts
