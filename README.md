# nfelosrs

## About
nfelosrs is a collection of models that leverage a Simple Rating System (SRS) appraoch to infer NFL team strength in a handle of contexts:

### WT Ratings
WT Ratings combine win total futures sourced from sportsbooks with the NFL schedule to determine the market implied team strength of each team. Team strengths are represented as both a spread vs the average team and an Elo rating.

The general methodology is as follows:
* Each team's win total is first adjusted up or down depending on the over and under probabilities derived from over and under odds.
* Then, each team is assigned a value that is used with the schedule to determine their probabilitiy of winning each game.
* These individual game probabilities are added together to create an overall estimate of their season win total.
* An optimizer is used to find the set of team values that minimize the error between the market implied win totals and the model's win total estimates.

This process is repeated for both the spread and Elo versions of the ratings. The optimized rating for each team is then fed back into the schedule to determine strength of schedule, which is defined as the average rating of a team's opponents.

This approach to determining pre-season ratings and strength of schedule is more accurate than commonly used methods for ranking teams like previous season's win percentage, DVOA, or unadjusted win totals.
