## [Monaco](../../) - [Examples](../)

### Election

This example attempts to predict the result of the 2020 US presidential election, 
based on polling data from 3 weeks prior to the election. 

Each state independently casts a normally distributed % of votes for the Democratic,
Republican, and Other candidates. That percentage is then normalized so the total is
100%. We also assume a uniform ±3% national swing due to polling error which is 
applied to all states equally. The winner of each state's election assigns their 
electoral votes to that candidate, and the candidate that wins at least 270 of 
the 538 electoral votes is the winner.

Note that Maine and Nebraska congressional district electoral votes are modeled, 
but not mapped.

The caculated win probabilities from this sim are: 93.4% Dem, 6.2% Rep, 0.4% Tie

The data in state_presidential_odds.csv was adapted from FiveThirtyEight's 
[state topline polls-plus data](https://github.com/fivethirtyeight/data/tree/master/election-forecasts-2020)
as pulled on 15 October 2020. That data is licensed under the 
[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). 

<p float="left" align="center">
<img width="420" height="240" src="state_presidential_outcomes.png">
<img width="400" height="225" src="ev_histogram.png">  
</p>
