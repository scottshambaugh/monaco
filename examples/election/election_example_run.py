import numpy as np
import pandas as pd

def election_example_run(states, state_evs, state_dem_pct, state_rep_pct, state_other_pct):

    dem_evs = 0
    rep_evs = 0
    other_evs = 0
    state_winners = dict()
    state_recounts = dict()
    
    for state in states:
        max_vote_share = max(state_dem_pct[state], state_rep_pct[state], state_other_pct[state])
        if state_dem_pct[state] == max_vote_share: # Note an exact tie is statistically impossible
            dem_evs += state_evs[state]
            state_winners[state] = 'Dem'
        elif state_rep_pct[state] == max_vote_share:
            rep_evs += state_evs[state]
            state_winners[state] = 'Rep'
        else:
            other_evs += state_evs[state]
            state_winners[state] = 'Other'

        state_recounts[state] = False
        if abs(state_dem_pct[state] - state_rep_pct[state]) <= 0.005:
            state_recounts[state] = True
            
    return (dem_evs, rep_evs, other_evs, state_winners, state_recounts)


'''
### Test ###
if __name__ == '__main__':
    from monaco.order_statistics import pct2sig
    data = pd.read_csv('state_presidential_odds.csv')
    states = data['State'].tolist()
    state_evs = dict(zip(data['State'], data['EV']))
    
    dem_sig = data['Dem_80_tol']/pct2sig(0.8, bound='2-sided')
    rep_sig = data['Rep_80_tol']/pct2sig(0.8, bound='2-sided')
    other_sig = data['Other_80_tol']/pct2sig(0.8, bound='2-sided')
    
    state_dem_pct = dict(zip(data['State'], data['Dem_Mean']))
    state_rep_pct = dict(zip(data['State'], data['Rep_Mean']))
    state_other_pct = dict(zip(data['State'], data['Other_Mean']))
    
    (dem_evs, rep_evs, other_evs, state_winners, state_recounts) = election_example_run(states, state_evs, state_dem_pct, state_rep_pct, state_other_pct)
#'''
