def election_example_preprocess(mccase):
    states = mccase.constvals['states']
    df = mccase.constvals['df']

    state_evs = dict(zip(df['State'], df['EV']))
    state_dem_pct = dict()
    state_rep_pct = dict()
    state_other_pct = dict()

    for state in states:
        total_pct = mccase.mcinvals[f'{state} Dem Unscaled Pct'].val + \
                    mccase.mcinvals[f'{state} Rep Unscaled Pct'].val + \
                    mccase.mcinvals[f'{state} Other Unscaled Pct'].val
        
        # Scale the percentages so the total is 100%
        state_dem_pct[state] = mccase.mcinvals[f'{state} Dem Unscaled Pct'].val / total_pct \
                               + mccase.mcinvals['National Dem Swing'].val
        state_rep_pct[state] = mccase.mcinvals[f'{state} Rep Unscaled Pct'].val / total_pct \
                               - mccase.mcinvals['National Dem Swing'].val
        state_other_pct[state] = mccase.mcinvals[f'{state} Other Unscaled Pct'].val / total_pct

    return (states, state_evs, state_dem_pct, state_rep_pct, state_other_pct)
