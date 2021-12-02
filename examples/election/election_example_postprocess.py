def election_example_postprocess(mccase, dem_evs, rep_evs, other_evs, state_winner, state_recount):

    # Note that for pandas dataframes, you must explicitly include the index
    mccase.addOutVal('Dem EVs', dem_evs)
    mccase.addOutVal('Rep EVs', rep_evs)
    mccase.addOutVal('Other EVs', other_evs)

    num_recounts = 0
    for state in mccase.constvals['states']:
        mccase.addOutVal(f'{state} Winner', state_winner[state])
        mccase.addOutVal(f'{state} Recount', state_recount[state])
        if state_recount[state]:
            num_recounts += 1

    mccase.addOutVal('Num Recounts', num_recounts)

    winner = 'Contested'
    if dem_evs >= 270:
        winner = 'Dem'
    elif rep_evs >= 270:
        winner = 'Rep'
    elif other_evs >= 270:
        winner = 'Other'

    mccase.addOutVal('Winner', winner)
