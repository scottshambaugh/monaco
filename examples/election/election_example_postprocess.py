def election_example_postprocess(case, dem_evs, rep_evs, other_evs, state_winner, state_recount):
    # Note that for pandas dataframes, you must explicitly include the index
    case.addOutVal("Dem EVs", dem_evs)
    case.addOutVal("Rep EVs", rep_evs)
    case.addOutVal("Other EVs", other_evs)

    num_recounts = 0
    for state in case.constvals["states"]:
        case.addOutVal(f"{state} Winner", state_winner[state])
        case.addOutVal(f"{state} Recount", state_recount[state])
        if state_recount[state]:
            num_recounts += 1

    case.addOutVal("Num Recounts", num_recounts)

    winner = "Contested"
    if dem_evs >= 270:
        winner = "Dem"
    elif rep_evs >= 270:
        winner = "Rep"
    elif other_evs >= 270:
        winner = "Other"

    case.addOutVal("Winner", winner)
