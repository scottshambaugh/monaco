from scipy.stats import norm, uniform
import logging
import monaco as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib

from election_example_run import election_example_run
from election_example_preprocess import election_example_preprocess
from election_example_postprocess import election_example_postprocess
fcns = {'preprocess' : election_example_preprocess,
        'run'        : election_example_run,
        'postprocess': election_example_postprocess}

ndraws = 1000
seed = 12362397

def election_example_monte_carlo_sim():
    sim = mc.Sim(name='election', ndraws=ndraws, fcns=fcns,
                 firstcaseismedian=False, seed=seed, singlethreaded=True,
                 samplemethod='random', savecasedata=False,
                 verbose=True, debug=False)

    rootdir = pathlib.Path(__file__).parent.absolute()
    df = pd.read_csv(rootdir / 'state_presidential_odds.csv')
    states = df['State'].tolist()
    df['Dem_Sig'] = df['Dem_80_tol']/mc.pct2sig(0.8, bound='2-sided')
    df['Rep_Sig'] = df['Rep_80_tol']/mc.pct2sig(0.8, bound='2-sided')
    df['Other_Sig'] = df['Other_80_tol']/mc.pct2sig(0.8, bound='2-sided')

    sim.addInVar(name='National Dem Swing', dist=uniform, distkwargs={'loc': -0.03, 'scale': 0.06})

    for state in states:
        i = df.loc[df['State'] == state].index[0]
        sim.addInVar(name=f'{state} Dem Unscaled Pct', dist=norm,
                     distkwargs={'loc': df['Dem_Mean'][i],   'scale': df['Dem_Sig'][i]})
        sim.addInVar(name=f'{state} Rep Unscaled Pct', dist=norm,
                     distkwargs={'loc': df['Rep_Mean'][i],   'scale': df['Rep_Sig'][i]})
        sim.addInVar(name=f'{state} Other Unscaled Pct', dist=norm,
                     distkwargs={'loc': df['Other_Mean'][i], 'scale': df['Other_Sig'][i]})

    sim.addConstVal(name='states', val=states)
    sim.addConstVal(name='df', val=df)

    sim.runSim()

    sim.outvars['Dem EVs'].addVarStat(stat='orderstatP',
                                      statkwargs={'p': 0.5, 'bound': 'nearest'})
    sim.outvars['Dem EVs'].addVarStat(stat='orderstatTI',
                                      statkwargs={'p': 0.75, 'c': 0.90, 'bound': '2-sided'})
    fig, ax = mc.plot(sim.outvars['Dem EVs'], plotkwargs={'bins': 50})
    ax.set_autoscale_on(False)
    ax.plot([270, 270], [0, 1], '--k')
    fig.set_size_inches(8.0, 4.5)
    plt.savefig('ev_histogram.png', dpi=100)

    pct_dem_win = sum(x == 'Dem' for x in sim.outvars['Winner'].vals)/sim.ncases
    pct_rep_win = sum(x == 'Rep' for x in sim.outvars['Winner'].vals)/sim.ncases
    pct_contested = sum(x == 'Contested' for x in sim.outvars['Winner'].vals)/sim.ncases
    logger = logging.getLogger(__name__)
    logger.info(
        f'Win probabilities: {100*pct_dem_win:0.1f}% Dem, '
        f'{100*pct_rep_win:0.1f}% Rep, '
        f'{100*pct_contested:0.1f}% Contested')
    mc.plot(sim.outvars['Winner'])

    pct_recount = sum(x != 0 for x in sim.outvars['Num Recounts'].vals)/sim.ncases
    logger.info(
        f'In {100*pct_recount:0.1f}% of runs there was a state close enough '
        'to trigger a recount (<0.5%)')

    dem_win_state_pct = dict()
    for state in states:
        dem_win_state_pct[state] \
            = sum(x == 'Dem' for x in sim.outvars[f'{state} Winner'].vals)/sim.ncases

    # Only generate state map if plotly installed. Want to avoid this as a dependency
    gen_map = False
    import importlib
    if importlib.util.find_spec('plotly') and gen_map:
        import plotly.graph_objects as go
        from plotly.offline import plot
        plt.figure()
        fig = go.Figure(data=go.Choropleth(
                        locations=df['State_Code'],  # Spatial coordinates
                        z=np.fromiter(dem_win_state_pct.values(), dtype=float)*100,
                        locationmode='USA-states',
                        colorscale='RdBu',
                        colorbar_title="Dem % Win" ))

        fig.update_layout( geo_scope='usa')
        plot(fig)

    return sim


if __name__ == '__main__':
    sim = election_example_monte_carlo_sim()
