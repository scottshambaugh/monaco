from scipy import stats

def retirement_example_postprocess(mccase, df):
    
    # Note that for pandas dataframes, you must explicitly include the index
    mccase.addOutVal('Date', df.index)
    mccase.addOutVal('Returns', df['Returns'])
    mccase.addOutVal('Spending', df['Spending'])
    mccase.addOutVal('Starting Balance', df['Starting Balance'])
    mccase.addOutVal('Ending Balance', df['Ending Balance'])
    
    mccase.addOutVal('Average Returns', stats.gmean(1 + df['Returns']) - 1)   
    mccase.addOutVal('Final Balance', df['Ending Balance'][-1])
    wentbroke = 'No'
    if df['Ending Balance'][-1] == 0:
        wentbroke = 'Yes'
    mccase.addOutVal('Went Broke', wentbroke, valmap={'Yes':0, 'No':1})
