from scipy import stats


def retirement_example_postprocess(case, df):
    # Note that for pandas dataframes, you must explicitly include the index
    case.addOutVal("Date", df.index)
    case.addOutVal("Returns", df["Returns"])
    case.addOutVal("Spending", df["Spending"])
    case.addOutVal("Starting Balance", df["Starting Balance"])
    case.addOutVal("Ending Balance", df["Ending Balance"])

    case.addOutVal("Average Returns", stats.gmean(1 + df["Returns"]) - 1)
    case.addOutVal("Final Balance", df["Ending Balance"][-1])
    wentbroke = "No"
    if df["Ending Balance"][-1] == 0:
        wentbroke = "Yes"
    case.addOutVal("Broke", wentbroke, valmap={"Yes": 0, "No": 1})
