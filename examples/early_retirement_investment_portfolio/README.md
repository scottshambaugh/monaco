## [monaco](../../) - [Examples](../)

### Early Retirement Investment Portfolio
This example simulates drawing down a fixed amount of money each year from an 
invested lump sum, as an example of monte-carlos applied to financial modeling.

This simulation assumes real rather than nominal values, and S&P500 returns 
independently drawn each year from a normal distribution modeled on historical 
returns. This is of course an incredibly simplified model, but it does show 
some interesting results such as the correlation between investment returns and 
going broke to be much higher in earlier years (below). In other words, we see 
the effect of "sequence of returns risk".

<p float="left" align="center">
<img width="360" height="270" src="./yearly_balances.png">  
<img width="360" height="270" src="./return_broke_corr.png">
</p>
