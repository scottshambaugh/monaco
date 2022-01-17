## [monaco](../) - Templates

### Overview
This directory contains a template you can use as a basis for building your Monte-Carlo simulation. The simulation is of a series of weighted coin flips that comes up 70% heads. By default, the simulation executes 500 draws, and tries to calculate a guess at what the coin's weighting is. These files should be very well commented, and hopefully gives you a sense of how a simulation is set up and run. 

### Running
After installing `monaco`, download the two python files here and execute `template_monte_carlo_sim.py` in a terminal window. 

### Learning
Here's some things to try out:
* Try changing the number of draws to 1000. How does this change the estimated weighting? How about if you change the number of draws to 10000?
* Try setting `savecasedata` to `False`. How much faster does the simulation run when it doesn't have to write results to file?
* Change the number of draws to 10 and save the sim to your workspace. Examine some of the data structures under the hood:
```
sim = template_monte_carlo_sim()
sim.__dict__
sim.cases[0].__dict__
sim.invars['flipper'].__dict__
``` 
* It turns out that Alex is a cheater and flips the weighted coin, while Sam doesn't know and flips a fair 50-50 coin. Try changing the sim so that it runs this scenario. Hint: try adding a second fair coin as an `InVar`, and changing which gets selected in the `template_preprocess` function based on the flipper. What is the correlation now between the flipper and getting Heads on the coin?
* Try changing the simulation and the functions so that instead of a weighted coin, you are rolling a 6-sided die.
* Add more dice as `InVar`s so that you have a total of 5. Edit the postprocessing function to answer the following questions:
  * What are the odds of a one-roll Yahtzee? (All 5 dice showing the same number)
  * What is the histogram of the sum of all 5 dice?

<p float="left" align="left">
<img width="500" height="400" src="https://raw.githubusercontent.com/scottshambaugh/monaco/main/docs/images/val_var_case_architecture.png">  
</p>

### Next Steps
Running through the steps above should start to give you a good sense of how the pieces of the Monte-Carlo simulation fit together, and hopefully the architecture diagram above starts to make sense. Once you're comfortable, head on over to the [examples](../examples) for some more complex simulations in different domains. You'll see that while the simulations have more depth to them, they all follow the same basic structure as this template.
