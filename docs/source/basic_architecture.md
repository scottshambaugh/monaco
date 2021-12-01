# Basic Architecture

At the center of the Monte Carlo simulation is a computational model which you wish to run with randomized inputs. Around this, monaco's Monte Carlo architecture is structured like a sandwich. At the top, you generate a large number of randomized values for your input variables. These input values are preprocessed into the form that your program expects, your program is run, and at the bottom the results are postprocessed to extract values for select output variables. You can then plot, collect statistics about, or otherwise use all the input and output variables from your sim. The sandwich is sliced vertically into individual cases, which are run in parallel to massively speed up computation.

<p float="left" align="center">
<img width="500" height="400" src="https://raw.githubusercontent.com/scottshambaugh/monaco/main/docs/val_var_case_architecture.png">  
</p>