## [Monaco](../../) - Sampling Methods

### Random Sampling Methodology

The way that random samples are drawn for different probability distributions in Monaco is that random numbers are first drawn from a uniform distribution in the range [0, 1]. These represent percentiles, which are then fed into the [inverse cumulative density function](https://en.wikipedia.org/wiki/Quantile_function) of the original probablity distribution to obtain correctly-weighted draws.

### Sampling Methods

For drawing percentiles from the underlying uniform distribution, there are currectly three sampling methods implemented in Monaco. The first is **random** sampling, which uses scipy's pseudorandom number generator as seeded by the program.

The second is **sobol** sampling. [Sobol's sequence](https://en.wikipedia.org/wiki/Sobol_sequence) is a well performing [quasi-random sequence](https://en.wikipedia.org/wiki/Low-discrepancy_sequence), which is designed to better fill a parameter space without gaps or clusters. Generally, Sobol sampling will lead to smoother and more accurate results, and faster convergence. However, it can also introduce unintended frequency content to the parameter space, as opposed to random's flat white noise spectrum. This is rarely an issue, but something to be aware of in niche applications. Note that Sobol sampling cannot be used for more than 1111 input variables, and may slow way down before that. Note also that sobol sampling is unaffected by the random seed.

The third is **sobol_random** sampling. This takes the sequence of sobol points and randomly skips the first [0, 2^10] points for each variable. Right now all this really does is allow for some randomness when adjusting the seed. In future versions this will use Owen Scrambling to optimize smoothness in the frequency spectrum.

Random sampling is the default as it is easier to conceptually understand, but most users should use sobol_random sampling for best results.

<p float="left" align="center">
<img width="768" height="240" src="random_sampling.png">  
</br>
<img width="768" height="240" src="sobol_sampling.png">
</br>
<img width="768" height="240" src="sobol_random_sampling.png">
</p>
