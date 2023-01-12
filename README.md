# pyhf_tutorial

each file exists as a python script and a jupyter notebook

- intro.py
    - generate toy example
    - create model dict
    - look at model some specs
    - do simple fit
    - draw profile likelihood
    - compare 3 different background normalisations (fixed, constrained, free)
- fit.py
    - nuisance parameters and auxdata
    - generate toys
    - pull distributions
    - significance and upper limits
- systematics.py
    - luminosity modifier
    - correlated shape modifier
    - uncorrelated shape modifier
    - toy study for model with wrong correlation assumption
    - model partially correlated uncertainties (eigendecomposition)
    - splitting uncertainties by systematic source
- ratio.py
    - create toy MC and data for a second channel
    - fit independent signal strength in channel 2, share background normalisation parameter
    - rescale signal strength to correspond to BFs
    - compute ratio of correlated BFs
    - extract ratio directly as POI from the fit
