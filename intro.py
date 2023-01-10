import json
from platform import python_version
from pprint import pprint

import cabinetry
import hist
import iminuit
import matplotlib.pyplot as plt
import mplhep
import numpy as np
import pyhf
from hist import Hist
from tabulate import tabulate

# matplotlib.rcParams['figure.figsize'] = [12.0, 8.0]
# matplotlib.rcParams['font.size'] = 14
# matplotlib.rcParams['axes.labelsize'] = 'large'

np.random.seed(1010)

print(f'{iminuit.__version__=}')  # 2.18.0
print(f'{pyhf.__version__=}')  # 0.7.0
print(f'{cabinetry.__version__=}')  # 0.5.1
print(f'{python_version()=}')  # 3.8.10

# %% markdown
# - pyhf
#   - documentation: https://pyhf.readthedocs.io/en/latest/
#   - tutorials: https://pyhf.github.io/pyhf-tutorial/introduction.html
#   - build complex models out of easy to handle building blocks
#   - fitting + limit setting
#   - easy way of storing the model
#   - only for binned fits
# - cabinetry
#   - documentation: https://cabinetry.readthedocs.io/en/latest/
#   - conference paper: https://www.epj-conferences.org/articles/epjconf/pdf/2021/05/epjconf_chep2021_03067.pdf
#   - provides many high level convenience functions on top of pyhf
#   - has its own configuration file that can be used to build a pyhf model (I don't do this, but could be useful)
#   - provides less flexibility than when just using pyhf
#   - still has some bugs, use with caution and cross-check results
# - PyHEP / SciPy conference talks
#   - 2022: https://indico.cern.ch/event/1150631
#   - iminuit, pyhf, cabinetry

# %%

# first, we create some toy MC and data: a gaussian signal on top of exponential background

bkg = np.random.exponential(scale=8, size=5_000)  # 5000 MC events for signal
sig = np.random.normal(loc=10, size=5_000)  # 5000 MC events for background

# fill a histogram with the hist library
ax = hist.axis.Regular(15, 0, 18, name='M')
h_bkg = Hist(ax, storage=hist.storage.Weight()).fill(bkg)

# weight signal with 0.1, only 500 signal events are expected in our toy experiment
# (but we use the larger MC sample for smaller uncertainties)
h_sig = Hist(ax, storage=hist.storage.Weight()).fill(sig, weight=0.1)

# some of the signal bins have 0 entries, this causes issues with some cabinetry functions
# we can set the values and variance of those bins to a small number
# pyhf can handle 0 entry bins since version 0.7.0, but the observed data should also be 0 in those bins
h_sig.variances()[h_sig.values() == 0] = 1E-6
h_sig.values()[h_sig.values() == 0] = 1E-4

# show our toy MC, not the statistical uncertainty for singal and background
mplhep.histplot([h_bkg, h_sig], label=['background', 'signal'], stack=True)
plt.legend()
plt.show()


# %%

# now we generate our toy data by sampling from a poission

data = np.concatenate([np.random.exponential(scale=8, size=np.random.poisson(5_000)),
                       np.random.normal(loc=10, size=np.random.poisson(500))])
h_data = Hist(ax, storage=hist.storage.Int64()).fill(data)

mplhep.histplot([h_bkg, h_sig], label=['background', 'signal'], stack=True)
h_data.plot(histtype='errorbar', color='k')
plt.legend()
plt.show()

# %%

# this function creates a pyhf model as a python dictionary from our MC and data histograms
# it can also be saved as a json file and directly read by pyhf


def create_model(h_sig, h_bkg, h_data, bkg_norm='normsys', save=False):
    """
    basic structure of the dictionary:
    {'channels': [{'name': 'channel_name',
                   'samples': [{'data': [...],
                                'modifiers': [{'data': None,
                                               'name': 'mu',
                                               'type': 'normfactor'},
                                              {'data': [...],
                                               'name': 'sig_stat_error',
                                               'type': 'staterror'}],
                                'name': 'signal'},
                               {'data': [...],
                                'modifiers': [{'data': {'hi': X.YZ, 'lo': X.YZ},
                                               'name': 'bkg_norm',
                                               'type': 'normsys'},
                                              {'data': [...],
                                               'name': 'bkg_stat_error',
                                               'type': 'staterror'}],
                                'name': 'background'}]}],
     'measurements': [{'config': {'parameters': [], 'poi': 'mu'},
                       'name': 'Measurement'}],
     'observations': [{'data': [...],
                       'name': 'channel_1'}],
     'version': '1.0.0'}
    """

    model_dict = {'measurements': [],
                  'observations': [],
                  'channels': [],
                  'version': '1.0.0'}

    model_dict['measurements'].append(
        {"name": "Measurement", "config": {"poi": "mu", "parameters": []}}
    )

    model_dict['observations'].append(
        {"name": 'channel_1',
         "data": list(h_data.values().astype(float))}
    )

    model_dict['channels'].append({
        'name': 'channel_1',
        'samples': []
    })

    model_dict['channels'][0]['samples'].append({
        'name': 'signal',
        'data': list(h_sig.values()),
        'modifiers': [
            {"name": "mu", "type": "normfactor", "data": None},
            {"name": "sig_stat_error",
             "type": "staterror",
             "data": list(np.sqrt(h_sig.variances()))}
        ]
    })

    bkg_modifiers = []
    if bkg_norm == 'normsys':  # constrained background normalisation
        bkg_modifiers.append({"name": "bkg_norm", "type": "normsys", "data": {"hi": 1.02, "lo": 0.98}})
    elif bkg_norm == 'normfactor':  # free floating background
        bkg_modifiers.append({"name": "bkg_norm", "type": "normfactor", "data": None})

    bkg_modifiers.append({"name": "bkg_stat_error",
                          "type": "staterror",
                          "data": list(np.sqrt(h_bkg.variances()))})

    model_dict['channels'][0]['samples'].append({
        'name': 'background',
        'data': list(h_bkg.values()),
        'modifiers': bkg_modifiers
    })

    pyhf.schema.validate(model_dict, 'workspace.json')

    if save:
        model_string = json.dumps(model_dict, sort_keys=True, indent=4)
        with open('workspace.json', 'w') as outfile:
            outfile.write(model_string)

    return model_dict


# in 'measurements' -> 'config' -> 'parameters' we could set inital values of the parameters,
# which would be our SM expectation, e.g. {"name": "mu", "inits": [2.0]}
model_dict = create_model(h_sig, h_bkg, h_data, bkg_norm='normsys', save=True)
pprint(model_dict)


# %%
# pyhf also has a command line interface
# ! pyhf

# %%


# ! pyhf inspect workspace.json

# %% markdown

# ## pyhf backends/optimizers
#
# What is best? Very much depens on your problem, try different combinations.
#
# ---
#
# - backends
#   - numpy
#   - jax
#   - pytorch
#   - tensorflow
#
#
# I found jax to be faster than numpy, but somethimes the optimisation fails with jax but works with numpy.
#
# ---
#
# - optimizers
#   - (i)minuit
#   - scipy.optimize
#
#
# When you are perfroming a fit and you need uncertainties, use minuit (cabinetry does this by default).
# When you don't need uncertainties (toys/limit setting) scipy can be faster and easier to handle.

# %%

# set pyhf backend
pyhf.set_backend('jax', 'minuit')

# %%

# get model and data object with cabinetry
# model_dict = cabinetry.workspace.load("workspace.json")  # from json file
model, data = cabinetry.model_utils.model_and_data(model_dict)  # use python dict directly


# %%

# simple fit with cabinetry api, run MINOS for parameter of interest mu
fit_results = cabinetry.fit.fit(model, data, minos=['mu'])
minos_unc = fit_results.minos_uncertainty['mu']
print('MINOS:', minos_unc)  # MINOS asymmetric uncertainties
print('2NLL:', fit_results.best_twice_nll)  # value of -2LL at the fitted parameters
print(tabulate(np.array([fit_results[i] for i in [2, 0, 1]]).T))

# %%

# model.config is very useful to get information about the model

model.config.npars
model.config.par_order
model.config.par_names
# model.config.par_slice('mu')
# model.config.par_slice('bkg_stat_error')

# %%

# same fit as before, but with pyhf api
# this has the advantage that we have access to the minuit object
par_estimates, results = pyhf.infer.mle.fit(data, model, return_result_obj=True, return_uncertainties=True)
print(tabulate([(par, par_estimates[model.config.par_map[par]['slice']]) for par in model.config.par_order]))

results.minuit.fmin
# results.minuit.params
# results.minuit.covariance

# %%

# performs a profile likelihood scan with cabinetry (should be the same as MINOS profile)
scan_results = cabinetry.fit.scan(model, data, "mu")
cabinetry.visualize.scan(scan_results, save_figure=False)

# %%

# we can now compare the fit with 3 different implementations on the background normalisation
# - None: fixed background
# - normsys: constrained background normalisation
# - normfactor: free floating background normalisation

profiles = {}
poi_estimates = []
for i, bkg_norm in enumerate([None, 'normsys', 'normfactor']):
    model_dict = create_model(h_sig, h_bkg, h_data, bkg_norm=bkg_norm)
    model, data = cabinetry.model_utils.model_and_data(model_dict)
    par_estimates, results = pyhf.infer.mle.fit(data, model, return_result_obj=True, return_uncertainties=True)
    profiles[bkg_norm] = results.minuit.draw_mnprofile(model.config.poi_name, band=False, text=False)
    poi_estimates.append(par_estimates[model.config.poi_index])
plt.hlines(1, *plt.xlim(), color='gray')
plt.vlines(1, *plt.ylim(), color='gray', ls=':')
plt.ylabel(r'-2$\Delta$LL')
plt.show()


for i, poi_estimate in enumerate(poi_estimates):
    plt.errorbar(poi_estimate[0].item(), -i * 0.1, xerr=poi_estimate[1].item(), marker='o')
plt.vlines(1, -1, 1, color='k', ls='--', alpha=0.5)
plt.yticks(ticks=[0, -0.1, -0.2], labels=['fixed bkg', 'constrained bkg', 'floating bkg'])
plt.ylim(-0.3, 0.1)
plt.xlabel(r'$\mu$')
plt.show()

# we end up with slightly different MLE, but more interesting are the uncertainties of our POI

# %%

# we can get a nice visual comparison of the uncertainty when we shift the parabulas to 0

for (key, value), poi_estimate, label in zip(profiles.items(),
                                             poi_estimates,
                                             ['fixed bkg', 'constrained bkg', 'floating bkg']):
    plt.plot(value[0] - poi_estimate[0].item(), value[1], label=label)
plt.xlabel(r'$\mu - \hat{\mu}$')
plt.ylabel(r'-2$\Delta$LL')
plt.legend()
plt.show()

# %%

# with the minuit object we can also draw 2D MINOS contours, here for the 68% confidence region in mu vs bkg_norm
# (we use the model with the floating background)
results.minuit.draw_mncontour('mu', 'bkg_norm')
plt.plot(1, 1, marker='+', ms=12)
plt.show()


# %%

# signal normalisation and background normalisation are anti-correlated, as we can also see from the covariance (correlation) matrix
results.minuit.covariance

# %%
# %%
# %%
# %%
