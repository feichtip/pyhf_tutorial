import cabinetry
import hist
import matplotlib
import matplotlib.pyplot as plt
import mplhep
import numpy as np
import uncertainties
import utils
from hist import Hist

matplotlib.rcParams['figure.figsize'] = [8.0, 6.0]
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['axes.labelsize'] = 'large'

np.random.seed(101)

# %% markdown
# fit the ratio
# %%

# we create more toy MC and data for a second channel

bkg = np.random.uniform(0, 18, size=5_000)
sig = np.random.exponential(scale=20, size=5_000)

# fill a histogram with the hist library
ax = hist.axis.Regular(15, 0, 18, name='M2')
h_bkg = Hist(ax, storage=hist.storage.Weight()).fill(bkg, weight=0.5)
h_sig = Hist(ax, storage=hist.storage.Weight()).fill(sig, weight=0.2)
h_sig.variances()[h_sig.values() == 0] = 1E-6
h_sig.values()[h_sig.values() == 0] = 1E-4

data = np.concatenate([np.random.uniform(0, 18, size=np.random.poisson(2_500)),
                       np.random.exponential(scale=20, size=np.random.poisson(1_000))])
h_data = Hist(ax, storage=hist.storage.Int64()).fill(data)

# show our toy MC and data
mplhep.histplot([h_bkg, h_sig], label=['background', 'signal'], stack=True)
h_data.plot(histtype='errorbar', color='k')
plt.legend()
plt.show()

# %%

# let's assume the background normalisation is correlated between both channels

model_dict = cabinetry.workspace.load("workspace.json")


model_dict['observations'].append(
    {"name": 'channel_2',
     "data": list(h_data.values().astype(float))}
)

model_dict['channels'].append({
    'name': 'channel_2',
    'samples': []
})

model_dict['channels'][0]['samples'][1]['modifiers'][0] = {"name": "bkg_norm", "type": "normsys", "data": {"hi": 1.2, "lo": 0.8}}
model_dict['channels'][1]['samples'].append({
    'name': 'signal_2',
    'data': list(h_sig.values()),
    'modifiers': [
        {"name": "mu_2", "type": "normfactor", "data": None},
        {"name": "sig_stat_error_2",
         "type": "staterror",
         "data": list(np.sqrt(h_sig.variances()))}
    ]
})

bkg_modifiers = []
bkg_modifiers.append({"name": "bkg_norm", "type": "normsys", "data": {"hi": 1.4, "lo": 0.6}})
bkg_modifiers.append({"name": "bkg_stat_error_2",
                      "type": "staterror",
                      "data": list(np.sqrt(h_bkg.variances()))})

model_dict['channels'][1]['samples'].append({
    'name': 'background',
    'data': list(h_bkg.values()),
    'modifiers': bkg_modifiers
})

utils.save_model(model_dict, 'workspace_2channel')

# plot
model, data = cabinetry.model_utils.model_and_data(model_dict)
fit_results = utils.fit_model(model, data, goodness_of_fit=True)

print(f'correlation mu, mu2: {fit_results.corr_mat[fit_results.labels.index("mu"), fit_results.labels.index("mu_2")]:.3f}')

# %% markdown
# ### fit BF/cross section instead of signal strenght
# - just need to apply a scaling factor to the templates
# - signal strenght of 1 would correspond to measuring the same BF as is in the generator
# - for this toy example:
# - signal in channel_1: BF=0.5
# - signal in channel_2: BF=0.25
# %%

model_dict = cabinetry.workspace.load("workspace_2channel.json")

parameters = model_dict['measurements'][0]['config']['parameters']
parameters.append({"name": "inv_BF_1", "inits": [1 / 0.5], "fixed": True})
parameters.append({"name": "inv_BF_2", "inits": [1 / 0.25], "fixed": True})
parameters.append({"name": "BF_1", "inits": [0.5]})
parameters.append({"name": "BF_2", "inits": [0.25]})

model_dict['measurements'][0]['config']['poi'] = 'BF_1'
model_dict['channels'][0]['samples'][0]['modifiers'][0]['name'] = 'BF_1'  # normfactor
model_dict['channels'][1]['samples'][0]['modifiers'][0]['name'] = 'BF_2'  # normfactor

model_dict['channels'][0]['samples'][0]['modifiers'].append(
    {"name": "inv_BF_1", "type": "normfactor", "data": None})  # fixed normfactor
model_dict['channels'][1]['samples'][0]['modifiers'].append(
    {"name": "inv_BF_2", "type": "normfactor", "data": None})  # fixed normfactor

utils.save_model(model_dict, 'workspace_bf')

# %%


model_dict = cabinetry.workspace.load("workspace_bf.json")
model, data = cabinetry.model_utils.model_and_data(model_dict)
fit_results = utils.fit_model(model, data, goodness_of_fit=True)

par_estimates = [(fit_results.bestfit[fit_results.labels.index(par_name)],
                  fit_results.uncertainty[fit_results.labels.index(par_name)])
                 for par_name in ['BF_1', 'BF_2']]

# %%
# now we can compute the ratio of the fitted branching fractions, taking into account their correlation

print('\ntaking into account correlation between fitted parameters')
bf1_i = fit_results.labels.index('BF_1')
bf2_i = fit_results.labels.index('BF_2')
corr_matrix = fit_results.corr_mat[[[bf1_i, bf1_i], [bf2_i, bf2_i]], [bf1_i, bf2_i]]
print(corr_matrix)
bf1, bf2 = uncertainties.correlated_values_norm(par_estimates, corr_matrix)
print(f'ratio={bf1 / bf2:.3}')

print('\nwithout correlation:')
corr_matrix = np.identity(2)
bf1, bf2 = uncertainties.correlated_values_norm(par_estimates, corr_matrix)
print(f'ratio={bf1 / bf2:.3}')

# %%

# or we can also fit the ratio directly

model_dict = cabinetry.workspace.load("workspace_bf.json")

parameters = model_dict['measurements'][0]['config']['parameters']

# replace BF_1 with ratio
parameters[2] = {"name": "ratio", "inits": [2.0], "bounds": [[0, 20]]}
model_dict['measurements'][0]['config']['poi'] = 'ratio'
model_dict['channels'][0]['samples'][0]['modifiers'][0]['name'] = 'ratio'

# multiply BF_2 to channel_1 signal normalisation, then we measure directly BF_1 / BF_2
model_dict['channels'][0]['samples'][0]['modifiers'].append(
    {"name": "BF_2", "type": "normfactor", "data": None})

utils.save_model(model_dict, 'workspace_ratio')

# %%

model_dict = cabinetry.workspace.load("workspace_ratio.json")
model, data = cabinetry.model_utils.model_and_data(model_dict)
fit_results = utils.fit_model(model, data, goodness_of_fit=True)
