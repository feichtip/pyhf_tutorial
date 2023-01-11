import cabinetry
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import utils
from IPython.display import Image
from tabulate import tabulate
from tqdm import tqdm
from uncertainties import unumpy

matplotlib.rcParams['figure.figsize'] = [8.0, 6.0]
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['axes.labelsize'] = 'large'

np.random.seed(1010)

# %% markdown
# - all systematic uncertainties can be included in the model with nuisance parameters
# - defined in the pyhf specification (usually as up/down variatons)
# - depending on the type of systematic and its correlation, different modifiers can be used
# - modifiers can be shared over different samples (by having the same name), except for uncorrelated shape
# - https://pyhf.readthedocs.io/en/v0.7.0/likelihood.html
#   - uncorrelated shape: *shapesys*
#   - correlated shape: *histosys*
#   - normalisation uncertainty: *normsys*
#   - mc statistical uncertainty: *staterror*
#   - luminosity: *lumi*
#   - unconstrained normalisation: *normfactor*

# %%

Image(filename='modifiers.png')


# %% markdown
# ## luminosity
# %%

# get model from before, with constrained background (normsys)
model_dict = cabinetry.workspace.load("workspace.json")  # from json file
model, data = cabinetry.model_utils.model_and_data(model_dict)

# multiply data by a factor of 2
mod_data = [d * 2.0 for i, d in enumerate(data) if (i < model.config.nmaindata)]
_ = utils.fit_model(model, mod_data + data[model.config.nmaindata:], goodness_of_fit=True)

# %%

# now we add the luminosity modifier

lumi = 2.0  # templates are scaled by this value
lumi_uncertainty = 0.02  # 2% uncertainty

model_dict = cabinetry.workspace.load("workspace.json")
model_dict['measurements'][0]['config']['parameters'].append(
    {"name": "lumi", "auxdata": [lumi], "sigmas": [lumi_uncertainty * lumi], "bounds": [[0.5, 5.0]], "inits": [lumi]})
model_dict['channels'][0]['samples'][0]['modifiers'].append({'name': 'lumi', 'type': 'lumi', 'data': None})  # signal modifiers
model_dict['channels'][0]['samples'][1]['modifiers'].append({'name': 'lumi', 'type': 'lumi', 'data': None})  # background modifiers

model, data = cabinetry.model_utils.model_and_data(model_dict)
_ = utils.fit_model(model, mod_data + data[model.config.nmaindata:], goodness_of_fit=True)

# %% markdown
# ## shape modifier
# %%

model_dict = cabinetry.workspace.load("workspace.json")
model, data = cabinetry.model_utils.model_and_data(model_dict)
bkg_data = model_dict['channels'][0]['samples'][1]['data']

# that's our expected deviation on the background template
abs_uncrt = np.linspace(-0.1, 0.35, len(bkg_data)) * bkg_data

# construct a covariance matrix
cov = abs_uncrt[np.newaxis].T * abs_uncrt  # 100% correlated
# cov = np.diag(abs_uncrt)**2  # 100% uncorrelated

plt.imshow(cov, origin='lower')
plt.xlabel('bin')
plt.ylabel('bin')
plt.show()


# %%

# sample from multivariate gauss using the covariance matrix
np.random.seed(80)
mod_data = np.random.multivariate_normal(mean=data[:model.config.nmaindata], cov=cov)
plt.step(range(len(mod_data)), (mod_data - data[:model.config.nmaindata]) / bkg_data, where='mid')
plt.xlabel('bin')
plt.ylabel('relative variation')
plt.show()

# %%

# fit the modified data with our original model
fit_results = utils.fit_model(model, list(mod_data) + data[model.config.nmaindata:], goodness_of_fit=True)


# %%

# add correlated shape as background modifier
corr_model_dict = cabinetry.workspace.load("workspace.json")
corr_model_dict['channels'][0]['samples'][1]['modifiers'].append({"name": 'corr_bkg_shape',
                                                                  "type": "histosys",
                                                                  "data": {"hi_data": list(bkg_data + abs_uncrt),
                                                                           "lo_data": list(bkg_data - abs_uncrt)}
                                                                  })
corr_model, corr_data = cabinetry.model_utils.model_and_data(corr_model_dict)
_ = utils.fit_model(corr_model, list(mod_data) + corr_data[corr_model.config.nmaindata:], goodness_of_fit=True)

# %%

# add uncorrelated shape as background modifier
uncorr_model_dict = cabinetry.workspace.load("workspace.json")
uncorr_model_dict['channels'][0]['samples'][1]['modifiers'].append({"name": 'uncorr_bkg_shape',
                                                                    "type": "shapesys",
                                                                    "data": list(np.abs(abs_uncrt))})

uncorr_model, uncorr_data = cabinetry.model_utils.model_and_data(uncorr_model_dict)
_ = utils.fit_model(uncorr_model, list(mod_data) + uncorr_data[uncorr_model.config.nmaindata:], goodness_of_fit=True)

# %%

# toy study with both models
# what if we model the correlated shape as uncorrelated and vice versa?

n_toys = 50
toy_mu = 1.0
minos = False
results = {}

for model, corr_type in zip([corr_model, uncorr_model], ['corr_bkg', 'uncorr_bkg']):

    # make toys for all paramters except 'bkg_shape'
    toys = utils.make_sys_toys(model, sys_name='bkg_shape', n_toys=n_toys, assumed_value=toy_mu, n1=True)

    # sample from multivariate gauss to simulate background shape variation
    while True:
        toys_actualdata = [np.random.multivariate_normal(mean=mean, cov=cov) for mean in toys[:, :model.config.nmaindata]]
        if (np.array(toys_actualdata) < 0).sum() == 0:
            break

    # replace actualdata in toys with new sampled data
    toys[:, :model.config.nmaindata] = toys_actualdata

    fails = 0
    res = []
    for i, toy in tqdm(enumerate(toys), total=n_toys):
        try:
            # preform the fit
            fit_results = cabinetry.fit.fit(model, toy, goodness_of_fit=True, minos=model.config.poi_name if minos else [])

            # get some information from the fit results
            bestfit = fit_results.bestfit[model.config.poi_index].item()
            fit_unc = fit_results.uncertainty[model.config.poi_index].item()
            gof = fit_results.goodness_of_fit
            nll = fit_results.best_twice_nll

            # compute pull
            if minos:
                minos_unc = fit_results.minos_uncertainty[model.config.poi_name]
                minos_unc_sym = np.abs(minos_unc).sum() / 2
                pull = (bestfit - toy_mu) / minos_unc_sym
            else:
                minos_unc = [np.nan, np.nan]
                pull = (bestfit - toy_mu) / fit_unc

            # print results and append to list
            result = [bestfit, fit_unc, minos_unc[0], minos_unc[1], pull, nll, gof]
            names = ['mu', 'unc', 'dn', 'up', 'pull', 'nll', 'gof']
            print(f'{i+1:>3}' + ' '.join([f'{name:>6}: {res:<7.3f}' for name, res in zip(names, result)]))
            res.append(result)
        except Exception as e:
            print(e)
            fails += 1
            print(f'optimisation failed already {fails} times :(')

    # clean up some failed fits if there are any
    results[corr_type] = np.array(res)[~np.isclose(np.array(res)[:, :4], 0).any(1) & ~np.isnan(np.array(res)[:, -1])]


# %%

for corr_type, res in results.items():
    print(len(res))
    utils.fit_pull(np.array(res)[:, 4], show_bins=20, xlabel=f'pull ({corr_type})')
    plt.show()

# %% markdown
# - it is important to model all systematic variations and their correlation correctly, but not easy to validate that this is the case
# - we can only test this here because we have the underlying model, this is usually not available
# - if we sample from our model we already assume that correlations/systematics are correct!
#
# %% markdown

# ## how to model any arbitrary correlation
# - we can model any covariance matrix with multiple, independent nuisance parameters, using correlated shape variations (histosys)
# - https://indico.cern.ch/event/1051224/contributions/4534929/
# %%

fill_val = 0.5  # off-diagonal elements in correlation matrix
corr = np.identity(len(abs_uncrt)) * (1 - fill_val) + np.full((len(abs_uncrt), len(abs_uncrt)), fill_val)
cov = np.diag(abs_uncrt) @ corr @ np.diag(abs_uncrt)

# sample from multivariate gauss using the covariance matrix
np.random.seed(80)
mod_data = np.random.multivariate_normal(mean=data[:model.config.nmaindata], cov=cov)

# %%

# covariance -> correlated shape variations
utils.plot_corr(cov)
plt.show()

e_val, e_vec = np.linalg.eigh(cov)
n_Eval_before = len(e_val)

to_keep = e_val > 1
n_Eval_keep = to_keep.sum()

print(f'remaining eigenvalues: {n_Eval_keep}/{n_Eval_before}')
e_val = e_val[to_keep]
e_vec = e_vec[:, to_keep]
lamb = np.diag(e_val)

gamma = (e_vec @ np.sqrt(lamb))
covMat_rep = gamma @ gamma.T

utils.plot_corr(covMat_rep)
plt.title(f'using {n_Eval_keep}/{n_Eval_before} eigenvalues')
plt.show()
# %%

# add correlated shapes as background modifier
part_model_dict = cabinetry.workspace.load("workspace.json")
for i, abs_var in enumerate(gamma.T):
    plt.step(range(len(abs_var)), 1 + abs_var / bkg_data, where='mid')
    part_model_dict['channels'][0]['samples'][1]['modifiers'].append({"name": f'corr_bkg_shape_{i}',
                                                                      "type": "histosys",
                                                                      "data": {"hi_data": list(bkg_data + abs_var),
                                                                               "lo_data": list(bkg_data - abs_var)}
                                                                      })
part_model, part_data = cabinetry.model_utils.model_and_data(part_model_dict)
_ = utils.fit_model(part_model, list(mod_data) + part_data[model.config.nmaindata:], goodness_of_fit=True)

# %%
# %%

# %% markdown
# ### splitting uncertainty on POI by systeamtic source
# %%

np.random.seed(42)
model_dict = cabinetry.workspace.load("workspace.json")
model, data = cabinetry.model_utils.model_and_data(model_dict)


# %%

utils.bestfit_toy_valid(model, sys_name='all', n_toys=500)

# %%

sys_uncrt = {}
sys_names = ['bkg_shape', 'bkg_norm', 'sig_stat_error', 'bkg_stat_error', 'data_stat_error']
for sys_name in sys_names:
    sys_uncrt[sys_name] = utils.bestfit_toy_valid(model, sys_name=sys_name)

# %%

print(tabulate(sys_uncrt.items()))
print(f'quadrature sum: {unumpy.sqrt(np.sum([uncrt**2 for uncrt in sys_uncrt.values()]))}')

# %%
# %%
# %%

# # parabula only for bkg norm, bc of 1 parameter with gaussian constraint ?
# toy_results = utils.bestfit_toy_valid(model, sys_name='all', n_toys=500, return_results=True)
# toy_results = utils.bestfit_toy_valid(model, sys_name='bkg_norm', n_toys=500, return_results=True)
# x_nll = np.array([[toy_result.x[model.config.poi_index], toy_result.fun] for toy_result in toy_results])
# plt.plot(*x_nll.T, marker='.', ls='')
# plt.hist(x_nll[:, 0], bins=50)
# plt.hist(x_nll[:, 1], bins=50)
