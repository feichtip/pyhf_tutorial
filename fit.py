import cabinetry
import hist
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyhf
import utils
from hist import Hist
from IPython.display import Image
from pyhf.contrib.viz import brazil
from tqdm import tqdm

matplotlib.rcParams['figure.figsize'] = [8.0, 6.0]
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['axes.labelsize'] = 'large'

np.random.seed(1010)

# %%


# get model from before, with constrained background (normsys)
model_dict = cabinetry.workspace.load("workspace.json")  # from json file

parameters = model_dict['measurements'][0]['config']['parameters']
parameters.append({"name": "bkg_norm", "inits": [0.0], "bounds": [[-15, 15]]})

model, data = cabinetry.model_utils.model_and_data(model_dict)

# fit and show pre/post-fit distributions
_ = utils.fit_model(model, data)

# %%

# a function to sample data from our underlying model


def sample_data(bkg_scale=1, sig_scale=1):
    axis = hist.axis.Regular(15, 0, 18)
    data = np.concatenate([np.random.exponential(scale=8, size=(np.random.poisson(bkg_scale * 5_000))),
                           np.random.normal(loc=10, size=(np.random.poisson(sig_scale * 500)))])

    h = Hist(axis, storage=hist.storage.Int64()).fill(data)
    return h


sample_data(1, 1)

# import scipy
# bin_i = 14
# axis = hist.axis.Regular(15, 0, 18)
# expected = (scipy.stats.expon(scale=8).cdf(axis.edges[bin_i + 1]) - scipy.stats.expon(scale=8).cdf(axis.edges[bin_i])) * 5_000 * 1.1
# bin0 = [sample_data(1.1, 1).values()[bin_i] for i in range(10_000)]
# np.mean(bin0)
# expected
# plt.hist(bin0, bins=50)

# %% markdown
# ### data in pyhf
# the data for our model always contains the bin entries ('maindata') and data associated with nuisance parameters ('auxdata')
#
# this can be a little bit confusing at first, but since we work in a frequentist framework it is the correct way to set up the likelihood
#
# in principle this can be additional information coming from a control channel, but for most appliciations of pyhf this
# should be left to the suggested initialisation paramters (information about the size of the systematics which manifests
# itself in the constraint terms is already included in the model from the specification)

#
# %%

Image(filename='fit_model.png')

# %%
Image(filename='modifiers.png')

# %%

data  # passed to the fit, always maindata + auxdata
model.config.nmaindata  # number of maindat
model.config.nauxdata  # number of auxdata
model.config.auxdata  # only the auxdata for the specific model
model.config.auxdata_order  # order of the modifiers corresponding to the auxdata


# %%

# to fit our model to the sampled data we have to merge it first with the auxdata
toy_data = list(sample_data(bkg_scale=1.0, sig_scale=1.0).values()) + model.config.auxdata

# we can also fit our model to the expected data (asimov data)
# toy_data = model.expected_data(model.config.suggested_init())

fit_results = utils.fit_model(model, toy_data)


# %% markdown
# - we have a 2% uncertainty on the background normalisation in our model
# - the fitted parameter ('bkg_norm') in this case corresponds to the 'pull' away from the expected normalisation (our nominal MC template)
# - this is, how many standard deviations of the given uncertainty (in our case 1 sigma = 2%) does the fitted parameter deviate?
# - pull plot with cabinetry is currently bugged, only correct for gaussian constraints centered at 0
# ---
# how does the bkg level affect the fitted parameters?
# %%

cabinetry.visualize.pulls(fit_results, exclude=[par for par in model.config.par_names if 'stat' in par] + [model.config.poi_name], save_figure=False)

# %% markdown
# a more systematic way to study such effects is with a toy study where we perform the fit many times
# - with pyhf we can easily sample data from our model
# - this will also sample auxdata from the constraint terms
# - it is important to include the sampled auxdata, since we want to sample from our whole model with all systematic variations
# - with this we can just study the estimator within our model, the toy study does not give us any information of how good our model can describe reality
# %%

# pyhf.set_backend('numpy')
pyhf.set_backend('jax')

np.random.seed(42)

n_toys = 100
pars = model.config.suggested_init()
true_mu = 1.0
true_bkg_norm = 1  # 1->2%
pars[model.config.par_slice('mu')] = [true_mu]
pars[model.config.par_slice('bkg_norm')] = [true_bkg_norm]
toys = model.make_pdf(pyhf.tensorlib.astensor(pars)).sample((n_toys,))

fit_results = utils.fit_model(model, toys[0])


# %%

bestfits = []
uncertainties = []

for toy in tqdm(toys):
    fit_results = utils.fit_model(model, toy, verbose=False)
    if fit_results:
        bestfits.append(fit_results.bestfit)
        uncertainties.append(fit_results.uncertainty)

bestfits = np.array(bestfits)
uncertainties = np.array(uncertainties)

# # %%

for par_name, true_val in zip(['mu', 'bkg_norm'], [true_mu, true_bkg_norm]):
    par_slice = model.config.par_slice(par_name)
    pull = (bestfits[:, par_slice] - true_val) / uncertainties[:, par_slice]
    utils.fit_pull(pull, show_bins=20, xlabel=f'pull ({par_name})')
    plt.show()
    print(f'mean: {pull.mean():.4f}, std: {pull.std():.4f}')

# %% markdown
# - pyhf also provides a nice framework to compute confidence intervals set limits on the POI
# - lets assume we don't observe a clear signal in our data, what is its significance and the upper limit on the signal strength?
#
# %%

np.random.seed(12)
toy_data = list(sample_data(bkg_scale=1.0, sig_scale=0.05).values()) + model.config.auxdata
_ = utils.fit_model(model, toy_data)

# %%

# we can compute the observed and expected significance (for mu=1) with cabinetry
cabinetry.fit.significance(model, toy_data)
# or directly with pyhf
pyhf.infer.hypotest(0, toy_data, model, test_stat="q0", return_expected_set=True)  # returns p-values

# %%

# do a parameter scan over different poi values to set a upper limit on mu
poi_vals = np.linspace(0, 0.5, 21)
results = [
    pyhf.infer.hypotest(
        test_poi, toy_data, model, test_stat="qtilde", return_expected_set=True,
    )
    for test_poi in poi_vals
]

# the brazil band in this case shows the expected limit for mu=0 !
fig, ax = plt.subplots()
fig.set_size_inches(7, 5)
brazil.plot_results(poi_vals, results, ax=ax)
plt.show()

# %%

# calculate upper limit with interpolation
observed = np.asarray([h[0] for h in results]).ravel()
expected = np.asarray([h[1][2] for h in results]).ravel()
print(f"Upper limit (obs): μ = {np.interp(0.05, observed[::-1], poi_vals[::-1]):.4f}")
print(f"Upper limit (exp): μ = {np.interp(0.05, expected[::-1], poi_vals[::-1]):.4f}")

# %%
# %%
