import cabinetry
import iminuit
import matplotlib.pyplot as plt
import numpy as np
import pyhf
from scipy.stats import norm
from tabulate import tabulate
from tqdm import tqdm
from uncertainties import ufloat


def fit_model(model, data, verbose=True, **kwargs):
    """
    fit model to data using cabinetry and plot pre/post-fit distributions
    """
    try:
        fit_results = cabinetry.fit.fit(model, data, **kwargs)
    except Exception as e:
        print(e)
        return None

    if verbose:
        print(tabulate(np.array([fit_results[i] for i in [2, 0, 1]]).T[[i for i, par in enumerate(model.config.par_names) if 'stat' not in par]]))

        model_pred = cabinetry.model_utils.prediction(model)  # pre-fit
        cabinetry.visualize.data_mc(model_pred, data, save_figure=False)
        plt.show()

        model_pred = cabinetry.model_utils.prediction(model, fit_results=fit_results)  # post-fit
        cabinetry.visualize.data_mc(model_pred, data, save_figure=False)
        plt.show()

        if kwargs.get('goodness_of_fit'):
            print(f'gof: {fit_results.goodness_of_fit}')

    return fit_results


def get_aux_slice_dict(model):
    """
    get slice dictionary for auxiliary data (indexed according to maindata + auxdata)
    """
    slice_edges = np.insert(np.cumsum([model.config.param_set(aux_name).n_parameters for aux_name in model.config.auxdata_order]), 0, 0)
    slice_edges += model.config.nmaindata
    slice_dict = {aux_name: slice(slice_edges[i], slice_edges[i + 1]) for i, aux_name in enumerate(model.config.auxdata_order)}
    return slice_dict


def make_sys_toys(model, sys_name, n_toys=100, assumed_value=None, n1=False):
    """
    return toy sample with sampled auxdata for selected systematic source
    n1 samples all data except for sys_name
    sys_name can be partial name of auxdata
    if sys_name is None, the expected data is returned (n1=False), or the full toy data (n1=True)
    if sys_name is 'data_stat_error', maindata is sampled
    """

    pars = model.config.suggested_init()
    if assumed_value is not None:
        pars[model.config.poi_index] = assumed_value

    # create toys
    toys = model.make_pdf(pyhf.tensorlib.astensor(pars)).sample((n_toys, ))  # toy data

    # create expected toy data
    expected_toys = np.tile(model.expected_data(pars), (n_toys, 1))

    if sys_name == 'data_stat_error':
        if n1:  # inverted
            expected_toys[:, model.config.nmaindata:] = toys[:, model.config.nmaindata:]
        else:
            expected_toys[:, :model.config.nmaindata] = toys[:, :model.config.nmaindata]
    else:
        slice_dict = get_aux_slice_dict(model)

        if n1:  # inverted
            expected_toys[:, :model.config.nmaindata] = toys[:, :model.config.nmaindata]  # use toy data
            slices = [aux_sl for aux_name, aux_sl in slice_dict.items() if (sys_name is None) or (sys_name not in aux_name)]
        else:
            slices = [aux_sl for aux_name, aux_sl in slice_dict.items() if (sys_name is not None) and (sys_name in aux_name)]

        # replace cunck of expected toys corresponding to variation of auxiliary data correspong to sys
        for sl in slices:
            expected_toys[:, sl] = toys[:, sl]

    return expected_toys


def bestfit_toy_valid(model, sys_name='all', n_toys=200, return_results=False):
    """
    toy validation only using bestfits
    """

    backend = pyhf.get_backend()
    pyhf.set_backend("jax", pyhf.optimize.scipy_optimizer())  # very fast

    pars = model.config.suggested_init()
    if sys_name == 'all':
        toys = model.make_pdf(pyhf.tensorlib.astensor(pars)).sample((n_toys, ))  # toy data
    else:
        toys = make_sys_toys(model, sys_name, n_toys=n_toys)

    bestfits = []
    toy_results = []
    n_fail = 0
    for toy in tqdm(toys):
        try:
            par_estimates, results = pyhf.infer.mle.fit(toy, model, return_result_obj=True, return_uncertainties=False)
            toy_results.append(results)
            assert results.success
            bestfits.append(par_estimates[model.config.poi_index].item())
        except Exception as e:
            n_fail += 1
            # print(e)
            print(f'optimisation failed already {n_fail} times :(')

    if return_results:
        return toy_results

    assumed_value = pars[model.config.poi_index]
    # mean = np.mean(bestfits)
    # std = np.std(bestfits)  # absolute, actually append relative uncertainties

    # from gaussian fit
    mu, sigma = fit_pull(bestfits, x0=[1, 0.001], show=False)
    mu = ufloat(*mu)
    sigma = ufloat(*sigma)

    pyhf.set_backend(*backend)

    # return absolute uncertainty
    return sigma


def fit_pull(datos, x0=[0, 1], show_bins=40, show=True, xlabel='pull'):
    datos = np.asarray(datos)
    datos = datos[~np.isnan(datos)]

    def nll(p, data):
        mu = p[0]
        sigma = p[1]
        return np.sum(np.log(2 * np.pi * (sigma**2)) / 2 + ((data - mu)**2) / (2 * (sigma**2)))

    # res = minimize(nll, x0, args=datos)  # scipy
    try:
        res = iminuit.minimize(nll, x0, args=(datos,), bounds=[None, (0, None)])  # minuit
        # print(f'{res.success=} (initial try)')
        assert res.success
    except Exception as e:
        print(f'{res.success=} (initial try)')
        print(e)
        for sigma_init in np.exp(np.linspace(-12, 0.5, 15)):
            x0_new = [x0[0], sigma_init]
            print(f'trying {sigma_init=}')
            res = iminuit.minimize(nll, x0_new, args=(datos,), bounds=[None, (0, None)])  # minuit
            print(f'{res.success=}')
            if res.success:
                break
        if not res.success:
            for mu_init in np.linspace(-1, 1, 15):
                x0_new = [mu_init, x0[1]]
                print(f'trying {mu_init=}')
                res = iminuit.minimize(nll, x0_new, args=(datos,), bounds=[None, (0, None)])  # minuit
                print(f'{res.success=}')
                if res.success:
                    break

    assert res.success

    mu, sigma = res.x
    mu_unc = np.sqrt(res.hess_inv[0, 0])
    sigma_unc = np.sqrt(res.hess_inv[1, 1])

    if show:
        # the histogram of the data
        n, bins, patches = plt.hist(datos, show_bins, density=True, facecolor='green', histtype='stepfilled', alpha=0.5)

        # add a 'best fit' line
        y = norm.pdf(bins, mu, sigma)
        l = plt.plot(bins, y, 'r--', linewidth=1.5)

        # plot
        plt.xlabel(xlabel)
        plt.ylabel('frequency')
        plt.title(r'$\mathrm{toys:}\ \mu=%.3f \pm %.3f,\ \sigma=%.3f \pm %.3f$' % (mu, mu_unc, sigma, sigma_unc))
        plt.grid(True)
        plt.gca().set_axisbelow(True)
        # plt.xlim(-4, 4)

    return np.array([mu, mu_unc]), np.array([sigma, sigma_unc])
