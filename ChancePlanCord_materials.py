import numpy
import chaospy
import scipy.stats
import scipy.special
import math
import sympy
import sklearn.mixture

'''
Perform uncertainty quantification
Nataf transformation

'''


class prepareMaterial:

    def __init__(self, dimen, nstage):
        self.dimen = dimen
        self.nstage = nstage

    def characterNormal(self, cent, std):
        # cent-average value of normal distribution
        # std-standard deviation of normal distribution

        # distr = scipy.stats.norm(loc=cent, scale=std)
        # mean = distr.mean()
        # sigma = distr.std()

        mean = cent
        sigma = std

        return mean, sigma

    def characterWeibull(self, k, c):
        # k-shape parameter of Weibull distribution
        # c-scale parameter of Weibull distribution

        # distr = scipy.stats.weibull_min(c=k, scale=c)
        # mean = distr.mean()
        # sigma = distr.std()

        mean = scipy.special.gamma(1+1/k)*c
        var = (scipy.special.gamma(1+2/k)-scipy.special.gamma(1+1/k)**2)*c**2
        sigma = math.sqrt(var)

        return mean, sigma

    def characterBeta(self, a, b):
        # a-first shape parameter of Beta distribution
        # b-second shape parameter of Beta distribution

        # distr = scipy.stats.beta(a=a, b=b)
        # mean = distr.mean()
        # sigma = distr.std()

        mean = a/(a+b)
        var = a*b/((a+b)**2)/(a+b+1)
        sigma = math.sqrt(var)

        return mean, sigma

    def characterUniform(self, a, b):
        # a-lower boundary of Uniform distribution
        # b-upper boundary of Uniform distribution

        # distr = scipy.stats.uniform(loc=a, scale=b-a)
        # mean = distr.mean()
        # sigma = distr.std()

        mean = (a+b)/2
        var = (b-a)**2/12
        sigma = math.sqrt(var)

        return mean, sigma

    def calculateCharaters(self, hdata_dgen, hdata_load, hdata_evel):
        # hdata_dgen-historical data of distributed generator
        # hdata_load-historical data of conventional load
        # hdata_evel-historical data of electric vehicle charging load

        Load_amean = numpy.mean(hdata_load)
        Load_sigma = numpy.std(hdata_load)
        Load_medin = numpy.quantile(hdata_load, 0.9)
        print('distribution of Load: mean={:.2f}, sigma={:.2f}, medin={:.2f}'.format(Load_amean, Load_sigma, Load_medin))

        Evel_amean = numpy.mean(hdata_evel)
        Evel_sigma = numpy.std(hdata_evel)
        Evel_medin = numpy.quantile(hdata_evel, 0.9)
        print('distribution of Evel: mean={:.2f}, sigma={:.2f}, medin={:.2f}'.format(Evel_amean, Evel_sigma, Evel_medin))

        Wind_amean = numpy.mean(hdata_dgen['Wind'])
        Wind_sigma = numpy.std(hdata_dgen['Wind'])
        Wind_medin = numpy.quantile(hdata_dgen['Wind'], 0.9)
        print('distribution of Wind: mean={:.2f}, sigma={:.2f}, medin={:.2f}'.format(Wind_amean, Wind_sigma, Wind_medin))

        Sola_amean = numpy.mean(hdata_dgen['Sola'])
        Sola_sigma = numpy.std(hdata_dgen['Sola'])
        Sola_medin = numpy.quantile(hdata_dgen['Sola'], 0.9)
        print('distribution of Sola: mean={:.2f}, sigma={:.2f}, medin={:.2f}'.format(Sola_amean, Sola_sigma, Sola_medin))

        # print(scipy.stats.skew(hdata_load, bias=False))
        # print(scipy.stats.skew(hdata_evel, bias=False))
        # print(scipy.stats.skew(hdata_dgen['Wind'], bias=False))
        # print(scipy.stats.skew(hdata_dgen['Sola'], bias=False))
        #
        # print(scipy.stats.kurtosis(hdata_load, bias=False))
        # print(scipy.stats.kurtosis(hdata_evel, bias=False))
        # print(scipy.stats.kurtosis(hdata_dgen['Wind'], bias=False))
        # print(scipy.stats.kurtosis(hdata_dgen['Sola'], bias=False))

        self.SorLd_amean = {'Load': Load_amean, 'Evel': Evel_amean, 'Wind': Wind_amean, 'Sola': Sola_amean}
        self.SorLd_sigma = {'Load': Load_sigma, 'Evel': Evel_sigma, 'Wind': Wind_sigma, 'Sola': Sola_sigma}
        self.SorLd_medin = {'Load': Load_medin, 'Evel': Evel_medin, 'Wind': Wind_medin, 'Sola': Sola_medin}
        # exit()

    def constructUniDistribution(self, hdata_dgen, hdata_load, hdata_evel, method='gmm'):
        # hdata_dgen-historical data of distributed generator
        # hdata_load-historical data of conventional load
        # hdata_evel-historical data of electric vehicle charging load
        # method-uncertainty model methods

        Load_distr = chaospy.Normal(mu=0.0, sigma=1.0)
        Evel_distr = chaospy.Normal(mu=0.0, sigma=1.0)
        Wind_distr = chaospy.Normal(mu=0.0, sigma=1.0)
        Sola_distr = chaospy.Normal(mu=0.0, sigma=1.0)

        if method == 'kde':
            Load_distr = chaospy.GaussianKDE(hdata_load)
            Evel_distr = chaospy.GaussianKDE(hdata_evel)
            Sola_distr = chaospy.GaussianKDE(hdata_dgen['Sola'])
        elif method == 'gmm':
            Load_mixture = sklearn.mixture.GaussianMixture(n_components=1, covariance_type="full", random_state=123)
            Evel_mixture = sklearn.mixture.GaussianMixture(n_components=1, covariance_type="full", random_state=123)
            Sola_mixture = sklearn.mixture.GaussianMixture(n_components=1, covariance_type="full", random_state=123)

            Load_mixture.fit(hdata_load.reshape(-1, 1))
            Evel_mixture.fit(hdata_evel.reshape(-1, 1))
            Sola_mixture.fit(hdata_dgen['Sola'].reshape(-1, 1))

            Load_distr = chaospy.GaussianMixture(Load_mixture.means_, Load_mixture.covariances_, Load_mixture.weights_)
            Evel_distr = chaospy.GaussianMixture(Evel_mixture.means_, Evel_mixture.covariances_, Evel_mixture.weights_)
            Sola_distr = chaospy.GaussianMixture(Sola_mixture.means_, Sola_mixture.covariances_, Sola_mixture.weights_)
        elif method == 'vbgmm':
            Load_mixture = sklearn.mixture.BayesianGaussianMixture(n_components=1, covariance_type="full", weight_concentration_prior_type='dirichlet_distribution', weight_concentration_prior=1e6,  random_state=123)
            Evel_mixture = sklearn.mixture.BayesianGaussianMixture(n_components=1, covariance_type="full", weight_concentration_prior_type='dirichlet_distribution', weight_concentration_prior=1e-3, random_state=123)
            Sola_mixture = sklearn.mixture.BayesianGaussianMixture(n_components=1, covariance_type="full", weight_concentration_prior_type='dirichlet_distribution', weight_concentration_prior=1e6,  random_state=123)

            Load_mixture.fit(hdata_load.reshape(-1, 1))
            Evel_mixture.fit(hdata_evel.reshape(-1, 1))
            Sola_mixture.fit(hdata_dgen['Sola'].reshape(-1, 1))

            Load_distr = chaospy.GaussianMixture(Load_mixture.means_, Load_mixture.covariances_, Load_mixture.weights_)
            Evel_distr = chaospy.GaussianMixture(Evel_mixture.means_, Evel_mixture.covariances_, Evel_mixture.weights_)
            Sola_distr = chaospy.GaussianMixture(Sola_mixture.means_, Sola_mixture.covariances_, Sola_mixture.weights_)

        self.SorLd_margin = {'Load': Load_distr, 'Evel': Evel_distr, 'Wind': Wind_distr, 'Sola': Sola_distr}

    def contructMulDistribution(self, Nrvs):
        # Nrvs-number of random variables

        self.SorLd_multi = dict()
        for key in Nrvs:
            if Nrvs[key] > 0:
                self.SorLd_multi[key] = chaospy.Iid(self.SorLd_margin[key], Nrvs[key])
            else:
                self.SorLd_multi[key] = []

    def computeNormalCovMat(self, rho_parameter, Nrvs, Sprvs, zone):
        # rho_parameter-correlation coefficient
        # Nrvs-number of random variables

        def rhoconvert(distr, mean, sigma, yi, yj, randseed):
            ni = chaospy.Normal(mu=0.0, sigma=1.0).cdf(yi)
            numpy.random.seed(randseed)
            ti = distr[0].ppf(ni)
            nj = chaospy.Normal(mu=0.0, sigma=1.0).cdf(yj)
            numpy.random.seed(randseed)
            tj = distr[1].ppf(nj)

            fai = 1/(2*math.pi*sympy.sqrt(1-x**2))*sympy.exp(-1/2*(yi**2-2*x*yi*yj+yj**2)/(1-x**2))
            mmt = ((ti - mean[0]) / sigma[0]) * ((tj - mean[1]) / sigma[1]) * fai
            ep = mmt / math.exp(-yi * yi) / math.exp(-yj * yj)
            return ep

        # =================================================
        stdNorm_rhoset = dict()

        x = sympy.Symbol('x')
        rank = 32
        root, weight = numpy.polynomial.hermite.hermgauss(rank)

        for key_pair in rho_parameter:
            rho = rho_parameter[key_pair]
            # print(key_pair, rho)

            if rho != 0.0:
                key = key_pair.split('_')
                distr_pair = [self.SorLd_margin[key[0]], self.SorLd_margin[key[1]]]
                amean_pair = [self.SorLd_amean[key[0]], self.SorLd_amean[key[1]]]
                sigma_pair = [self.SorLd_sigma[key[0]], self.SorLd_sigma[key[1]]]

                GauSum = 0.0
                for i in range(len(root)):
                    for j in range(len(root)):
                        GauSum += rhoconvert(distr_pair, amean_pair, sigma_pair, root[i], root[j], randseed=123) * weight[i] * weight[j]

                eq = GauSum - rho
                norm_rho = sympy.re(sympy.nsolve(eq, x, rho, tol=1e-4))
                # print('norm_rho: ', norm_rho)
            else:
                norm_rho = 0.0

            stdNorm_rhoset[key_pair] = norm_rho
        # print(stdNorm_rhoset)

        # =================================================
        stdNorm_muivet = numpy.zeros((self.dimen, 1))
        stdNorm_covmat = numpy.identity(self.dimen)
        stdNorm_chosky = numpy.identity(self.dimen)

        for key_pair in stdNorm_rhoset:
            key = key_pair.split('_')

            if key[0] == key[1]:
                n = Nrvs[key[0]]
                zone_key = zone[key[0]]
                patMat = numpy.identity(n)

                if not zone_key:
                    up = numpy.triu_indices(n, 1)
                    lw = numpy.tril_indices(n, -1)
                    patMat[up] = stdNorm_rhoset[key_pair]
                    patMat[lw] = stdNorm_rhoset[key_pair]
                else:
                    cnt = 0
                    for k in range(len(zone_key)):
                        nfddg = len(zone_key[k])
                        patMat_fd = numpy.identity(nfddg)
                        up = numpy.triu_indices(nfddg, 1)
                        lw = numpy.tril_indices(nfddg, -1)
                        patMat_fd[up] = stdNorm_rhoset[key_pair]
                        patMat_fd[lw] = stdNorm_rhoset[key_pair]

                        patMat[numpy.ix_(numpy.arange(cnt, cnt + nfddg), numpy.arange(cnt, cnt + nfddg))] = patMat_fd
                        cnt += nfddg
                stdNorm_covmat[numpy.ix_(Sprvs[key[0]], Sprvs[key[1]])] = patMat
            else:
                patMat = numpy.full((Nrvs[key[0]], Nrvs[key[1]]), stdNorm_rhoset[key_pair])

                stdNorm_covmat[numpy.ix_(Sprvs[key[0]], Sprvs[key[1]])] = patMat
                stdNorm_covmat[numpy.ix_(Sprvs[key[1]], Sprvs[key[0]])] = patMat.transpose()

        stdNorm_chosky = numpy.linalg.cholesky(stdNorm_covmat)

        self.SorLd_stdNormRhoSet = stdNorm_rhoset
        self.SorLd_stdNormMuiVet = stdNorm_muivet
        self.SorLd_stdNormCovMat = stdNorm_covmat
        self.SorLd_stdNormChoSky = stdNorm_chosky

    def packMaterial(self):
        material = dict()

        material['SorLd_margin'] = self.SorLd_margin
        material['SorLd_multi'] = self.SorLd_multi

        material['SorLd_stdNormMuiVet'] = self.SorLd_stdNormMuiVet
        material['SorLd_stdNormCovMat'] = self.SorLd_stdNormCovMat
        material['SorLd_stdNormChoSky'] = self.SorLd_stdNormChoSky

        return material




