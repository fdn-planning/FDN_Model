import numpy
import chaospy

'''
Generate independent or correlated samples
Perform uncertainty propagation based on Monte Carlo simulation

'''


class baseUncertain(object):

    def __init__(self, material, pi=1, dimen=1, nstate=1):
        self.pi = pi
        self.dimen = dimen
        self.nstate = nstate

        self.SorLd_margin = material['SorLd_margin']
        self.SorLd_multi = material['SorLd_multi']

        self.SorLd_stdNormMuiVet = material['SorLd_stdNormMuiVet']
        self.SorLd_stdNormCovMat = material['SorLd_stdNormCovMat']
        self.SorLd_stdNormChoSky = material['SorLd_stdNormChoSky']

    def calculateOLS(self, x, y, method='svd'):
        if method == 'direct':
            m = numpy.linalg.inv(numpy.dot(numpy.transpose(x), x))
            beta = numpy.dot(numpy.dot(m, numpy.transpose(x)), y)
        elif method == 'svd':
            beta = numpy.linalg.lstsq(x, y, rcond=None)[0]
        else:
            beta = numpy.zeros(self.dimen)
            print('illegal method for calculate OLS')
            exit(1)

        return beta

    def constructPloynomial(self, Nrvs):
        normsign = True
        self.ploynomial = list()

        for key in Nrvs:
            for i in range(Nrvs[key]):
                uni_ploynomial = chaospy.generate_expansion(self.pi, self.SorLd_margin[key], normed=normsign)
                self.ploynomial.append(uni_ploynomial)

    def selectDistribution(self, t, Sprvs):
        distrX = []
        for key in Sprvs:
            if t in Sprvs[key]:
                distrX = self.SorLd_margin[key]
        return distrX

    def generateSample(self, randseed, Sprvs, ndata):
        RandVar = numpy.zeros((self.dimen, ndata))
        for key in Sprvs:
            if self.SorLd_multi[key]:
                numpy.random.seed(randseed)
                RandVar[Sprvs[key]] = self.SorLd_multi[key].sample(ndata)

        return RandVar

    def generateStdNormalSample(self, randseed, ndata, method='sobol'):
        numpy.random.seed(randseed)
        udistr = chaospy.Iid(chaospy.Uniform(0, 1), self.dimen)
        usamp = udistr.sample(size=ndata, rule=method)

        ufirst = usamp[:self.dimen]
        xid = chaospy.Normal(mu=0.0, sigma=1.0).ppf(ufirst)

        return xid

    def genetateKsaiSample(self, randseed, xid, Sprvs, ndata):
        x = numpy.dot(self.SorLd_stdNormChoSky, xid) + self.SorLd_stdNormMuiVet

        ksai_x = numpy.zeros((self.dimen, ndata))
        for i in range(self.dimen):
            numpy.random.seed(randseed)
            distrX = self.selectDistribution(i, Sprvs)
            ksai_x[i] = distrX.ppf(chaospy.Normal(mu=0.0, sigma=1.0).cdf(x[i]))

        return ksai_x


class MCSpprox(baseUncertain):
    def __init__(self, material, pi=1, dimen=1, nstate=1):
        super(MCSpprox, self).__init__(material, pi, dimen, nstate)
