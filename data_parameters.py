'''
Set parameters for distribution network planning

'''

from data_construction import Cbase, Sbase, Vbase, Ibase, Zbase, Vstation, \
    nnode, inode, vnode, znode, mnode, islck, nslck, ipnode, \
    nline, iline, bgbus, edbus, rline, xline, sline, abond, ibond, nbond, zbond, \
    znode_cnt, znode_key, znode_val, zline_cnt, zline_key, zline_val, nzone, \
    nties, ities, bgtiebus, edtiebus, \
    nxyts, ixyts, cxyts, xteml, yteml, mteml, \
    npall, ipall, zpall, lpall, mpall, \
    nload, iload, cload, pload, qload, tload, \
    nevcs, ievcs, cevcs, sevcs, \
    ndgen, idgen, cdgen, sdgen, fdgen, tdgen, \
    nstog, istog, cstog,  \
    casedata

# ======================================================================================================================
nstage = 4  # number of stages
nyear = 5   # years of a stage
nhour = 8760  # hours of a year
infla = 0.1  # inflation rate
epsilon = (pow(1 + infla, nyear) - 1) / (infla * pow(1 + infla, nyear))

# ======================================================================================================================
smini_cvt, smaxi_cvt = [1e-6, 10]  # minimum and maximum values of converter capacity (p.u.)
smini_bat, smaxi_bat = [1e-6, 30]  # minimum and maximum values of battery (p.u.)
floss_cvt = 0.02  # loss factor of converter
scint_bat = 0.5  # initial SOC of battery
scmin_bat = 0.1  # minimum SOC of battery
scmax_bat = 0.9  # maximum SOC of battery
decml_cap = 2  # decimal digit

pcu_max = [0.00, 0.00, 0.00, 0.00]  # curtailment rate of distributed generator
pty_per = {'Sola': 1.00, 'Wind': 0.00}  # proportion of distributed generator

load_increas = [1.02, 1.015,  1.01, 1.005]  # average annual growth rate of conventional load
dgen_penetrn = [0.15,  0.30,  0.50,  0.65]  # average annual growth rate of distributed generator
evel_penetrn = [0.05,  0.15,  0.20,  0.24]  # average annual growth rate of electric vehicle charging load

Nstog_max = [nstog] * nstage  # maximum number of energy storage systems
Ndgen_max = [ndgen] * nstage  # maximum number of distributed generators
Nevcs_max = [nevcs] * nstage  # maximum number of electric vehicle charging stations

price_bat = [i * 1e3 for i in [1.0,  0.8,  0.5, 0.3]]  # unit capacity cost of battery in each stage (yuan/kWh)
price_trs = [i * 1e2 for i in [5,      4,  3.5,   3]]  # unit capacity cost of transformer in each stage (yuan/kVA)
price_cvt = [i * 1e3 for i in [0.8,  0.6,  0.4, 0.2]]  # unit capacity cost of converter in each stage (yuan/kVA)
price_sit = [i * 1e6 for i in [3,    3.5,    4,   5]]  # land exploitation cost of SOP in each stage (yuan)
price_ref = [i * 1e5 for i in [1,    1.2,  1.6,   2]]  # line construction cost in each stage (yuan/km)
price_ele = [i * 1e0 for i in [0.35, 0.3, 0.25, 0.2]]  # electricity price in each stage (yuan/kWh)

price_exp = [i * 3e0 for i in price_ref]  # line expansion cost in each stage (yuan/km)
Rxper_exp = 0.5  # factor of resistance and reactance after line expansion
Ctper_exp = 1.5  # factor of capacity after line expansion

price_dga = [i * 1e1 for i in price_ele]  # Not Adapted (yuan/kWh)
price_lda = [i * 1e1 for i in price_ele]  # Not Adapted (yuan/kWh)
price_eva = [i * 1e1 for i in price_ele]  # Not Adapted (yuan/kWh)
price_vas = 100  # Not Adapted
price_bas = 100  # Not Adapted

# ======================================================================================================================
modfy_geo = 1.0  # terrain correction factor for line construction
Irate = 400 / Ibase  # rated capacity of line (p.u.)

Vminlit_lad = [0.95]  # lower limit of node voltage (p.u.)
Vmaxlit_lad = [1.05]  # upper limit of node voltage (p.u.)
Imaxlit_lad = [i * Irate for i in [1.0]]  # upper limit of line current (p.u.)

Vrisk = [0.05]  # allowable risk of voltage violation
Irisk = [0.05]  # allowable risk of current violation
Crisk = [0.05]  # Not Adapted
volt_nlad = len(Vrisk)
line_nlad = len(Irisk)
evcs_nlad = len(Crisk)

if volt_nlad != len(Vminlit_lad):
    print('No corresponding limit ladder of Vmin')
    exit(0)
if volt_nlad != len(Vmaxlit_lad):
    print('No corresponding limit ladder of Vmax')
    exit(0)
if line_nlad != len(Imaxlit_lad):
    print('No corresponding limit ladder of Imax')
    exit(0)

Vminlit_sec = Vminlit_lad[-1]  # lower limit of node voltage for secure operations
Vmaxlit_sec = Vmaxlit_lad[-1]  # upper limit of node voltage for secure operations
Imaxlit_sec = Imaxlit_lad[-1]  # upper limit of line current for secure operations

Vminlit_vas = Vminlit_lad[-1]  # lower limit of node voltage for optimal operations
Vmaxlit_vas = Vmaxlit_lad[-1]  # upper limit of node voltage for optimal operations
Imaxlit_bas = Imaxlit_lad[-1]  # upper limit of line current for optimal operations

# ======================================================================================================================
nsamp = 1000  # Sample size of experiment design
ntest = 20000  # Sample size of experiment test
nappl = 20000  # Sample size of application test
nstate = nnode + nline  # number of states

Rstate = [1, 1, 1, 1]  # initialization of rank
lra_deltermin = [1e-3, 1e-3, 1e-3, 1e-3]  # initialization of stop criterion
lra_pi = [1, 1, 1, 1]  # initialization of polynomial order
lra_itermax = [3, 3, 3, 3]  # initialization of maximum iterations