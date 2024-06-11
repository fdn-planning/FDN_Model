import numpy
import pandas
import picos
import time

'''
Establish the flexible distribution network planning model

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
    nstog, istog, cstog, \
    casedata

from data_parameters import Vminlit_sec, Vmaxlit_sec, Imaxlit_sec, Vminlit_vas, Vmaxlit_vas, Imaxlit_bas, \
    smini_cvt, smaxi_cvt, floss_cvt, decml_cap, modfy_geo, Irate, \
    price_cvt, price_sit, price_ref, price_trs, price_ele, price_dga, price_lda, price_eva, price_vas, price_bas, \
    price_exp, Rxper_exp, Ctper_exp, \
    nstage, nyear, nhour, infla, epsilon, \
    load_increas, dgen_penetrn, evel_penetrn, pcu_max, pty_per, Nevcs_max, Ndgen_max, \
    Vrisk, Irisk, volt_nlad, line_nlad, Vminlit_lad, Vmaxlit_lad, Imaxlit_lad

from data_preprocess import pload_stg, qload_stg, pevel_stg,pdgen_stg, \
    pload_aux, qload_aux, pload_init_aux, qload_init_aux, load_pvt, \
    conctmax, connect_id, connect_sr, \
    iport, nport, mport, scheme, nschm, nscht, nschp, lschm, scheme_evolve, scheme_contain, \
    nliNm, lwsNm, dimen, schNm

from ChancePlanCord_distrgen import WindSpeedtoPower, SolaradiatoPower, limitGeneratorOutput
from ChancePlanCord_auxiliary import ceilDecimal


def computeOptimizationPlanningEngine(ExptVar, LimtVal, InitSign=False, BavaSign=False, Solver='gurobi'):
    # ExptVar-mean value of random variable
    # LimtVal-status limits of flexible distribution network
    # InitSign-whether to perform initializated power flow calculation
    # BavaSign-whether the objective function includes load balancing and voltage deviation
    # Solver-optimization solver

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    print('Planning model: Define parameters')
    start = time.time()

    # //////////////////////////////////////////////////////////////
    Vminlit = LimtVal['Vminlit']  # (nstage, nnode)
    Vmaxlit = LimtVal['Vmaxlit']  # (nstage, nnode)
    Imaxlit = LimtVal['Imaxlit']  # (nstage, nline)

    QdgcrSign = False
    vci, vco, vn = [2.0, 6.0, 4.0]
    refactor, squarekw = [0.15, 0.10]

    Windout_expt = WindSpeedtoPower(ExptVar['Wind'], vci, vco, vn)
    Solaout_expt = SolaradiatoPower(ExptVar['Sola'], refactor, squarekw)

    Windout_limt = limitGeneratorOutput(Windout_expt)
    Solaout_limt = limitGeneratorOutput(Solaout_expt)

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    Picos = picos.Problem()

    end = time.time()
    print('Time= {}'.format(end-start))

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    print('Planning model: Define varaibles')
    start = time.time()

    # //////////////////////////////////////////////////////////////
    # continuous-static variable Number
    if BavaSign:
        # ->Pij Qij Iij
        NXcon1 = nline * 3  # ->Ui
        NXcon2 = NXcon1 + nnode * 1  # ->UIa UIb
        NXcon3 = NXcon2 + nline * 2  # ->Pba Qba
        NXcon4 = NXcon3 + nslck * 2  # ->PSopt QSopt MSopt SSopt
        NXcon5 = NXcon4 + nschp * 4  # ->SSopk
        NXcon6 = NXcon5 + nschm * 1  # ->Pevel Sevcs
        NXcon7 = NXcon6 + nevcs * 2  # ->Pdgen Qdgen Sdgen
        NXcon8 = NXcon7 + ndgen * 3  # ->Aij
        NXcon9 = NXcon8 + nline * 1  # ->Bi
        NXcon = NXcon9 + nnode * 1
    else:
        # ->Pij Qij Iij
        NXcon1 = nline * 3  # ->Ui
        NXcon2 = NXcon1 + nnode * 1  # ->UIa UIb
        NXcon3 = NXcon2 + nline * 2  # ->Pba Qba
        NXcon4 = NXcon3 + nslck * 2  # ->PSopt QSopt MSopt SSopt
        NXcon5 = NXcon4 + nschp * 4  # ->SSopk
        NXcon6 = NXcon5 + nschm * 1  # ->Pevel Sevcs
        NXcon7 = NXcon6 + nevcs * 2  # ->Pdgen Qdgen Sdgen
        NXcon8 = 0
        NXcon9 = 0
        NXcon = NXcon7 + ndgen * 3
 
    # //////////////////////////////////////////////////////////////
    # integer variable Number
    # ->alpha
    NXbin1 = nschm  # ->beta
    NXbin2 = NXbin1 + nevcs  # ->delta
    NXbin = NXbin2 + ndgen
    
    NXconsum = nstage * NXcon
    NXbinsum = nstage * NXbin

    # //////////////////////////////////////////////////////////////
    xcon_lower = numpy.zeros(NXconsum)
    xcon_upper = numpy.zeros(NXconsum)
    for p in range(nstage):
        idx = p * NXcon
        for i in range(nline):  # Pij
            xcon_lower[idx + i] = -numpy.inf
            xcon_upper[idx + i] = +numpy.inf
        idx = p * NXcon + nline
        for i in range(nline):  # Qij
            xcon_lower[idx + i] = -numpy.inf
            xcon_upper[idx + i] = +numpy.inf
        idx = p * NXcon + nline * 2
        for i in range(nline):  # Iij
            if InitSign:
                xcon_lower[idx + i] = 0.0
                xcon_upper[idx + i] = +numpy.inf
            else:
                xcon_lower[idx + i] = 0.0
                xcon_upper[idx + i] = Imaxlit[p][i] ** 2

        idx = p * NXcon + NXcon1
        for i in range(nnode):  # Ui
            if inode[i] in islck:
                xcon_lower[idx + i] = Vstation ** 2
                xcon_upper[idx + i] = Vstation ** 2
            else:
                if InitSign:
                    xcon_lower[idx + i] = 0.0
                    xcon_upper[idx + i] = +numpy.inf
                else:
                    xcon_lower[idx + i] = Vminlit[p][i] ** 2
                    xcon_upper[idx + i] = Vmaxlit[p][i] ** 2

        idx = p * NXcon + NXcon2
        for i in range(nline):  # UIa
            xcon_lower[idx + i] = -numpy.inf
            xcon_upper[idx + i] = +numpy.inf
        idx = p * NXcon + NXcon2 + nline
        for i in range(nline):  # UIb
            xcon_lower[idx + i] = -numpy.inf
            xcon_upper[idx + i] = +numpy.inf

        idx = p * NXcon + NXcon3
        for i in range(nslck):  # Pba
            xcon_lower[idx + i] = -numpy.inf
            xcon_upper[idx + i] = +numpy.inf
        idx = p * NXcon + NXcon3 + nslck
        for i in range(nslck):  # Qba
            xcon_lower[idx + i] = -numpy.inf
            xcon_upper[idx + i] = +numpy.inf

        if not InitSign:
            idx = p * NXcon + NXcon4
            for i in range(nschp):  # Psopt
                xcon_lower[idx + i] = -numpy.inf
                xcon_upper[idx + i] = +numpy.inf
            idx = p * NXcon + NXcon4 + nschp
            for i in range(nschp):  # Qsopt
                xcon_lower[idx + i] = -numpy.inf
                xcon_upper[idx + i] = +numpy.inf
            idx = p * NXcon + NXcon4 + nschp * 2
            for i in range(nschp):  # Msopt
                xcon_lower[idx + i] = 0.0
                xcon_upper[idx + i] = +numpy.inf
            idx = p * NXcon + NXcon4 + nschp * 3
            for i in range(nschp):  # Ssopt
                xcon_lower[idx + i] = 0.0
                xcon_upper[idx + i] = +numpy.inf

        if not InitSign:
            idx = p * NXcon + NXcon5
            for i in range(nschm):  # Ssopk
                xcon_lower[idx + i] = 0.0
                xcon_upper[idx + i] = +numpy.inf

        if not InitSign:
            idx = p * NXcon + NXcon6
            for i in range(nevcs):  # Pevel
                xcon_lower[idx + i] = 0.0
                xcon_upper[idx + i] = +numpy.inf

            idx = p * NXcon + NXcon6 + nevcs
            for i in range(nevcs):  # Sevcs
                xcon_lower[idx + i] = 0.0
                xcon_upper[idx + i] = +numpy.inf

        if not InitSign:
            idx = p * NXcon + NXcon7
            for i in range(ndgen):  # Pdgen
                xcon_lower[idx + i] = 0.0
                xcon_upper[idx + i] = +numpy.inf

            idx = p * NXcon + NXcon7 + ndgen
            for i in range(ndgen):  # Qdgen
                xcon_lower[idx + i] = 0.0
                xcon_upper[idx + i] = +numpy.inf

            idx = p * NXcon + NXcon7 + ndgen * 2
            for i in range(ndgen):  # Sdgen
                xcon_lower[idx + i] = 0.0
                xcon_upper[idx + i] = +numpy.inf

        if BavaSign:
            idx = p * NXcon + NXcon8
            for i in range(nline):  # Aij
                xcon_lower[idx + i] = 0.0
                xcon_upper[idx + i] = +numpy.inf

            idx = p * NXcon + NXcon9
            for i in range(nnode):  # Bi
                xcon_lower[idx + i] = 0.0
                xcon_upper[idx + i] = +numpy.inf

    xbin_lower = numpy.zeros(NXbinsum)
    xbin_upper = numpy.zeros(NXbinsum)
    if not InitSign:
        for p in range(nstage):
            idx = p * NXbin
            for i in range(nschm):  # alpha
                xbin_lower[idx + i] = 0
                xbin_upper[idx + i] = 1

            idx = p * NXbin + NXbin1
            for i in range(nevcs):  # beta
                xbin_lower[idx + i] = 0
                xbin_upper[idx + i] = 1

            idx = p * NXbin + NXbin2
            for i in range(ndgen):  # delta
                xbin_lower[idx + i] = 0
                xbin_upper[idx + i] = 1

    xcon = picos.RealVariable('xcon', NXconsum, lower=xcon_lower, upper=xcon_upper)
    xbin = picos.IntegerVariable('xbin', NXbinsum, lower=xbin_lower, upper=xbin_upper)

    end = time.time()
    print('Time= {}'.format(end-start))

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    print('Planning model: Define objectices')
    start = time.time()

    ObjectCo_sop = numpy.zeros(NXconsum)
    ObjectCo_sit = numpy.zeros(NXbinsum)
    ObjectCo_ref = numpy.zeros(NXbinsum)
    ObjectCo_evs = numpy.zeros(NXconsum)
    ObjectCo_sev = numpy.zeros(NXbinsum)
    ObjectCo_dgc = numpy.zeros(NXconsum)

    ObjectOp_sls = numpy.zeros(NXconsum)
    ObjectOp_bas = numpy.zeros(NXconsum)
    ObjectOp_vas = numpy.zeros(NXconsum)

    # //////////////////////////////////////////////////////////////
    # Investment cost
    for p in range(nstage):
        zeta = pow(1 + infla, -p * nyear)

        if p == 0:
            ObjectCo_sop[NXcon5: NXcon6] += zeta * price_cvt[p] * Sbase
            ObjectCo_sit[0: NXbin1] += zeta * price_sit[p]
            ObjectCo_ref[0: NXbin1] += zeta * modfy_geo * price_ref[p] * lschm

            ObjectCo_evs[NXcon6 + nevcs: NXcon7] += zeta * price_cvt[p] * Sbase
            ObjectCo_sev[NXbin1: NXbin2] += zeta * price_sit[p]

            ObjectCo_dgc[NXcon7 + ndgen * 2: NXcon] += zeta * price_cvt[p] * Sbase
        else:
            # //////////////////////////////////////////////////////////////
            # ObjectCo_sop
            interval_con_pb = range((p - 1) * NXcon + NXcon5, (p - 1) * NXcon + NXcon6)  # SSopk
            interval_con_pp = range(p * NXcon + NXcon5, p * NXcon + NXcon6)

            ObjectCo_sop[interval_con_pb] -= zeta * price_cvt[p] * Sbase
            ObjectCo_sop[interval_con_pp] += zeta * price_cvt[p] * Sbase

            # ObjectCo_sit ObjectCo_ref
            interval_bin_pb = range((p - 1) * NXbin, (p - 1) * NXbin + NXbin1)  # alpha
            interval_bin_pp = range(p * NXbin, p * NXbin + NXbin1)

            ObjectCo_sit[interval_bin_pb] -= zeta * price_sit[p]
            ObjectCo_sit[interval_bin_pp] += zeta * price_sit[p]

            ObjectCo_ref[interval_bin_pb] -= zeta * modfy_geo * price_ref[p] * lschm
            ObjectCo_ref[interval_bin_pp] += zeta * modfy_geo * price_ref[p] * lschm

            # //////////////////////////////////////////////////////////////
            # ObjectCo_evs
            interval_con_pb = range((p - 1) * NXcon + NXcon6 + nevcs, (p - 1) * NXcon + NXcon7)  # Sevcs
            interval_con_pp = range(p * NXcon + NXcon6 + nevcs, p * NXcon + NXcon7)

            ObjectCo_evs[interval_con_pb] -= zeta * price_cvt[p] * Sbase
            ObjectCo_evs[interval_con_pp] += zeta * price_cvt[p] * Sbase

            # ObjectCo_sev
            interval_bin_pb = range((p - 1) * NXbin + NXbin1, (p - 1) * NXbin + NXbin2)  # beta
            interval_bin_pp = range(p * NXbin + NXbin1, p * NXbin + NXbin2)

            ObjectCo_sev[interval_bin_pb] -= zeta * price_sit[p]
            ObjectCo_sev[interval_bin_pp] += zeta * price_sit[p]

            # //////////////////////////////////////////////////////////////
            # ObjectCo_dgc
            interval_con_pb = range((p - 1) * NXcon + NXcon7 + ndgen * 2, p * NXcon)  # Sdgen
            interval_con_pp = range(p * NXcon + NXcon7 + ndgen * 2, (p + 1) * NXcon)

            ObjectCo_dgc[interval_con_pb] -= zeta * price_cvt[p] * Sbase
            ObjectCo_dgc[interval_con_pp] += zeta * price_cvt[p] * Sbase

    # printCheckVector(ObjectCo_sop, NXconsum, 'ObjectCo_sop')
    # printCheckVector(ObjectCo_sit, NXbinsum, 'ObjectCo_sit')
    # printCheckVector(ObjectCo_ref, NXbinsum, 'ObjectCo_ref')
    # printCheckVector(ObjectCo_evs, NXconsum, 'ObjectCo_evs')
    # printCheckVector(ObjectCo_sev, NXbinsum, 'ObjectCo_sev')
    # printCheckVector(ObjectCo_dgc, NXconsum, 'ObjectCo_dgc')
    # exit()

    # //////////////////////////////////////////////////////////////
    # Operational cost
    for p in range(nstage):
        zeta = pow(1 + infla, -p * nyear)
        zeta_hours = zeta * nhour * epsilon

        for i in range(nline):
            ObjectOp_sls[p * NXcon + nline * 2 + i] = zeta_hours * price_ele[p] * rline[i] * Sbase  # Rij *Iij
        for i in range(nschp):
            ObjectOp_sls[p * NXcon + NXcon4 + nschp * 2 + i] = zeta_hours * price_ele[p] * Sbase  # MSopt

        if BavaSign:
            for i in range(nline):
                ObjectOp_bas[NXcon8 + i] = zeta_hours * price_bas  # Aij
        if BavaSign:
            for i in range(nnode):
                ObjectOp_vas[NXcon9 + i] = zeta_hours * price_vas  # Bi

    # printCheckVector(ObjectOp_sls, NXconsum, 'ObjectOp_sls')
    # printCheckVector(ObjectOp_bas, NXconsum, 'ObjectOp_bas')
    # printCheckVector(ObjectOp_vas, NXconsum, 'ObjectOp_vas')
    # exit()

    CbaseCal = 1e6
    if InitSign:
        Picos.minimize = ObjectOp_sls / CbaseCal * xcon
    else:
        Picos.minimize = (ObjectCo_sop + ObjectCo_evs + ObjectCo_dgc + ObjectOp_sls + ObjectOp_bas + ObjectOp_vas) / CbaseCal * xcon + \
                         (ObjectCo_sit + ObjectCo_ref + ObjectCo_sev) / CbaseCal * xbin

    end = time.time()
    print('Time= {}'.format(end-start))

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    print('Planning model: Construct constraint matrices')
    start = time.time()

    # //////////////////////////////////////////////////////////////
    # Pineq Qineq
    Pineq = numpy.zeros((nnode, NXcon))
    Qineq = numpy.zeros((nnode, NXcon))
    for i in range(nnode):
        for j in range(int(conctmax)):
            if connect_id[i][j] < 0:
                k = connect_sr[i][j]
                Pineq[i][k] = 1.0
                Pineq[i][nline * 2 + k] = -rline[k]
                Qineq[i][nline * 1 + k] = 1.0
                Qineq[i][nline * 2 + k] = -xline[k]

            if connect_id[i][j] > 0:
                k = connect_sr[i][j]
                Pineq[i][k] = -1.0
                Qineq[i][nline * 1 + k] = -1.0

        for j in range(nslck):
            if inode[i] == islck[j]:
                Pineq[i][NXcon3 + j] = 1.0
                Qineq[i][NXcon3 + nslck + j] = 1.0

        for j, schm in enumerate(scheme):
            for k, schm_node in enumerate(schm):
                if inode[i] == schm_node:
                    idx = sum(nscht[:j]) + k
                    Pineq[i][NXcon4 + idx] = 1.0  # PSopt
                    Qineq[i][NXcon4 + nschp + idx] = 1.0  # QSopt

        for j in range(nevcs):
            if inode[i] == cevcs[j]:
                Pineq[i][NXcon6 + j] = -ExptVar['Evel']  # -E[*]*Pevel

        for j in range(ndgen):
            if inode[i] == cdgen[j]:
                if tdgen[j] == 1:
                    Pineq[i][NXcon7 + j] = Solaout_limt          # E[*]*Pdgen
                    Qineq[i][NXcon7 + ndgen + j] = Solaout_limt  # E[*]*Qdgen
                if tdgen[j] == 2:
                    Pineq[i][NXcon7 + j] = Windout_limt          # E[*]*Pdgen
                    Qineq[i][NXcon7 + ndgen + j] = Windout_limt  # E[*]*Qdgen

    # Uineq
    Uineq = numpy.zeros((nline, NXcon))
    for i in range(nline):
        Uineq[i][i] = -2 * rline[i]
        Uineq[i][nline * 1 + i] = -2 * xline[i]
        Uineq[i][nline * 2 + i] = numpy.square(rline[i]) + numpy.square(xline[i])

        for j in range(nnode):
            if bgbus[i] == inode[j]:
                Uineq[i][NXcon1 + j] = 1.0
            if edbus[i] == inode[j]:
                Uineq[i][NXcon1 + j] = -1.0

    # Uiaeq Uibeq
    Uiaeq = numpy.zeros((nline, NXcon))
    Uibeq = numpy.zeros((nline, NXcon))
    for i in range(nline):
        Uiaeq[i][nline * 2 + i] = 0.5
        Uibeq[i][nline * 2 + i] = 0.5

        for j in range(nnode):
            if bgbus[i] == inode[j]:
                Uiaeq[i][NXcon1 + j] = -0.5
                Uibeq[i][NXcon1 + j] = 0.5

        Uiaeq[i][NXcon2 + i] = -1.0  # UIa
        Uibeq[i][NXcon2 + nline + i] = -1.0  # UIb

    # Pisop
    Pisop = numpy.zeros((nschm, NXcon))
    for i, schm in enumerate(scheme):
        for j, schm_node in enumerate(schm):
            idx = sum(nscht[:i]) + j
            Pisop[i][NXcon4 + idx] = 1.0  # PSopt
            Pisop[i][NXcon4 + nschp * 2 + idx] = -1.0  # MSopt

    # Alzeq
    Alzeq = numpy.zeros((nline, NXcon))
    if BavaSign:
        for i in range(nline):
            Alzeq[i][NXcon8 + i] = 1.0  # Aij
            Alzeq[i][nline * 2 + i] = -1.0  # -Iij

    # Blzeq Blseq
    Blzeq = numpy.zeros((nnode, NXcon))
    Blseq = numpy.zeros((nnode, NXcon))
    if BavaSign:
        for i in range(nnode):
            Blzeq[i][NXcon9 + i] = 1.0  # Bi
            Blzeq[i][NXcon1 + i] = -1.0  # -Ui

            Blseq[i][NXcon9 + i] = 1.0  # Bi
            Blseq[i][NXcon1 + i] = 1.0  # Ui

    # printCheckMatrix(Pineq, nnode, NXcon, 'Pineq')
    # printCheckMatrix(Qineq, nnode, NXcon, 'Qineq')
    # printCheckMatrix(Uineq, nline, NXcon, 'Uineq')
    # printCheckMatrix(Uiaeq, nline, NXcon, 'Uiaeq')
    # printCheckMatrix(Uibeq, nline, NXcon, 'Uibeq')
    # printCheckMatrix(Pisop, nschm, NXcon, 'Pisop')
    # printCheckMatrix(Alzeq, nline, NXcon, 'Alzeq')
    # printCheckMatrix(Blzeq, nnode, NXcon, 'Blzeq')
    # printCheckMatrix(Blseq, nnode, NXcon, 'Blseq')
    # exit()

    # //////////////////////////////////////////////////////////////
    # Sadeq
    Sadeq = numpy.zeros((nschm, NXcon))
    for i, schm in enumerate(scheme):
        Sadeq[i][NXcon5 + i] = 1.0  # SSopk

        for j, schm_node in enumerate(schm):
            idx = sum(nscht[:i]) + j
            Sadeq[i][NXcon4 + nschp * 3 + idx] = -1.0  # SSopt

    # Slseq Slxeq
    Slseq = numpy.zeros((nschm, NXcon))
    Slxeq = numpy.zeros((nschm, NXcon))
    for i in range(nschm):
        Slseq[i][NXcon5 + i] = 1.0  # SSopk
        Slxeq[i][NXcon5 + i] = -1.0  # -SSopk

    # Slseqplus Slxeqplus
    Slseqplus = numpy.zeros((nschm, NXbin))
    Slxeqplus = numpy.zeros((nschm, NXbin))
    for i in range(nschm):
        Slseqplus[i][i] = -smaxi_cvt
        Slxeqplus[i][i] = smini_cvt

    # printCheckMatrix(Sadeq, nschm, NXcon, 'Sadeq')
    # printCheckMatrix(Slseq, nschm, NXcon, 'Slseq')
    # printCheckMatrix(Slxeq, nschm, NXcon, 'Slxeq')
    # printCheckMatrix(Slseqplus, nschm, NXbin, 'Slseqplus')
    # printCheckMatrix(Slxeqplus, nschm, NXbin, 'Slxeqplus')
    # exit()

    # //////////////////////////////////////////////////////////////
    # Spteq
    Spteq = numpy.zeros((nport * (nstage - 1), NXconsum))
    for p in range(1, nstage):
        for i, port_node in enumerate(iport):
            schm_idx = scheme_contain[i]

            for j in schm_idx:
                schm = scheme[j]
                k = schm.index(port_node)
                idx = sum(nscht[:j]) + k

                Spteq[(p - 1) * nport + i][(p - 1) * NXcon + NXcon4 + nschp * 3 + idx] = 1.0  # Ssopt_(p-1)
                Spteq[(p - 1) * nport + i][p * NXcon + NXcon4 + nschp * 3 + idx] = -1.0  # Ssopt_(p)

    # printCheckMatrix(Spteq, nport * (nstage - 1), NXconsum, 'Spteq')
    # exit()

    # //////////////////////////////////////////////////////////////
    # Kuneq
    Kuneq = numpy.zeros((nport, NXbin))
    for i, port_node in enumerate(iport):
        Kuneq[i][scheme_contain[i]] = 1.0

    # Kcoeq
    Kcoeq = numpy.zeros((nschm * (nstage - 1), NXbinsum))
    for p in range(1, nstage):
        for i in range(nschm):
            Kcoeq[(p - 1) * nschm + i][(p - 1) * NXbin + i] = 1.0  # alpha_(p-1)
            Kcoeq[(p - 1) * nschm + i][p * NXbin + scheme_evolve[i]] = -1.0  # alpha_(p)

    # printCheckMatrix(Kuneq, nport, NXbin, 'Kuneq')
    # printCheckMatrix(Kcoeq, nschm * (nstage - 1), NXbinsum, 'Kcoeq')
    # exit()

    # //////////////////////////////////////////////////////////////
    # Ccoeq
    Ccoeq = numpy.zeros((nevcs * (nstage - 1), NXbinsum))
    for p in range(1, nstage):
        for i in range(nevcs):
            Ccoeq[(p - 1) * nevcs + i][(p - 1) * NXbin + NXbin1 + i] = 1.0  # beta_(p-1)
            Ccoeq[(p - 1) * nevcs + i][p * NXbin + NXbin1 + i] = -1.0  # beta_(p)

    # Cpteq
    Cpteq = numpy.zeros((nevcs * (nstage - 1), NXconsum))
    for p in range(1, nstage):
        for i in range(nevcs):
            Cpteq[(p - 1) * nevcs + i][(p - 1) * NXcon + NXcon6 + nevcs + i] = 1.0  # Sevcs_(p-1)
            Cpteq[(p - 1) * nevcs + i][p * NXcon + NXcon6 + nevcs + i] = -1.0  # Sevcs_(p)

    # Cnueq
    Cnueq = numpy.zeros((nstage, NXbin))
    for p in range(nstage):
        for i in range(nevcs):
            Cnueq[p][NXbin1 + i] = 1.0  # beta

    # Clseq Clxeq
    Clseq = numpy.zeros((nevcs, NXcon))
    Clxeq = numpy.zeros((nevcs, NXcon))
    for i in range(nevcs):
        Clseq[i][NXcon6 + nevcs + i] = 1.0  # Sevcs
        Clxeq[i][NXcon6 + nevcs + i] = -1.0  # -Sevcs

    # Clseqplus Clxeqplus
    Clseqplus = numpy.zeros((nevcs, NXbin))
    Clxeqplus = numpy.zeros((nevcs, NXbin))
    for i in range(nevcs):
        Clseqplus[i][NXbin1 + i] = -sevcs[i]  # beta
        Clxeqplus[i][NXbin1 + i] = smini_cvt  # beta

    # Ceveq
    Ceveq = numpy.zeros((nevcs, NXcon))
    for i in range(nevcs):
        Ceveq[i][NXcon6 + i] = 1.0  # Pevel
        Ceveq[i][NXcon6 + nevcs + i] = -1.0  # -Sevcs

    # Csmeq
    Csmeq = numpy.zeros((1, NXcon))
    for i in range(nevcs):
        Csmeq[0][NXcon6 + i] = 1.0  # Pevel

    # printCheckMatrix(Ccoeq, nevcs * (nstage - 1), NXbinsum, 'Ccoeq')
    # printCheckMatrix(Cpteq, nevcs * (nstage - 1), NXconsum, 'Cpteq')
    # printCheckMatrix(Cnueq, nstage, NXbin, 'Cnueq')
    # printCheckMatrix(Clseq, nevcs, NXcon, 'Clseq')
    # printCheckMatrix(Clxeq, nevcs, NXcon, 'Clxeq')
    # printCheckMatrix(Clseqplus, nevcs, NXbin, 'Clseqplus')
    # printCheckMatrix(Clxeqplus, nevcs, NXbin, 'Clxeqplus')
    # printCheckMatrix(Ceveq, nevcs, NXcon, 'Ceveq')
    # printCheckMatrix(Csmeq, 1, NXcon, 'Csmeq')
    # exit()

    # //////////////////////////////////////////////////////////////
    # Dcoeq
    Dcoeq = numpy.zeros((ndgen * (nstage - 1), NXbinsum))
    for p in range(1, nstage):
        for i in range(ndgen):
            Dcoeq[(p - 1) * ndgen + i][(p - 1) * NXbin + NXbin2 + i] = 1.0  # delta_(p-1)
            Dcoeq[(p - 1) * ndgen + i][p * NXbin + NXbin2 + i] = -1.0  # delta_(p)

    # Dpteq
    Dpteq = numpy.zeros((ndgen * (nstage - 1), NXconsum))
    for p in range(1, nstage):
        for i in range(ndgen):
            Dpteq[(p - 1) * ndgen + i][(p - 1) * NXcon + NXcon7 + ndgen * 2 + i] = 1.0  # Sdgen_(p-1)
            Dpteq[(p - 1) * ndgen + i][p * NXcon + NXcon7 + ndgen * 2 + i] = -1.0  # Sdgen_(p)

    # Dnueq
    Dnueq = numpy.zeros((nstage, NXbin))
    for p in range(nstage):
        for i in range(ndgen):
            Dnueq[p][NXbin2 + i] = 1.0  # delta

    # Dlseq Dlxeq
    Dlseq = numpy.zeros((ndgen, NXcon))
    Dlxeq = numpy.zeros((ndgen, NXcon))
    for i in range(ndgen):
        Dlseq[i][NXcon7 + ndgen * 2 + i] = 1.0  # Sdgen
        Dlxeq[i][NXcon7 + ndgen * 2 + i] = -1.0  # -Sdgen

    # Dlseqplus Dlxeqplus
    Dlseqplus = numpy.zeros((ndgen, NXbin))
    Dlxeqplus = numpy.zeros((ndgen, NXbin))
    for i in range(ndgen):
        Dlseqplus[i][NXbin2 + i] = -sdgen[i]  # delta
        Dlxeqplus[i][NXbin2 + i] = smini_cvt  # delta

    # Ddgeq
    Ddgeq = numpy.zeros((ndgen, NXcon))
    for i in range(ndgen):
        Ddgeq[i][NXcon7 + i] = 1.0  # Pdgen
        Ddgeq[i][NXcon7 + ndgen * 2 + i] = -fdgen[i]  # -μ*Sdgen

    # Dsmeq
    Dsmeq = numpy.zeros((1, NXcon))
    for i in range(ndgen):
        Dsmeq[0][NXcon7 + i] = 1.0  # Pdgen

    # printCheckMatrix(Dcoeq, ndgen * (nstage - 1), NXconsum, 'Dcoeq')
    # printCheckMatrix(Dpteq, ndgen * (nstage - 1), NXconsum, 'Dpteq')
    # printCheckMatrix(Dnueq, nstage, NXbin, 'Dnueq')
    # printCheckMatrix(Dlseq, ndgen, NXcon, 'Dlseq')
    # printCheckMatrix(Dlxeq, ndgen, NXcon, 'Dlxeq')
    # printCheckMatrix(Dlseqplus, ndgen, NXbin, 'Dlseqplus')
    # printCheckMatrix(Dlxeqplus, ndgen, NXbin, 'Dlxeqplus')
    # printCheckMatrix(Ddgeq, ndgen, NXcon, 'Ddgeq')
    # printCheckMatrix(Dsmeq, 1, NXcon, 'Dsmeq')
    # exit()

    end = time.time()
    print('Time= {}'.format(end-start))

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    print('Planning model: Write constraints')
    start = time.time()

    # //////////////////////////////////////////////////////////////
    print('Planning model: Write constraints-variable')
    for p in range(nstage):
        # Pfneq
        for i in range(ndgen):
            idxa = p * NXcon + NXcon7 + i  # Pdgen
            idxb = p * NXcon + NXcon7 + ndgen + i  # Qdgen

            finit = fdgen[i]
            tanpf = numpy.sqrt(1 - numpy.square(finit)) / finit

            if QdgcrSign:
                Picos += xcon[idxb] <= +xcon[idxa] * tanpf
                Picos += xcon[idxb] >= -xcon[idxa] * tanpf
            else:
                Picos += xcon[idxb] == xcon[idxa] * tanpf

        # Pcveq Qcveq
        for i, schm in enumerate(scheme):
            for j, schm_node in enumerate(schm):
                idx = sum(nscht[:i]) + j
                idxa = p * NXcon + NXcon4 + idx  # Psopt
                idxb = p * NXcon + NXcon4 + nschp + idx  # Qsopt
                idxc = p * NXcon + NXcon4 + nschp * 3 + idx  # Ssopt

                Picos += -xcon[idxc] <= xcon[idxa] <= xcon[idxc]
                Picos += -xcon[idxc] <= xcon[idxb] <= xcon[idxc]

    # //////////////////////////////////////////////////////////////
    print('Planning model: Write constraints-linear')
    for p in range(nstage):
        interval_con = range(p * NXcon, (p + 1) * NXcon)
        interval_bin = range(p * NXbin, (p + 1) * NXbin)

        if nnode > 0:
            Picos += Pineq * xcon[interval_con] == pload_aux[p] * ExptVar['Load']
            Picos += Qineq * xcon[interval_con] == qload_aux[p] * ExptVar['Load']
        if nline > 0:
            Picos += Uineq * xcon[interval_con] == 0.0
            Picos += Uiaeq * xcon[interval_con] == 0.0
            Picos += Uibeq * xcon[interval_con] == 0.0
        if nschm > 0:
            Picos += Pisop * xcon[interval_con] == 0.0
        if nschm > 0:
            Picos += Sadeq * xcon[interval_con] == 0.0
            Picos += Slseq * xcon[interval_con] + Slseqplus * xbin[interval_bin] <= 0.0
            Picos += Slxeq * xcon[interval_con] + Slxeqplus * xbin[interval_bin] <= 0.0
        if nport > 0:
            Picos += Kuneq * xbin[interval_bin] <= 1.0
        if nevcs > 0:
            Picos += Clseq * xcon[interval_con] + Clseqplus * xbin[interval_bin] <= 0.0
            Picos += Clxeq * xcon[interval_con] + Clxeqplus * xbin[interval_bin] <= 0.0
            Picos += Ceveq * xcon[interval_con] <= 0.0
            Picos += Csmeq * xcon[interval_con] == pevel_stg[p]
            Picos += Cnueq * xbin[interval_bin] <= Nevcs_max[p]
        if ndgen > 0:
            Picos += Dlseq * xcon[interval_con] + Dlseqplus * xbin[interval_bin] <= 0.0
            Picos += Dlxeq * xcon[interval_con] + Dlxeqplus * xbin[interval_bin] <= 0.0
            Picos += Ddgeq * xcon[interval_con] <= 0.0
            Picos += Dsmeq * xcon[interval_con] == pdgen_stg[p]
            Picos += Dnueq * xbin[interval_bin] <= Ndgen_max[p]
        if nline > 0:
            if BavaSign:
                Picos += Alzeq * xcon[interval_con] >= -Imaxlit_bas ** 2
        if nnode > 0:
            if BavaSign:
                Picos += Blzeq * xcon[interval_con] >= -Vmaxlit_vas ** 2
                Picos += Blseq * xcon[interval_con] >= Vminlit_vas ** 2

    if nschm > 0:
        Picos += Kcoeq * xbin <= 0.0
    if nport > 0:
        Picos += Spteq * xcon <= 0.0
    if nevcs > 0:
        Picos += Ccoeq * xbin <= 0.0
        Picos += Cpteq * xcon <= 0.0
    if ndgen > 0:
        Picos += Dcoeq * xbin <= 0.0
        Picos += Dpteq * xcon <= 0.0

    # //////////////////////////////////////////////////////////////
    print('Planning model: Write constraints-cone')
    for p in range(nstage):
        for i in range(nline):  # (UIb)2≥(Pij)2+(Qij)2+(UIa)2
            idxa = p * NXcon + i  # Pij
            idxb = p * NXcon + nline + i  # Qij
            idxc = p * NXcon + NXcon2 + i  # UIa
            idxd = p * NXcon + NXcon2 + nline + i  # UIb
            Picos += abs(xcon[[idxa, idxb, idxc]]) <= xcon[idxd]

        for i, schm in enumerate(scheme):  # (Ssopt)2≥(Psopt)2+(Qsopt)2
            for j, schm_node in enumerate(schm):
                idx = sum(nscht[:i]) + j
                idxa = p * NXcon + NXcon4 + idx  # Psopt
                idxb = p * NXcon + NXcon4 + nschp + idx  # Qsopt
                idxc = p * NXcon + NXcon4 + nschp * 3 + idx  # Ssopt
                Picos += abs(xcon[[idxa, idxb]]) <= xcon[idxc]

        for i, schm in enumerate(scheme):  # (Msopt)2≥(Psopt)2+(Qsopt)2
            for j, schm_node in enumerate(schm):
                idx = sum(nscht[:i]) + j
                idxa = p * NXcon + NXcon4 + idx  # Psopt
                idxb = p * NXcon + NXcon4 + nschp + idx  # Qsopt
                idxc = p * NXcon + NXcon4 + nschp * 2 + idx  # Msopt
                Picos += abs(xcon[[idxa, idxb]]) * floss_cvt <= xcon[idxc]
    # print(Picos)
    # exit()

    end = time.time()
    print('Time= {}'.format(end-start))

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    print('Planning model: Solve problem')
    start = time.time()

    solution = Picos.solve(solver=Solver)
    print('Solver= ', Solver)
    print('Status= ', solution.claimedStatus)

    end = time.time()
    print('Time= {}'.format(end-start))

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    print('Planning model: Extract results')
    start = time.time()

    xsol_con = numpy.array(xcon.value).flatten()
    xsol_bin = numpy.array(xbin.value).flatten()

    Unode = numpy.zeros((nstage, nnode))
    Iline = numpy.zeros((nstage, nline))
    Ildrt = numpy.zeros((nstage, nline))
    Gapxy = numpy.zeros((nstage, nline))
    Gapls = numpy.zeros((nstage, nschp))
    Gapps = numpy.zeros((nstage, nschp))
    Psopt = numpy.zeros((nstage, nschp))
    Qsopt = numpy.zeros((nstage, nschp))
    Msopt = numpy.zeros((nstage, nschp))
    Ssopt = numpy.zeros((nstage, nschp))
    Ssopk = numpy.zeros((nstage, nschm))
    Pevel = numpy.zeros((nstage, nevcs))
    Sevcs = numpy.zeros((nstage, nevcs))
    Pdgen = numpy.zeros((nstage, ndgen))
    Qdgen = numpy.zeros((nstage, ndgen))
    Sdgen = numpy.zeros((nstage, ndgen))
    alpha = numpy.zeros((nstage, nschm), dtype=int)
    betas = numpy.zeros((nstage, nevcs), dtype=int)
    delta = numpy.zeros((nstage, ndgen), dtype=int)
    nshop = numpy.zeros(nstage)

    # Unode
    for p in range(nstage):
        for i in range(nnode):
            Unode[p][i] = numpy.sqrt(xsol_con[p * NXcon + NXcon1 + i])  # Ui/Uj (p.u.)

    # Iline
    for p in range(nstage):
        for i in range(nline):
            Iline[p][i] = numpy.sqrt(xsol_con[p * NXcon + nline * 2 + i])  # Iij (p.u.)
            Ildrt[p][i] = Iline[p][i] / Irate

    # Psopt Qsopt Msopt Ssopt
    for p in range(nstage):
        for i in range(nschp):
            Psopt[p][i] = xsol_con[p * NXcon + NXcon4 + i]
            Qsopt[p][i] = xsol_con[p * NXcon + NXcon4 + nschp * 1 + i]
            Msopt[p][i] = xsol_con[p * NXcon + NXcon4 + nschp * 2 + i]
            Ssopt[p][i] = xsol_con[p * NXcon + NXcon4 + nschp * 3 + i]

    # Gapxy
    for p in range(nstage):
        for i in range(nline):
            for j in range(nnode):
                if bgbus[i] == inode[j]:
                    Gapxy[p][i] = xsol_con[p * NXcon + nline * 2 + i] * xsol_con[p * NXcon + NXcon1 + j] - \
                                  numpy.square(xsol_con[p * NXcon + i]) - numpy.square(xsol_con[p * NXcon + nline + i])

    # Gapls
    for p in range(nstage):
        for i in range(nschp):
            Gapls[p][i] = numpy.square(Msopt[p][i]) - numpy.square(floss_cvt) * (numpy.square(Psopt[p][i]) + numpy.square(Qsopt[p][i]))

    # Gapps
    for p in range(nstage):
        for i, schm in enumerate(scheme):
            schm_range = range(sum(nscht[:i]), sum(nscht[:i + 1]))

            schm_Psopt = numpy.sum(Psopt[p][schm_range])
            schm_Msopt = numpy.sum(Msopt[p][schm_range])
            Gapps[p][i] = schm_Psopt - schm_Msopt

    # Ssopk
    for p in range(nstage):
        for i in range(nschm):
            Ssopk[p][i] = xsol_con[p * NXcon + NXcon5 + i]

    # Pevel Sevcs
    for p in range(nstage):
        for i in range(nevcs):
            Pevel[p][i] = xsol_con[p * NXcon + NXcon6 + i]
            Sevcs[p][i] = xsol_con[p * NXcon + NXcon6 + nevcs + i]

            float_betas = xsol_bin[p * NXbin + NXbin1 + i]
            if float_betas >= 1 - 1e-3:
                betas[p][i] = 1

    # Pdgen Qdgen Sdgen
    for p in range(nstage):
        for i in range(ndgen):
            Pdgen[p][i] = xsol_con[p * NXcon + NXcon7 + i]
            Qdgen[p][i] = xsol_con[p * NXcon + NXcon7 + ndgen * 1 + i]
            Sdgen[p][i] = xsol_con[p * NXcon + NXcon7 + ndgen * 2 + i]

            float_delta = xsol_bin[p * NXbin + NXbin2 + i]
            if float_delta >= 1 - 1e-3:
                delta[p][i] = 1

    # /////////////////////////////////////////////////////////////
    Ssopt = ceilDecimal(Ssopt, smaxi_cvt, decml_cap)
    Ssopk = ceilDecimal(Ssopk, smaxi_cvt, decml_cap)
    Sevcs = ceilDecimal(Sevcs, sevcs, decml_cap)
    Sdgen = ceilDecimal(Sdgen, sdgen, decml_cap)

    # /////////////////////////////////////////////////////////////
    Schme_select = [list() for i in range(nstage)]
    Ssopt_select = numpy.zeros((nstage, nport))
    for p in range(nstage):
        for i in range(nschm):
            float_alpha = xsol_bin[p * NXbin + i]
            # print(p, scheme[i], float_alpha)

            if float_alpha >= 1 - 1e-3:
                alpha[p][i] = 1

        alpha_idxn = numpy.nonzero(alpha[p])[0]
        Schme_select[p] = [scheme[i] for i in alpha_idxn]

        for i, schm in zip(alpha_idxn, Schme_select[p]):
            for j, schm_node in enumerate(schm):
                idx = sum(nscht[:i]) + j
                Ssopt_select[p][mport[schm_node]] = Ssopt[p][idx]

    Unodemax = numpy.max(Unode, axis=1)
    Unodemin = numpy.min(Unode, axis=1)
    Ilinemax = numpy.max(Iline, axis=1)
    Ildrtmax = numpy.max(Ildrt, axis=1)

    # /////////////////////////////////////////////////////////////
    Gapxymax = numpy.max(Gapxy)
    Gaplsmax = numpy.max(Gapls)
    Gappsmax = numpy.max(Gapps)

    # //////////////////////////////////////////////////////////////
    Df_index = ['stage{}'.format(i+1) for i in range(nstage)]
    Df_Uigap = pandas.DataFrame({'Unodemax': Unodemax, 'Unodemin': Unodemin, 'Ildrtmax': Ildrtmax,
                                 'Gapxymax': Gapxymax, 'Gaplsmax': Gaplsmax, 'Gappsmax': Gappsmax}, index=Df_index)
    pandas.set_option('display.max_columns', None)
    print('Imaxlit_sec: {:.6f}'.format(Imaxlit_sec))
    print('Df_Ungap:\n', Df_Uigap, '\n')

    # //////////////////////////////////////////////////////////////
    Df_Schme_select = pandas.DataFrame(Schme_select, index=Df_index)
    print('Df_Schme_select:\n', Df_Schme_select, '\n')

    Df_Ssopt_select = pandas.DataFrame(Ssopt_select, index=Df_index, columns=iport)
    Df_Ssopt_select['sum'] = Df_Ssopt_select.apply(lambda x: x.sum(), axis=1)
    print('Df_Ssopt_select:\n', Df_Ssopt_select, '\n')

    # //////////////////////////////////////////////////////////////
    Df_Pevel = pandas.DataFrame(Pevel, index=Df_index, columns=cevcs)
    Df_Pevel['sum'] = Df_Pevel.apply(lambda x: x.sum(), axis=1)
    Df_Pevel['ptr'] = Df_Pevel['sum'] / load_pvt
    print('Df_Pevel:\n', Df_Pevel.round(decml_cap), '\n')

    Df_Sevcs = pandas.DataFrame(Sevcs, index=Df_index, columns=cevcs)
    Df_Sevcs['sum'] = Df_Sevcs.apply(lambda x: x.sum(), axis=1)
    print('Df_Sevcs:\n', Df_Sevcs, '\n')
    # exit()

    # //////////////////////////////////////////////////////////////
    Df_Pdgen = pandas.DataFrame(Pdgen, index=Df_index, columns=cdgen)
    Df_Pdgen['sum'] = Df_Pdgen.apply(lambda x: x.sum(), axis=1)
    Df_Pdgen['ptr'] = Df_Pdgen['sum'] / load_pvt
    print('Df_Pdgen:\n', Df_Pdgen.round(decml_cap), '\n')

    Df_Sdgen = pandas.DataFrame(Sdgen, index=Df_index, columns=cdgen)
    Df_Sdgen['sum'] = Df_Sdgen.apply(lambda x: x.sum(), axis=1)
    print('Df_Sdgen:\n', Df_Sdgen, '\n')

    # //////////////////////////////////////////////////////////////
    CostCon_sop = numpy.zeros(nstage)
    CostCon_sit = numpy.zeros(nstage)
    CostCon_ref = numpy.zeros(nstage)
    CostCon_evs = numpy.zeros(nstage)
    CostCon_sev = numpy.zeros(nstage)
    CostCon_dgc = numpy.zeros(nstage)

    for p in range(nstage):
        zeta = pow(1 + infla, -p * nyear)

        if p == 0:
            CostCon_sop[p] = zeta * price_cvt[p] * Sbase * numpy.sum(Ssopk[p])
            CostCon_sit[p] = zeta * price_sit[p] * numpy.sum(alpha[p])
            CostCon_ref[p] = zeta * price_ref[p] * modfy_geo * numpy.sum(alpha[p] * lschm)

            CostCon_evs[p] = zeta * price_cvt[p] * Sbase * numpy.sum(Sevcs[p])
            CostCon_sev[p] = zeta * price_sit[p] * numpy.sum(betas[p])

            CostCon_dgc[p] = zeta * price_cvt[p] * Sbase * numpy.sum(Sdgen[p])
        else:
            # //////////////////////////////////////////////////////////////
            increment = Ssopk[p] - Ssopk[p - 1]
            CostCon_sop[p] = zeta * price_cvt[p] * Sbase * numpy.sum(increment)

            # //////////////////////////////////////////////////////////////
            increment = alpha[p] - alpha[p - 1]
            CostCon_sit[p] = zeta * price_sit[p] * numpy.sum(increment)
            CostCon_ref[p] = zeta * price_ref[p] * modfy_geo * numpy.sum(increment * lschm)

            # //////////////////////////////////////////////////////////////
            increment = Sevcs[p] - Sevcs[p - 1]
            CostCon_evs[p] = zeta * price_cvt[p] * Sbase * numpy.sum(increment)

            # //////////////////////////////////////////////////////////////
            increment = betas[p] - betas[p - 1]
            CostCon_sev[p] = zeta * price_sit[p] * numpy.sum(increment)

            # //////////////////////////////////////////////////////////////
            increment = Sdgen[p] - Sdgen[p - 1]
            CostCon_dgc[p] = zeta * price_cvt[p] * Sbase * numpy.sum(increment)

    CostCon_sop = numpy.around(CostCon_sop / Cbase, decimals=2)
    CostCon_sit = numpy.around(CostCon_sit / Cbase, decimals=2)
    CostCon_ref = numpy.around(CostCon_ref / Cbase, decimals=2)
    CostCon_evs = numpy.around(CostCon_evs / Cbase, decimals=2)
    CostCon_sev = numpy.around(CostCon_sev / Cbase, decimals=2)
    CostCon_dgc = numpy.around(CostCon_dgc / Cbase, decimals=2)

    # print('CostCon_sop: ', CostCon_sop)
    # print('CostCon_sit: ', CostCon_sit)
    # print('CostCon_ref: ', CostCon_ref)
    # print('CostCon_evs: ', CostCon_evs)
    # print('CostCon_sev: ', CostCon_sev)
    # print('CostCon_dgc: ', CostCon_dgc)

    # //////////////////////////////////////////////////////////////
    CostOpr_sls = numpy.zeros(nstage)
    CostOpr_bas = numpy.zeros(nstage)
    CostOpr_vas = numpy.zeros(nstage)

    for p in range(nstage):
        zeta = pow(1 + infla, -p * nyear)
        zeta_hours = zeta * nhour * epsilon

        interval_line = range(p * NXcon + nline * 2, p * NXcon + nline * 3)
        interval_sopp = range(p * NXcon + NXcon4 + nschp * 2, p * NXcon + NXcon4 + nschp * 3)
        CostOpr_lineloss = zeta_hours * price_ele[p] * Sbase * numpy.sum(xsol_con[interval_line] * rline)
        CostOpr_sopploss = zeta_hours * price_ele[p] * Sbase * numpy.sum(xsol_con[interval_sopp])
        CostOpr_sls[p] = CostOpr_lineloss + CostOpr_sopploss

        if BavaSign:
            interval_bas = range(p * NXcon + NXcon8, p * NXcon + NXcon8 + nline)
            CostOpr_bas[p] = zeta_hours * price_bas * numpy.sum(xsol_con[interval_bas])
        if BavaSign:
            interval_vas = range(p * NXcon + NXcon9, p * NXcon + NXcon9 + nnode)
            CostOpr_vas[p] = zeta_hours * price_vas * numpy.sum(xsol_con[interval_vas])

    CostOpr_sls = numpy.around(CostOpr_sls / Cbase, decimals=2)
    CostOpr_bas = numpy.around(CostOpr_bas / Cbase, decimals=2)
    CostOpr_vas = numpy.around(CostOpr_vas / Cbase, decimals=2)

    # print('CostOpr_sls: ', CostOpr_sls)
    # print('CostOpr_bas: ', CostOpr_bas)
    # print('CostOpr_vas: ', CostOpr_vas)

    Dc_Cost = {'sit': CostCon_sit, 'sop': CostCon_sop, 'ref': CostCon_ref,
               'sev': CostCon_sev, 'evs': CostCon_evs, 'dgc': CostCon_dgc, 'sls': CostOpr_sls}
    for keys in Dc_Cost: Dc_Cost[keys] = list(Dc_Cost[keys])

    Df_Cost = pandas.DataFrame(Dc_Cost, index=Df_index)
    Df_Cost['rsum'] = Df_Cost.apply(lambda x: x.sum(), axis=1)
    Df_Cost.loc['csum'] = Df_Cost.apply(lambda x: x.sum())
    print('Df_Cost:\n', Df_Cost, '\n')

    Resplot = {'volt': Unode, 'line': Iline, 'cost': Dc_Cost,
               'strg': Ssopt, 'pevl': Pevel, 'pdgn': Pdgen,
               'alph': alpha, 'beta': betas, 'delt': delta}

    end = time.time()
    print('Time= {}'.format(end-start))

    return Resplot


