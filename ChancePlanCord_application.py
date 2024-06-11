import numpy
from collections import deque
import picos


'''
Establish the flexible distribution network operation model for application

'''

from ChancePlanCord_distrgen import WindSpeedtoPower, SolaradiatoPower, limitGeneratorOutput


def computeOptimizationOperationApplEngine(p, RandVar, OptStrg, Sprvs, ndata, params, pchild, DgLdEvCur=False, BavaSign=False, Solver='gurobi'):
    # p-stage index
    # RandVar-random variable sampling
    # OptStrg-given planning strategy
    # Sprvs-dimension intervals corresponding to random variables in each stage
    # BavaSign-whether the objective function includes load balancing and voltage deviation
    # ndata-number of samples
    # Solver-optimization solver
    # params-auxiliary parameters
    # DgLdEvCur-whether renewable energy curtailment and load shedding are permitted

    [Sbase, Vstation, Vminlit_vas, Vmaxlit_vas, Imaxlit_bas, vci, vco, vn, refactor, squarekw,
     nnode, inode, mnode, islck, nslck, nline, bgbus, edbus, rline, xline, iline_pal,
     nload, cload, pload_aux, qload_aux, cevcs, cdgen, fdgen, tdgen, nyear, nhour, epsilon, infla,
     conctmax, connect_id, connect_sr, scheme, nscht, floss_cvt,
     price_ele, price_dga, price_lda, price_eva, price_vas, price_bas, Irisk, Vrisk,
     volt_nlad, line_nlad, Vminlit_lad, Vmaxlit_lad, Imaxlit_lad, Vminlit_sec, Vmaxlit_sec, Imaxlit_sec] = params

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # Planning schemes of SOP
    Ssopt = OptStrg['strg']
    alpha = OptStrg['alph']

    alpha_modi = numpy.where(alpha >= 1 - 1e-3, 1.0, alpha)
    alpha_idxn = numpy.nonzero(alpha_modi)[0]

    nscht_op = [nscht[i] for i in alpha_idxn]
    schme_op = [scheme[i] for i in alpha_idxn]
    nshmp = len(nscht_op)
    nshop = sum(nscht_op)
    # print('nscht_op: ', nscht_op)
    # print('schme_op: ', schme_op)

    schnd_op = numpy.zeros(nshop, dtype=int)
    Ssopt_op = numpy.zeros(nshop)
    cnt = 0
    for i, schm in zip(alpha_idxn, schme_op):
        for j, schm_node in enumerate(schm):
            idx = sum(nscht[:i]) + j
            schnd_op[cnt] = schm_node
            Ssopt_op[cnt] = Ssopt[idx]
            cnt += 1
    # print('Ssopt_op: ', Ssopt_op)

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # Planning schemes of electric vehicle charging stations
    Pevel = OptStrg['pevl']
    betas = OptStrg['beta']

    betas_modi = numpy.where(betas >= 1 - 1e-3, 1.0, betas)
    betas_idxn = numpy.nonzero(betas_modi)[0]
    Pevel_op = Pevel[betas_idxn]
    cevop = cevcs[betas_idxn]
    nevop = len(betas_idxn)
    # print('betas_idxn: ', betas_idxn)
    # print('cevop: ', cevop)
    # print('Pevel_op: ', Pevel_op)

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # Planning schemes of distributed generators
    Pdgen = OptStrg['pdgn']
    delta = OptStrg['delt']

    delta_modi = numpy.where(delta >= 1 - 1e-3, 1.0, delta)
    delta_idxn = numpy.nonzero(delta_modi)[0]
    Pdgen_op = Pdgen[delta_idxn]
    cdgop = cdgen[delta_idxn]
    ndgop = len(delta_idxn)
    # print('delta_idxn: ', delta_idxn)
    # print('cdgop: ', cdgop)
    # print('Pdgen_op: ', Pdgen_op)
    # exit()

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    pevel_aux = numpy.zeros(nnode)
    for i in range(nevop):
        pevel_aux[mnode[cevop[i]]] = Pevel_op[i]
    # print('pevel_aux: ', pevel_aux)

    pdgen_aux = numpy.zeros(nnode)
    qdgen_aux = numpy.zeros(nnode)
    for i in range(ndgop):
        finit = fdgen[delta_idxn[i]]
        tanpf = numpy.sqrt(1 - numpy.square(finit)) / finit

        pdgen_aux[mnode[cdgop[i]]] = Pdgen_op[i]
        qdgen_aux[mnode[cdgop[i]]] = Pdgen_op[i] * tanpf

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    Load_real = RandVar[Sprvs['Load']]
    Evel_real = RandVar[Sprvs['Evel']]

    Load_real_aux = numpy.zeros((nnode, ndata))
    Evel_real_aux = numpy.zeros((nnode, ndata))
    for i in range(nload):
        Load_real_aux[mnode[cload[i]]] = Load_real[i]
    for i in range(nevop):
        Evel_real_aux[mnode[cevop[i]]] = Evel_real[betas_idxn[i]]

    # //////////////////////////////////////////////////////////////
    Wind_real = RandVar[Sprvs['Wind']]
    Sola_real = RandVar[Sprvs['Sola']]

    Wind_real_out = WindSpeedtoPower(Wind_real, vci, vco, vn)
    Sola_real_out = SolaradiatoPower(Sola_real, refactor, squarekw)

    Wind_real_lit = limitGeneratorOutput(Wind_real_out)
    Sola_real_lit = limitGeneratorOutput(Sola_real_out)

    Dgen_real_aux = numpy.zeros((nnode, ndata))
    for i in range(ndgop):
        if tdgen[delta_idxn[i]] == 1: Dgen_real_aux[mnode[cdgop[i]]] = Sola_real_lit[delta_idxn[i]]
        if tdgen[delta_idxn[i]] == 2: Dgen_real_aux[mnode[cdgop[i]]] = Wind_real_lit[delta_idxn[i]]

    # //////////////////////////////////////////////////////////////
    Load_real_aux = numpy.where(Load_real_aux <= 0.0, 0.0, Load_real_aux)
    Evel_real_aux = numpy.where(Evel_real_aux <= 0.0, 0.0, Evel_real_aux)
    Dgen_real_aux = numpy.where(Dgen_real_aux <= 0.0, 0.0, Dgen_real_aux)

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    print('Operation model-Preparation: Define varaibles')

    # //////////////////////////////////////////////////////////////
    # continuous-static variable Number
    if BavaSign:
        # ->Pij Qij Iij
        NXcon1 = nline * 3  # ->Ui
        NXcon2 = NXcon1 + nnode * 1  # ->UIa UIb
        NXcon3 = NXcon2 + nline * 2  # ->Pba Qba
        NXcon4 = NXcon3 + nslck * 2  # ->PSopt QSopt MSopt SSopt
        NXcon5 = NXcon4 + nshop * 4  # ->Pdgcur Qdgcur
        NXcon6 = NXcon5 + ndgop * 2  # ->Pldcur Qldcur
        NXcon7 = NXcon6 + nload * 2  # ->Pevcur
        NXcon8 = NXcon7 + nevop * 1  # ->Aij
        NXcon9 = NXcon8 + nline * 1  # ->Bi
        NXcon = NXcon9 + nnode * 1
    else:
        # ->Pij Qij Iij
        NXcon1 = nline * 3  # ->Ui
        NXcon2 = NXcon1 + nnode * 1  # ->UIa UIb
        NXcon3 = NXcon2 + nline * 2  # ->Pba Qba
        NXcon4 = NXcon3 + nslck * 2  # ->PSopt QSopt MSopt SSopt
        NXcon5 = NXcon4 + nshop * 4  # ->Pdgcur Qdgcur
        NXcon6 = NXcon5 + ndgop * 2  # ->Pldcur Qldcur
        NXcon7 = NXcon6 + nload * 2  # ->Pevcur
        NXcon8 = 0
        NXcon9 = 0
        NXcon = NXcon7 + nevop * 1

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    print('Operation model-Preparation: Confine varaibles')

    xcon_lower = numpy.zeros(NXcon)
    xcon_upper = numpy.zeros(NXcon)

    for i in range(nline):  # Pij
        xcon_lower[i] = -numpy.inf
        xcon_upper[i] = +numpy.inf
    for i in range(nline):  # Qij
        xcon_lower[nline + i] = -numpy.inf
        xcon_upper[nline + i] = +numpy.inf
    for i in range(nline):  # Iij
        if DgLdEvCur:
            if i in iline_pal:
                xcon_lower[nline * 2 + i] = 0.0
                xcon_upper[nline * 2 + i] = (Imaxlit_sec * 2) ** 2
            else:
                xcon_lower[nline * 2 + i] = 0.0
                xcon_upper[nline * 2 + i] = Imaxlit_sec ** 2
        else:
            xcon_lower[nline * 2 + i] = 0.0
            xcon_upper[nline * 2 + i] = +numpy.inf

    for i in range(nnode):  # Ui
        if inode[i] in islck:
            xcon_lower[NXcon1 + i] = Vstation ** 2
            xcon_upper[NXcon1 + i] = Vstation ** 2
        else:
            if DgLdEvCur:
                xcon_lower[NXcon1 + i] = Vminlit_sec ** 2
                xcon_upper[NXcon1 + i] = Vmaxlit_sec ** 2
            else:
                xcon_lower[NXcon1 + i] = 0.0
                xcon_upper[NXcon1 + i] = +numpy.inf

    for i in range(nline):  # UIa
        xcon_lower[NXcon2 + i] = -numpy.inf
        xcon_upper[NXcon2 + i] = +numpy.inf
    for i in range(nline):  # UIb
        xcon_lower[NXcon2 + nline + i] = -numpy.inf
        xcon_upper[NXcon2 + nline + i] = +numpy.inf

    for i in range(nslck):  # Pba
        xcon_lower[NXcon3 + i] = -numpy.inf
        xcon_upper[NXcon3 + i] = +numpy.inf
    for i in range(nslck):  # Qba
        xcon_lower[NXcon3 + nslck + i] = -numpy.inf
        xcon_upper[NXcon3 + nslck + i] = +numpy.inf

    for i in range(nshop):  # Psopt
        xcon_lower[NXcon4 + i] = -numpy.inf
        xcon_upper[NXcon4 + i] = +numpy.inf
    for i in range(nshop):  # Qsopt
        xcon_lower[NXcon4 + nshop + i] = -numpy.inf
        xcon_upper[NXcon4 + nshop + i] = +numpy.inf
    for i in range(nshop):  # Msopt
        xcon_lower[NXcon4 + nshop * 2 + i] = 0.0
        xcon_upper[NXcon4 + nshop * 2 + i] = +numpy.inf
    for i in range(nshop):  # Ssopt
        xcon_lower[NXcon4 + nshop * 3 + i] = 0.0
        xcon_upper[NXcon4 + nshop * 3 + i] = Ssopt_op[i]

    if BavaSign:
        for i in range(nline):  # Aij
            xcon_lower[NXcon8 + i] = 0.0
            xcon_upper[NXcon8 + i] = +numpy.inf

        for i in range(nnode):  # Bi
            xcon_lower[NXcon9 + i] = 0.0
            xcon_upper[NXcon9 + i] = +numpy.inf

    # print('xcon_lower: ', xcon_lower)
    # print('xcon_upper: ', xcon_upper)

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    print('Operation model-Preparation: Construct constraint matrices')

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

        for j, schm in enumerate(schme_op):
            for k, schm_node in enumerate(schm):
                if inode[i] == schm_node:
                    idx = sum(nscht_op[:j]) + k
                    Pineq[i][NXcon4 + idx] = 1.0  # PSopt
                    Qineq[i][NXcon4 + nshop + idx] = 1.0  # QSopt

        for j in range(ndgop):
            if inode[i] == cdgop[j]:
                Pineq[i][NXcon5 + j] = -1.0  # -Pdgcur
                Qineq[i][NXcon5 + ndgop + j] = -1.0  # -Qdgcur

        for j in range(nload):
            if inode[i] == cload[j]:
                Pineq[i][NXcon6 + j] = 1.0          # Pldcur
                Qineq[i][NXcon6 + nload + j] = 1.0  # Qldcur

        for j in range(nevop):
            if inode[i] == cevop[j]:
                Pineq[i][NXcon7 + j] = 1.0  # Pevcur

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
    Pisop = numpy.zeros((nshmp, NXcon))
    for i, schm in enumerate(schme_op):
        for j, schm_node in enumerate(schm):
            idx = sum(nscht_op[:i]) + j
            Pisop[i][NXcon4 + idx] = 1.0  # PSopt
            Pisop[i][NXcon4 + nshop * 2 + idx] = -1.0  # MSopt

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
    # printCheckMatrix(Pisop, nshmp, NXcon, 'Pisop')
    # printCheckMatrix(Alzeq, nline, NXcon, 'Alzeq')
    # printCheckMatrix(Blzeq, nnode, NXcon, 'Blzeq')
    # printCheckMatrix(Blseq, nnode, NXcon, 'Blseq')
    # exit()

    # //////////////////////////////////////////////////////////////
    Unode = numpy.zeros((nnode, ndata))
    Iline = numpy.zeros((nline, ndata))

    CostOpr_lineloss = 0.0
    CostOpr_sopploss = 0.0

    CostOpr_sls = 0.0
    CostOpr_dga = 0.0
    CostOpr_lda = 0.0
    CostOpr_eva = 0.0
    CostOpr_bas = 0.0
    CostOpr_vas = 0.0

    zeta = pow(1 + infla, -p * nyear)
    zeta_hours = zeta * nhour * epsilon / ndata

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    print('Operation model-Iteratiion: calculation')

    for m in range(ndata):
        if m % 10000 == 0 or m == ndata - 1:
            print('p={} m={}'.format(p, m))

        Psopt = numpy.zeros(nshop)
        Qsopt = numpy.zeros(nshop)
        Msopt = numpy.zeros(nshop)

        Gapxy = numpy.zeros(nline)
        Gapls = numpy.zeros(nshop)
        Gapps = numpy.zeros(nshop)

        # //////////////////////////////////////////////////////////////////////////////////////////////////////////////
        Picos = picos.Problem()

        # //////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # print('Operation model-Iteratiion: Confine varaibles')

        if DgLdEvCur:
            for i in range(ndgop):  # Pdgcur
                idx = mnode[cdgop[i]]
                xcon_lower[NXcon5 + i] = 0.0
                xcon_upper[NXcon5 + i] = pdgen_aux[idx] * Dgen_real_aux[idx][m]
            for i in range(ndgop):  # Qdgcur
                xcon_lower[NXcon5 + ndgop + i] = 0.0
                xcon_upper[NXcon5 + ndgop + i] = +numpy.inf

            for i in range(nload):  # Pldcur
                idx = mnode[cload[i]]
                xcon_lower[NXcon6 + i] = 0.0
                xcon_upper[NXcon6 + i] = pload_aux[p, idx] * Load_real_aux[idx][m]
            for i in range(nload):  # Qldcur
                xcon_lower[NXcon6 + nload + i] = 0.0
                xcon_upper[NXcon6 + nload + i] = +numpy.inf

            for i in range(nevop):  # Pevcur
                idx = mnode[cevop[i]]
                xcon_lower[NXcon7 + i] = 0.0
                xcon_upper[NXcon7 + i] = pevel_aux[idx] * Evel_real_aux[idx][m]

        # print('xcon_lower:\n', xcon_lower[NXcon5: NXcon])
        # print('xcon_upper:\n', xcon_upper[NXcon5: NXcon])

        xcon = picos.RealVariable('xcon', NXcon, lower=xcon_lower, upper=xcon_upper)

        # //////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # print('Operation model-Iteratiion: Define objectices')

        ObjectOp_sls = numpy.zeros(NXcon)
        ObjectOp_dga = numpy.zeros(NXcon)
        ObjectOp_lda = numpy.zeros(NXcon)
        ObjectOp_eva = numpy.zeros(NXcon)

        ObjectOp_bas = numpy.zeros(NXcon)
        ObjectOp_vas = numpy.zeros(NXcon)

        for i in range(nline):
            ObjectOp_sls[nline * 2 + i] = price_ele[p] * rline[i] * Sbase  # Rij *Iij
        for i in range(nshop):
            ObjectOp_sls[NXcon4 + nshop * 2 + i] = price_ele[p] * Sbase  # MSopt

        if DgLdEvCur:
            samll_factor = 0.01
            for i in range(ndgop):
                ObjectOp_dga[NXcon5 + i] = price_dga[p] * Sbase * samll_factor
            for i in range(nload):
                ObjectOp_lda[NXcon6 + i] = price_lda[p] * Sbase * samll_factor
            for i in range(nevop):
                ObjectOp_eva[NXcon7 + i] = price_eva[p] * Sbase * samll_factor

        if BavaSign:
            for i in range(nline):
                ObjectOp_bas[NXcon8 + i] = price_bas  # Aij
        if BavaSign:
            for i in range(nnode):
                ObjectOp_vas[NXcon9 + i] = price_vas  # Bi

        Picos.minimize = (ObjectOp_sls + ObjectOp_dga + ObjectOp_lda + ObjectOp_eva +
                          ObjectOp_bas + ObjectOp_vas) * xcon

        # //////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # print('Operation model-Iteratiion: Write constraints')

        # //////////////////////////////////////////////////////////////
        # Pfneq
        for i in range(ndgop):
            idxa = NXcon5 + i  # Pdgcur
            idxb = NXcon5 + ndgop + i  # Qdgcur

            finit = fdgen[delta_idxn[i]]
            tanpf = numpy.sqrt(1 - numpy.square(finit)) / finit
            Picos += xcon[idxb] == xcon[idxa] * tanpf  # 无功按功率因数不可调

        # Pfnld
        for i in range(nload):
            idxa = NXcon6 + i  # Pldcur
            idxb = NXcon6 + nload + i  # Qldcur

            idx = mnode[cload[i]]
            tanpf = qload_aux[p, idx] / pload_aux[p, idx]
            Picos += xcon[idxb] == xcon[idxa] * tanpf

        # Pcveq Qcveq
        for i in range(nshop):
            idxa = NXcon4 + i  # Psopt
            idxb = NXcon4 + nshop + i  # Qsopt
            idxc = NXcon4 + nshop * 3 + i  # Ssopt

            Picos += -xcon[idxc] <= xcon[idxa] <= xcon[idxc]
            Picos += -xcon[idxc] <= xcon[idxb] <= xcon[idxc]

        # //////////////////////////////////////////////////////////////
        if nnode > 0:
            Picos += Pineq * xcon == pload_aux[p] * Load_real_aux[:, m] - pdgen_aux * Dgen_real_aux[:, m] + pevel_aux * Evel_real_aux[:, m]
            Picos += Qineq * xcon == qload_aux[p] * Load_real_aux[:, m] - qdgen_aux * Dgen_real_aux[:, m]
        if nline > 0:
            Picos += Uineq * xcon == 0.0
            Picos += Uiaeq * xcon == 0.0
            Picos += Uibeq * xcon == 0.0
        if nshmp > 0:
            Picos += Pisop * xcon == 0.0
        if nline > 0:
            if BavaSign:
                Picos += Alzeq * xcon >= -Imaxlit_bas ** 2
        if nnode > 0:
            if BavaSign:
                Picos += Blzeq * xcon >= -Vmaxlit_vas ** 2
                Picos += Blseq * xcon >= Vminlit_vas ** 2

        # //////////////////////////////////////////////////////////////
        for i in range(nline):  # (UIb)2≥(Pij)2+(Qij)2+(UIa)2
            idxa = i  # Pij
            idxb = nline + i  # Qij
            idxc = NXcon2 + i  # UIa
            idxd = NXcon2 + nline + i  # UIb
            Picos += abs(xcon[[idxa, idxb, idxc]]) <= xcon[idxd]

        for i in range(nshop):  # (Ssopt)2≥(Psopt)2+(Qsopt)2
            idxa = NXcon4 + i  # Psopt
            idxb = NXcon4 + nshop + i  # Qsopt
            idxc = NXcon4 + nshop * 3 + i  # Ssopt
            Picos += abs(xcon[[idxa, idxb]]) <= xcon[idxc]

        for i in range(nshop):  # (Msopt)2≥(Psopt)2+(Qsopt)2
            idxa = NXcon4 + i  # Psopt
            idxb = NXcon4 + nshop + i  # Qsopt
            idxc = NXcon4 + nshop * 2 + i  # Msopt
            Picos += abs(xcon[[idxa, idxb]]) * floss_cvt <= xcon[idxc]

        # //////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # print('Operation model-Iteratiion: Solve problem')
        # print(Picos)

        try:
            solution = Picos.solve(solver=Solver)
        except:
            solution = Picos.solve(solver=Solver, primals=None)
            print('{}th iteration with Solver={}, Status={}.'.format(m, Solver, solution.claimedStatus))
        # print('p={}, m={}, Solver={}, Status={}, Time={}'.format(p, m, Solver, solution.claimedStatus, solution.searchTime))

        # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # print('Planning model: Extract results')

        xsol_con = numpy.array(xcon.value).flatten()

        # //////////////////////////////////////////////////////////////
        # Unode
        for i in range(nnode):
            Unode[i][m] = numpy.sqrt(xsol_con[NXcon1 + i])  # Ui/Uj (p.u.)

        # Iline
        for i in range(nline):
            Iline[i][m] = numpy.sqrt(xsol_con[nline * 2 + i])  # Iij (p.u.)

        # Psopt Qsopt Msopt
        for i in range(nshop):
            Psopt[i] = xsol_con[NXcon4 + i]
            Qsopt[i] = xsol_con[NXcon4 + nshop * 1 + i]
            Msopt[i] = xsol_con[NXcon4 + nshop * 2 + i]

        # Gapps
        for i, schm in enumerate(schme_op):
            schm_range = range(sum(nscht_op[:i]), sum(nscht_op[:i+1]))

            schm_Psopt = numpy.sum(Psopt[schm_range])
            schm_Qsopt = numpy.sum(Qsopt[schm_range])
            schm_Msopt = numpy.sum(Msopt[schm_range])
            Gapps[i] = schm_Psopt - schm_Msopt

        # Gapxy
        for i in range(nline):
            for j in range(nnode):
                if bgbus[i] == inode[j]:
                    Gapxy[i] = xsol_con[nline * 2 + i] * xsol_con[NXcon1 + j] - numpy.square(xsol_con[i]) - numpy.square(xsol_con[nline + i])

        # Gapls
        for i in range(nshop):
            Gapls[i] = numpy.square(Msopt[i]) - numpy.square(floss_cvt) * (numpy.square(Psopt[i]) + numpy.square(Qsopt[i]))

        # //////////////////////////////////////////////////////////////
        # CostOpr_sls
        interval_line = range(nline * 2, nline * 3)
        interval_sopp = range(NXcon4 + nshop * 2, NXcon4 + nshop * 3)
        CostOpr_lineloss += numpy.sum(xsol_con[interval_line] * rline)
        CostOpr_sopploss += numpy.sum(xsol_con[interval_sopp])

        # CostOpr_dga
        interval_dga = range(NXcon5, NXcon5 + ndgop)
        CostOpr_dga += numpy.sum(xsol_con[interval_dga])

        # CostOpr_lda
        interval_lda = range(NXcon6, NXcon6 + nload)
        CostOpr_lda += numpy.sum(xsol_con[interval_lda])

        # CostOpr_eva
        interval_eva = range(NXcon7, NXcon)
        CostOpr_eva += numpy.sum(xsol_con[interval_eva])

        # CostOpr_bas (revise)
        dsequence = list(range(line_nlad))
        for d in dsequence:
            Imaxlit_cal = Imaxlit_lad[d]

            comp_res = numpy.where(Iline[:, m] > Imaxlit_cal)
            for start_node in edbus[comp_res]:
                visited = set()
                queue = deque([start_node])
                while queue:
                    current_node = queue.popleft()
                    if current_node not in visited:
                        visited.add(current_node)
                        for i in range(nline):
                            if current_node == bgbus[i]:
                                neighbor = edbus[i]
                                if neighbor not in visited:
                                    queue.append(neighbor)

                for cn in visited:
                    idx_cn = mnode[cn]
                    CostOpr_bas += pload_aux[p][idx_cn] * Load_real_aux[idx_cn, m]

        # CostOpr_vas (revise)
        dsequence = list(range(volt_nlad))
        for d in dsequence:
            Vminlit_cal = Vminlit_lad[d]
            Vmaxlit_cal = Vmaxlit_lad[d]

            comp_res = numpy.where((Unode[:, m] < Vminlit_cal) | (Unode[:, m] > Vmaxlit_cal))
            CostOpr_vas += numpy.sum(pload_aux[p][comp_res] * Load_real_aux[comp_res, m])

        Gapxymax = 0.0
        Gaplsmax = 0.0
        Gappsmax = 0.0
        if nline: Gapxymax = numpy.max(Gapxy)
        if nshop: Gaplsmax = numpy.max(Gapls)
        if nshop: Gappsmax = numpy.max(Gapps)

        if Gapxymax > 1e-3 or Gaplsmax > 1e-3 or Gappsmax > 1e-3:
            print('{}th iteration cannot converge with Gapxymax={}, Gaplsmax={}, Gappsmax={}.'.format(m, Gapxymax, Gaplsmax, Gappsmax))

    CostOpr_lineloss *= zeta_hours * price_ele[p] * Sbase
    CostOpr_sopploss *= zeta_hours * price_ele[p] * Sbase
    CostOpr_sls = CostOpr_lineloss + CostOpr_sopploss

    CostOpr_dga *= zeta_hours * price_dga[p] * Sbase
    CostOpr_lda *= zeta_hours * price_lda[p] * Sbase
    CostOpr_eva *= zeta_hours * price_eva[p] * Sbase

    CostOpr_bas *= zeta_hours * price_ele[p] * Sbase
    CostOpr_vas *= zeta_hours * price_ele[p] * Sbase

    # //////////////////////////////////////////////////////////////
    violate_vmi = numpy.zeros((volt_nlad, nnode))
    violate_vmx = numpy.zeros((volt_nlad, nnode))
    violate_lmx = numpy.zeros((line_nlad, nline))

    dsequence = list(range(volt_nlad))
    for d in dsequence:
        violate_vmi[d] = numpy.sum(Unode < Vminlit_lad[d], axis=1) / ndata
        violate_vmx[d] = numpy.sum(Unode > Vmaxlit_lad[d], axis=1) / ndata

    dsequence = list(range(line_nlad))
    for d in dsequence:
        violate_lmx[d] = numpy.sum(Iline.transpose() > Imaxlit_lad[d], axis=0) / ndata

    Rptplot = {'volt': Unode, 'line': Iline,
               'CostOpr_sls': CostOpr_sls, 'CostOpr_dga': CostOpr_dga, 'CostOpr_lda': CostOpr_lda,
               'CostOpr_eva': CostOpr_eva, 'CostOpr_bas': CostOpr_bas, 'CostOpr_vas': CostOpr_vas,
               'violate_vmi': violate_vmi, 'violate_vmx': violate_vmx, 'violate_lmx': violate_lmx}

    pchild.send(Rptplot)
