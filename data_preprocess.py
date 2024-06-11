import math
import pandas
import scipy.optimize
import numpy
from itertools import combinations

'''
Preprocess the distribution netwrok data

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

from data_parameters import Vminlit_sec, Vmaxlit_sec, Imaxlit_sec, Vminlit_vas, Vmaxlit_vas, Imaxlit_bas, \
    smini_cvt, smaxi_cvt, floss_cvt, decml_cap, modfy_geo, Irate, \
    price_cvt, price_sit, price_ref, price_trs, price_ele, price_dga, price_lda, price_eva, price_vas, price_bas, \
    price_exp, Rxper_exp, Ctper_exp, \
    nstage, nyear, nhour, infla, epsilon, \
    load_increas, dgen_penetrn, evel_penetrn, pcu_max, pty_per, Nevcs_max, Ndgen_max, \
    Vrisk, Irisk, volt_nlad, line_nlad, Vminlit_lad, Vmaxlit_lad, Imaxlit_lad

# ======================================================================================================================
nliNm = {'Node': nnode, 'Line': nline, 'Ties': nties, 'Slck': nslck}
lwsNm = {'Load': 0, 'Evel': 0, 'Wind': 0, 'Sola': 0}

lwsNm['Load'] = nload  # number of conventional loads in each stage
lwsNm['Evel'] = nevcs  # number of electric vehicle charging loads in each stage
lwsNm['Sola'] = len(numpy.where(tdgen == 1)[0])  # number of photovoltaics in each stage
lwsNm['Wind'] = len(numpy.where(tdgen == 2)[0])  # number of wind turbines in each stage

dimen = sum(lwsNm.values())

# ======================================================================================================================
# Conventional loads in each stage
pload_stg = numpy.zeros((nstage, nload))  # ->pload_aux
qload_stg = numpy.zeros((nstage, nload))  # ->qload_aux

for i in range(nstage):
    if i == 0:
        pload_stg[i] = pload * math.pow(load_increas[i], nyear)
        qload_stg[i] = qload * math.pow(load_increas[i], nyear)
    else:
        pload_stg[i] = pload_stg[i - 1] * math.pow(load_increas[i], nyear)
        qload_stg[i] = qload_stg[i - 1] * math.pow(load_increas[i], nyear)

# Electric vehicle charging loads in each stage
pevel_stg = numpy.zeros(nstage)
for i in range(nstage):
    factor = evel_penetrn[i] / (1 - evel_penetrn[i])
    pevel_stg[i] = sum(pload_stg[i]) * factor

# Distributed generators in each stage
pdgen_stg = numpy.zeros(nstage)
for i in range(nstage):
    factor = dgen_penetrn[i]
    pdgen_stg[i] = (sum(pload_stg[i]) + pevel_stg[i]) * factor

# ======================================================================================================================
pload_init_aux = numpy.zeros(nnode)
qload_init_aux = numpy.zeros(nnode)
for i in range(nload):
    pload_init_aux[mnode[cload[i]]] = pload[i]
    qload_init_aux[mnode[cload[i]]] = qload[i]

pload_aux = numpy.zeros((nstage, nnode))
qload_aux = numpy.zeros((nstage, nnode))
for i in range(nstage):
    for k in range(nload):
        pload_aux[i][mnode[cload[k]]] = pload_stg[i][k]
        qload_aux[i][mnode[cload[k]]] = qload_stg[i][k]
# print(pload_aux)
# print(qload_aux)

load_pvt = numpy.zeros(nstage)
for i in range(nstage):
    load_pvt[i] = sum(pload_stg[i]) + pevel_stg[i]
# print(load_pvt)

CasePentInfro = pandas.DataFrame({'Ptr_evel': pevel_stg / load_pvt, 'Pevel': pevel_stg,
                                  'Sevcs': [sum(sorted(sevcs, reverse=True)[:Nevcs_max[i]]) for i in range(nstage)],
                                  'Ptr_dgen': pdgen_stg / load_pvt, 'Pdgen': pdgen_stg,
                                  'Sdgen': [sum(sorted(sdgen*fdgen, reverse=True)[:Ndgen_max[i]]) for i in range(nstage)]},
                                 index=['CasePent_Stage_{}'.format(i) for i in range(nstage)])
print(CasePentInfro, '\n')

# ======================================================================================================================
# Connection relationship between lines and nodes
conctnum = numpy.zeros((nnode, 1), dtype=int)
for i in range(nnode):
    for j in range(nline):
        if inode[i] == bgbus[j] or inode[i] == edbus[j]:
            conctnum[i] = conctnum[i] + 1

conctmax = conctnum[0]
for i in range(nnode):
    if conctnum[i] > conctmax:
        conctmax = conctnum[i]

connect_id = numpy.zeros((nnode, int(conctmax)), dtype=int)
connect_sr = numpy.zeros((nnode, int(conctmax)), dtype=int)

for i in range(nnode):
    k = 0
    for j in range(nline):
        if inode[i] == bgbus[j]:
            connect_id[i][k] = iline[j]
            connect_sr[i][k] = j
            k = k + 1
        if inode[i] == edbus[j]:
            connect_id[i][k] = -iline[j]
            connect_sr[i][k] = j
            k = k + 1
# print('connect_id:\n', connect_id)
# print('connect_sr:\n', connect_sr)
# exit(0)

# ======================================================================================================================
# All accessible nodes for SOP
portAccess = numpy.concatenate((ipnode, bgtiebus, edtiebus))
iport = numpy.array(sorted(list(set(portAccess))))
nport = len(iport)
# print(iport)

mport = dict()
for i, port_id in enumerate(iport):
    mport[port_id] = i
# print(mport)
# exit()

# ======================================================================================================================
# Construction of SOP planning schemes
minterminal = 2
maxterminal = 4

schNm = dict()
nscht = list()
scheme = list()

for i in range(minterminal, maxterminal+1):
    schNm['{}_port'.format(i)] = 0

    for schm in combinations(iport, i):
        schm_zone = list()

        for schm_node in schm:
            node_zone = znode[mnode[schm_node]]
            schm_zone.append(node_zone)

        if len(set(schm_zone)) == len(schm_zone):
            nscht.append(i)
            schNm['{}_port'.format(i)] += 1
            scheme.append(schm)
nschp = sum(nscht)
nschm = len(scheme)
schNm['total'] = nschm

# ======================================================================================================================
# Line construction length of SOP planning schemes
dist_method = 'Manhattan'  # Manhattan distance/ Euclidean distance
lschm = numpy.zeros(nschm)
pschm = numpy.zeros((nschm, 2))
dschm = numpy.zeros(nschm, dtype=int)

for i, schm in enumerate(scheme):
    # ==================================================================================================================
    # Not considering existing connecting lines
    teml_idx = numpy.zeros(len(schm), dtype=int)

    for k, schm_node in enumerate(schm):
        teml_idx[k] = mteml[schm_node]

    teml_xloc = xteml[teml_idx]
    teml_yloc = yteml[teml_idx]

    if dist_method == 'Euclidean':
        find_pos = lambda pos: numpy.sum(numpy.sqrt(numpy.square(pos[0] - teml_xloc) + numpy.square(pos[1] - teml_yloc)))
    elif dist_method == 'Manhattan':
        find_pos = lambda pos: numpy.sum(numpy.absolute(pos[0] - teml_xloc) + numpy.absolute(pos[1] - teml_yloc))
    else:
        print('Not defined distance method')
        break

    init_pos = numpy.array([numpy.mean(teml_xloc), numpy.mean(teml_yloc)])

    res = scipy.optimize.minimize(find_pos, init_pos)
    opti_dist = res['fun']
    opti_pos = res['x']

    lschm[i] = opti_dist
    pschm[i] = opti_pos

    # ==================================================================================================================
    # Considering existing connecting lines
    for k in range(nties):
        for tiebus in (bgtiebus[k], edtiebus[k]):
            temp = 0.0
            teml_xloc = xteml[mteml[tiebus]]
            teml_yloc = yteml[mteml[tiebus]]

            extra_teml_idx = numpy.setdiff1d(schm, (bgtiebus[k], edtiebus[k]), assume_unique=True)

            for m in range(len(extra_teml_idx)):
                extra_teml_xloc = xteml[mteml[extra_teml_idx[m]]]
                extra_teml_yloc = yteml[mteml[extra_teml_idx[m]]]

                if dist_method == 'Euclidean':
                    dist_part = numpy.sqrt(numpy.square(extra_teml_xloc - teml_xloc) + numpy.square(extra_teml_yloc - teml_yloc))
                elif dist_method == 'Manhattan':
                    dist_part = numpy.absolute(extra_teml_xloc - teml_xloc) + numpy.absolute(extra_teml_yloc - teml_yloc)
                else:
                    print('Not defined distance method')
                    dist_part = 0.0

                temp += dist_part

            if temp <= lschm[i]:
                lschm[i] = temp
                pschm[i] = [teml_xloc, teml_yloc]
                dschm[i] = tiebus

# scheme_doc = dict()
# scheme_doc['scheme'] = scheme
# scheme_doc['nterminal'] = [len(i) for i in scheme]
# scheme_doc['length'] = lschm
# scheme_doc = pandas.DataFrame(scheme_doc)
# print(scheme_doc)
# exit()

# ===================================================================================
scheme_evolve = dict()
scheme_set = [set(i) for i in scheme]

for i, schm in enumerate(scheme):
    evolve_sign = [xx.issuperset(schm) for xx in scheme_set]
    evolve_idxn = numpy.nonzero(evolve_sign)[0]
    scheme_evolve[i] = evolve_idxn
    # print([scheme[i] for i in evolve_idxn])
    # exit()

# ===================================================================================
scheme_contain = dict()
for i, port_node in enumerate(iport):
    contain_sign = [port_node in xx for xx in scheme_set]
    contain_idxn = numpy.nonzero(contain_sign)[0]
    scheme_contain[i] = contain_idxn
    # print([scheme[i] for i in contain_idxn])
    # exit()
