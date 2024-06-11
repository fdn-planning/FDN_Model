import numpy
from collections import Counter

'''
Prepare the distribution netwrok data
Construct data structures

'''

node = numpy.array([
    # (1,     11.4,    0,  0,  'A')
], dtype=[('id', 'int'), ('rated_volt', 'float'), ('slack_sign', 'int'), ('teml_sign', 'int'), ('zone', 'U1')])

line = numpy.array([
    # (1,     1,      2,   0.2096,   0.4304,  1,   0)
], dtype=[('id', 'int'), ('bgbus', 'int'), ('edbus', 'int'), ('rline', 'float'), ('xline', 'float'), ('state', 'int'), ('boundary', 'int')])

ties = numpy.array([
    # (1,     7,    60,   0.1310,   0.2690)
], dtype=[('id', 'int'), ('bgbus', 'int'), ('edbus', 'int'), ('rline', 'float'), ('xline', 'float')])

xyts = numpy.array([
    # (1,     5,  10.53,    7.80)
], dtype=[('id', 'int'), ('cnbus', 'int'), ('xloc', 'float'), ('yloc', 'float')])

pall = numpy.array([
    # (1,   'A',   12.05)
],dtype=[('id', 'int'), ('zone', 'U1'), ('length', 'float')])

load = numpy.array([
    # (1,     2,       100,     50,   1)
], dtype=[('id', 'int'), ('cnbus', 'int'),  ('pload', 'float'), ('qload', 'float'), ('type', 'int')])

evcs = numpy.array([
    # (1,    28,    2000)
], dtype=[('id', 'int'), ('cnbus', 'int'),  ('smax', 'int')])

dgen = numpy.array([
    # (1,   10,   3000,    0.95,   1)
], dtype=[('id', 'int'), ('cnbus', 'int'),  ('smax', 'float'), ('pfactor', 'float'), ('type', 'int')])

stos = numpy.array([
    # (1, 18,  -500,  500,  1000,  10000,  0.5,  0.1,  0.9)
], dtype=[('id', 'int'), ('cnbus', 'int'),  ('pmin', 'float'), ('pmax', 'float'),
          ('scvt', 'float'), ('sbat', 'float'), ('soc_current', 'float'), ('soc_min', 'float'), ('soc_max', 'float')])

caps = numpy.array([
    # (1, 18,  100,  10)
], dtype=[('id', 'int'), ('cnbus', 'int'),  ('qunit', 'float'), ('bank', 'int')])

sops = numpy.array([
    # (1,  12, 22,  500,  500),
    # (2,  18, 33, 1000, 1000)
], dtype=[('id', 'int'), ('aport', 'int'),  ('bport', 'int'), ('asvct', 'float'), ('bsvct', 'float')])

# ======================================================================================================================
casedata = {'node': node, 'line': line, 'ties': ties,
            'load': load, 'evcs': evcs, 'dgen': dgen,
            'stos': stos, 'caps': caps, 'sops': sops}

# bus data
nnode = len(node)
inode = node['id']
vnode = node['rated_volt']
znode = node['zone']
ipnode = inode[numpy.argwhere(node['teml_sign'] == 1).flatten()]  # sop-port-bus
islck = inode[numpy.argwhere(node['slack_sign'] == 1).flatten()]  # slack-bus
nslck = len(islck)
znode_cnt = dict(Counter(znode))
znode_key = znode_cnt.keys()
znode_val = list(znode_cnt.values())
nzone = len(znode_cnt)

mnode = dict()
for i, node_id in enumerate(inode):
    mnode[node_id] = i

# base capacity
Cbase = 1e4  # 1e4 yuan
Sbase = 1e3  # kVA
Vbase = vnode[mnode[islck[0]]]
Ibase = Sbase / Vbase
Zbase = numpy.square(Vbase) * 1e3 / Sbase
Vstation = 1.00

# line data
nline = len(line)
iline = line['id']
bgbus = line['bgbus']
edbus = line['edbus']
rline = line['rline'] / Zbase
xline = line['xline'] / Zbase
sline = line['state']
zline = [znode[mnode[x]] for x in bgbus]
abond = numpy.argwhere(line['boundary'] == 1).flatten()
ibond = iline[abond]
nbond = len(ibond)
zbond = [znode[mnode[x]] for x in bgbus[abond]]
zline_cnt = dict(Counter(zline))
zline_key = zline_cnt.keys()
zline_val = list(zline_cnt.values())
# print(rline)
# print(xline)
# print(zbond)
# exit()

# ties data
nties = len(ties)
ities = ties['id']
bgtiebus = ties['bgbus']
edtiebus = ties['edbus']

# xyts data
nxyts = len(xyts)
ixyts = xyts['id']
cxyts = xyts['cnbus']
xteml = xyts['xloc']
yteml = xyts['yloc']

mteml = dict()
for i, teml_id in enumerate(cxyts):
    mteml[teml_id] = i
# print(mteml)
# exit()

# pall data
npall = len(pall)
ipall = pall['id']
zpall = pall['zone']
lpall = pall['length']

mpall = dict()
for zp, lp in zip(zpall, lpall):
    mpall[zp] = lp
# print(mpall)
# exit()

# load data
nload = len(load)
iload = load['id']
cload = load['cnbus']
pload = load['pload'] / Sbase
qload = load['qload'] / Sbase
tload = load['type']

# storage data
portAccess = numpy.concatenate((bgtiebus, edtiebus))
cstog = numpy.array(sorted(list(set(portAccess))))
nstog = len(cstog)
istog = numpy.arange(nstog) + 1

# evcs data
nevcs = len(evcs)
if evcs.size == 0:
    ievcs, cevcs, sevcs = [numpy.empty(0) for i in range(3)]
else:
    ievcs = evcs['id']
    cevcs = evcs['cnbus']
    sevcs = evcs['smax'] / Sbase  # (kVA->p.u.)

# dgen data
ndgen = len(dgen)
if dgen.size == 0:
    idgen, cdgen, sdgen, fdgen, tdgen = [numpy.empty(0) for i in range(5)]
else:
    idgen = dgen['id']
    cdgen = dgen['cnbus']
    sdgen = dgen['smax'] / Sbase  # (kVA->p.u.)
    fdgen = dgen['pfactor']
    tdgen = dgen['type']