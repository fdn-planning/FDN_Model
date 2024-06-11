import numpy
import pandas

'''
Compute and record the cost benefits of different planning schemes

'''


from data_construction import Cbase
from data_parameters import nstage
from ChancePlanCord_auxiliary import savedictny


def saveStrategyCostBenefit(stgys, ycost_expt, yvolt_appl, yline_appl, Cinvt_appl, Coper_appl):
    # stgys-strategy identifier characters
    # ycost-planning cost calculated in the optimization planning model
    # yvolt-voltage results
    # yline-current results
    # Cinvt-line expansion cost calculated in the optimization testing model
    # Coper-operational costs calculated in the optimization testing model

    for keys in Cinvt_appl:
        Cinvt_appl[keys] = numpy.around(Cinvt_appl[keys] / Cbase, decimals=2)
        Cinvt_appl[keys] = list(Cinvt_appl[keys])
    for keys in Coper_appl:
        Coper_appl[keys] = numpy.around(Coper_appl[keys] / Cbase, decimals=2)
        Coper_appl[keys] = list(Coper_appl[keys])

    numpy.save('\\yvolt_appl_{}.npy'.format(stgys), yvolt_appl)
    numpy.save('\\yline_appl_{}.npy'.format(stgys), yline_appl)
    savedictny('\\Cinvt_appl_{}.txt'.format(stgys), Cinvt_appl)  # dict
    savedictny('\\Coper_appl_{}.txt'.format(stgys), Coper_appl)  # dict

    # //////////////////////////////////////////////////////////////
    Dc_Cost_appl_detail = {'pal': Cinvt_appl['pal'], 'trs': Cinvt_appl['trs'],
                           'sit': ycost_expt['sit'], 'sop': ycost_expt['sop'], 'ref': ycost_expt['ref'],
                           'sev': ycost_expt['sev'], 'evs': ycost_expt['evs'], 'dgc': ycost_expt['dgc'],
                           'sls': Coper_appl['sls'], 'dga': Coper_appl['dga'], 'lda': Coper_appl['lda'],
                           'eva': Coper_appl['eva'], 'bas': Coper_appl['bas'], 'vas': Coper_appl['vas']}
    Df_index = ['stage{}'.format(i + 1) for i in range(nstage)]
    Df_Cost_appl_detail = pandas.DataFrame(Dc_Cost_appl_detail, index=Df_index)
    Df_Cost_appl_detail['rsum'] = Df_Cost_appl_detail.apply(lambda x: x.sum(), axis=1)
    Df_Cost_appl_detail.loc['csum'] = Df_Cost_appl_detail.apply(lambda x: x.sum())
    print('Df_Cost_appl_detail:\n', Df_Cost_appl_detail, '\n')

    savedictny('\\Dc_Cost_appl_detail_{}.txt'.format(stgys), Dc_Cost_appl_detail)  # dict
    savedictny('\\Df_Cost_appl_detail_{}.txt'.format(stgys), Df_Cost_appl_detail)  # dataframe

    # //////////////////////////////////////////////////////////////
    Dc_Cost_appl_simply = {'exp': list(numpy.add.reduce([Cinvt_appl['pal'], Cinvt_appl['trs']])),
                           'sop': list(numpy.add.reduce([ycost_expt['sit'], ycost_expt['sop'], ycost_expt['ref']])),
                           'evs': list(numpy.add.reduce([ycost_expt['sev'], ycost_expt['evs']])),
                           'dgc': ycost_expt['dgc'],
                           'sls': Coper_appl['sls'],
                           # 'dga': Coper_appl['dga'],
                           # 'lda': list(numpy.add.reduce([Coper_appl['lda'], Coper_appl['eva']])),
                           'bas': Coper_appl['bas'],
                           'vas': Coper_appl['vas']}

    Df_Cost_appl_simply = pandas.DataFrame(Dc_Cost_appl_simply, index=Df_index)
    Df_Cost_appl_simply['rsum'] = Df_Cost_appl_simply.apply(lambda x: x.sum(), axis=1)
    Df_Cost_appl_simply.loc['csum'] = Df_Cost_appl_simply.apply(lambda x: x.sum())
    print('Df_Cost_appl_simply:\n', Df_Cost_appl_simply, '\n')

    savedictny('\\Dc_Cost_appl_simply_{}.txt'.format(stgys), Dc_Cost_appl_simply)  # dict
    savedictny('\\Df_Cost_appl_simply_{}.txt'.format(stgys), Df_Cost_appl_simply)  # dataframe


