import numpy

'''
Compute and record the violation probabilities of the flexible distribution network in different planning schemes

'''


from data_parameters import nstage


def saveStrategyViolateProb(stgys, violate_vmi, violate_vmx, violate_lmx):
    # stgys-strategy identifier characters
    # violate_vmi-probability of nodal voltage crossing the lower limit
    # violate_vmx-probability of nodal voltage crossing the upper limit
    # violate_lmx-probability of line current crossing the upper limit

    for i in range(nstage):
        violate_vmi_max = numpy.max(violate_vmi[i], axis=1)
        violate_vmi_mag = numpy.argmax(violate_vmi[i], axis=1)

        violate_vmx_max = numpy.max(violate_vmx[i], axis=1)
        violate_vmx_mag = numpy.argmax(violate_vmx[i], axis=1)

        violate_lmx_max = numpy.max(violate_lmx[i], axis=1)
        violate_lmx_mag = numpy.argmax(violate_lmx[i], axis=1)

        print('stage {}: maximum violate of volt_min {} at index {}'.format(i, violate_vmi_max, violate_vmi_mag))
        print('stage {}: maximum violate of volt_max {} at index {}'.format(i, violate_vmx_max, violate_vmx_mag))
        print('stage {}: maximum violate of line_max {} at index {}'.format(i, violate_lmx_max, violate_lmx_mag))
        print('')

    numpy.save('\\violate_volt_min_appl_{}.npy'.format(stgys), violate_vmi)
    numpy.save('\\violate_volt_max_appl_{}.npy'.format(stgys), violate_vmx)
    numpy.save('\\violate_line_max_appl_{}.npy'.format(stgys), violate_lmx)


