import numpy

'''
Compute the output of distributed generators

'''


def WindSpeedtoPower(speed, vmin, vmax, vn):
    # speed-real-time wind speed (m/s)
    # vmin-cut-in wind speed (m/s)
    # vmax-cut-out wind speed (m/s)
    # vn-rated wind speed (m/s)
    # WindPower-profile of wind output (p.u.)

    # WindPower = 1.0
    # if speed < vmin or speed > vmax:
    #     WindPower = 0
    # elif speed >= vmin and speed <= vn:
    #     WindPower = 1.0 * (speed-vmin) / (vn-vmin)
    # elif speed > vn and speed <= vmax:
    #     WindPower = 1.0

    if not isinstance(speed, numpy.ndarray):
        speed = numpy.array(speed)

    WindPower = numpy.ones_like(speed)
    WindPower = numpy.where((speed < vmin) | (speed > vmax), 0.0, WindPower)
    WindPower = numpy.where((speed >= vmin) & (speed <= vn), 1.0*(speed-vmin)/(vn-vmin), WindPower)
    WindPower = numpy.where((speed > vn) & (speed <= vmax), 1.0, WindPower)

    return WindPower


def SolaradiatoPower(radia, refactor, squarekw):
    # radia-light intensity
    # refactor-photoelectric conversion efficiency
    # squarekw-photovoltaic installed capacity in unit square meter
    # SolaPower-profile of photovoltaic (p.u.)

    if not isinstance(radia, numpy.ndarray):
        radia = numpy.array(radia)

    SolaPower = radia * refactor / squarekw

    return SolaPower


def limitGeneratorOutput(output):
    if not isinstance(output, numpy.ndarray):
        output = numpy.array(output)

    output = numpy.where(output > 1.0, 1.0, output)
    output = numpy.where(output < 0.0, 0.0, output)

    return output

