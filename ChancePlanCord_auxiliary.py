import numpy
import copy
import sys
import math
from collections.abc import Iterable
from collections import Counter


'''
Auxiliary functions

'''

def printCheckMatrix(matrix, row, col, title):
    print(title + '***********************')
    for i in range(row):
        for j in range(col):
            if matrix[i][j] != 0:
                print(matrix[i][j], '(', i, ',', j, ') ', sep='', end='')
        print('\n')
    print('\n')


def printCheckVector(vector, vlen, title):
    print(title + '***********************')
    for i in range(vlen):
        if vector[i] != 0:
            print(vector[i], '(', i, ') ', sep='', end='')
    print('\n')


def roundFloat(x, dec=3):
    xx = float(format(x, '.{}f'.format(dec)))
    y = xx if xx else 0.0

    return y


def findMedian(data):
    data_sort = numpy.sort(data)
    half = len(data_sort) // 2
    median = (data_sort[half] + data_sort[~half])/2

    return median


def computeValidMean(data):
    exist = (data != 0)
    data_valid_mean = data.sum() / exist.sum()

    return data_valid_mean


def ceilDecimal(data, maxval, decml=2):
    pow_decml = math.pow(10, decml)
    thr_decml = math.pow(10, -(decml+1))

    data_filter = numpy.where(data < thr_decml, 0.0, data)
    data_ceiled = numpy.ceil(data_filter * pow_decml) / pow_decml

    data_limitd = numpy.where(data_ceiled > maxval, maxval, data_ceiled)

    return data_limitd


def flattenList(items, ignore_types=(str, bytes)):
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, ignore_types):
            yield from flattenList(x)
        else:
            yield x


def delList(mlist, n):
    mlist_copy = copy.deepcopy(mlist)
    del mlist_copy[n]
    return mlist_copy


def swapPosition(mlist, p):
    indx = list(range(len(mlist)))
    del indx[p]

    rest_value = [mlist[i] for i in indx]
    nlist = [mlist[p]] + rest_value
    return nlist


def prepareContour(data):
    nx = data.shape[0]
    ny = data.shape[1]

    xloc = numpy.tile(numpy.arange(nx).reshape(-1, 1), ny)
    yloc = numpy.round(data, 3)
    hval = numpy.zeros_like(data)

    for i in range(nx):
        ylocs = yloc[i]
        count_dict = dict(Counter(ylocs))
        hval[i] = [count_dict[t] / ny for t in ylocs]

    return xloc, yloc, hval


def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def savedictny(path, dict):
    f = open(path, 'w')
    f.write(str(dict))
    f.close()


def readdictny(path):
    f = open(path, 'r')
    a = f.read()
    dict = eval(a)
    f.close()

    return dict
