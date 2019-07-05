#!/usr/bin/env python

import numpy as np
import pandas as pd

'''
Generate test sample.
'''
def main_two_gaussians(filename = 'data.h5'):
    # make input file
    N = 10000
    x = {}

    f = pd.HDFStore(filename, 'w')
    all_data = np.zeros(shape = (0, 2+2))
    signal = np.random.normal(loc = -1.0, scale = 0.5, size = (N, 2))
    bkg    = np.random.normal(loc =  1.0, scale = 0.5, size = (N, 2))
    data   = np.append(signal, bkg, axis = 0)
    data_t = np.append(np.ones(N), np.zeros(N))
    data_w = np.append(np.ones(N), np.ones(N))
    add_all_data = np.concatenate( (data_t[:,np.newaxis], data_w[:,np.newaxis], data), axis=1)
    all_data = np.concatenate((all_data, add_all_data), axis = 0)

    df = pd.DataFrame(all_data, columns = ['sig', 'w', 'A', 'B'])
    f.put('df', df, format = 'table', data_columns = True)

def main_rainbow(filename = 'data.h5'):
    # make input file
    N = 10000
    x = {}

    f = pd.HDFStore(filename, 'w')
    all_data = np.zeros(shape = (0, 2+2))
    signal = np.zeros( (N, 2) )
    signal[:,0] = np.random.uniform(low = -10.0, high = 10.0, size = (N,))
    signal[:,1] = -signal[:,0]**2 + 10.0 + np.random.normal(loc = 0.0, scale = 0.3, size = (N,))
    bkg = np.zeros( (N, 2) )
    bkg[:,0] = np.random.uniform(low = -10.0, high = 10.0, size = (N,))
    bkg[:,1] = -bkg[:,0]**2 + 9.0 + np.random.normal(loc = 0.0, scale = 0.3, size = (N,))
    data   = np.append(signal, bkg, axis = 0)
    data_t = np.append(np.ones(N), np.zeros(N))
    data_w = np.append(np.ones(N), np.ones(N))
    add_all_data = np.concatenate( (data_t[:,np.newaxis], data_w[:,np.newaxis], data), axis=1)
    all_data = np.concatenate((all_data, add_all_data), axis = 0)

    df = pd.DataFrame(all_data, columns = ['sig', 'w', 'A', 'B'])
    f.put('df', df, format = 'table', data_columns = True)

if __name__ == '__main__':
    main_rainbow()
