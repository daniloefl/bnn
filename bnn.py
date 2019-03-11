#!/usr/bin/env python3

# to be able to run this:
# sudo apt-get install python3 python3-pip
# pip3 install --user matplotlib seaborn numpy h5py tensorflow edward

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import gc

import matplotlib as mpl
mpl.use('Agg')

# numerical library
import numpy as np
import h5py

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed

class BNN(object):
  '''
  Implementation of a test BNN.
  '''

  def __init__(self, n_iteration = 1050,
               n_batch = 32,
               n_eval = 50,
               n_posterior = 1000,
               variational_inference = False):
    '''
    Initialise the network.

    :param n_iteration: Number of batches to run over in total.
    :param n_batch: Number of samples in a batch.
    :param n_eval: Number of batches to train before evaluating metrics.
    :param variational_inference: Whether to use variational inference instead of MCMC sampling.
    '''
    self.n_iteration = n_iteration
    self.n_batch = n_batch
    self.n_eval = n_eval
    self.n_posterior = n_posterior
    self.variational_inference = variational_inference
    self.var = {} # variables
    self.session = None

    self.input = None

    self.posterior = {} # posterior

  def add_prior(self, dim):
    self.count += 1
    dim_prev = self.n_dimensions
    if self.count > 1:
      dim_prev = self.var['b_%d' % (self.count-1)].shape[-1]
    W = ed.Normal(loc = tf.zeros( (dim_prev, dim) ), scale = tf.ones( (dim_prev, dim) ), name = "W_%d" % self.count)
    b = ed.Normal(loc = tf.zeros( (dim) ), scale = tf.ones( (dim) ), name = "b_%d" % self.count)
    self.var['W_%d' % self.count] = W
    self.var['b_%d' % self.count] = b

  '''
    Calculate NN value.
  '''
  def nn(self, x):
    for i in range(1, self.count+1):
      W = self.var['W_%d' % i]
      b = self.var['b_%d' % i]
      x = tf.matmul(x, W, name = 'prod_%d' % i) + b
      if i < self.count:
        x = tf.nn.leaky_relu(x, alpha = 0.2, name = 'act_%d' % i)
      else:
        x = tf.nn.sigmoid(x, name = 'act_%d' % i)
    return x

  def model(self, x):
    return ed.Normal(loc = self.nn(x), scale = tf.constant(1), name = 'out')

  def target_log_prob_fn(self, *param):
    kwargs = {}
    c = 0
    for i in sorted(self.var.keys()):
      kwargs[i] = param[c]
      c += 1
    log_prob = 0
    for x,w,y in self.get_batch():
      log_prob += self.log_joint(x, out = y, **kwargs)
    return log_prob

  '''
    Create network.
  '''
  def create_model(self):
    if not self.session:
      self.session = tf.Session()
    with self.session.as_default():
      with tf.variable_scope('model'):
        if not self.input:
          self.input = tf.placeholder(tf.float32, shape = (None, self.n_dimensions), name = 'input')
        if len(self.var) == 0:
          self.count = 0
          self.add_prior(10)
          self.add_prior(1)
        self.log_joint = ed.make_log_joint_fn(self.model)
      #with tf.variable_scope('posterior'):
      #  # for variational inference
      #  if self.variational_inference:
      #    for i in self.var:
      #      name = "q%s" % i
      #      with tf.variable_scope(name):
      #        loc = tf.get_variable("loc", shape = self.var[i].shape)
      #        scale = tf.nn.softplus(tf.get_variable("scale", shape = self.var[i].shape))
      #        self.posterior[name] = ed.models.Normal(loc = loc, scale = scale)
      #  else:
      #    for i in self.var:
      #      name = "q%s" % i
      #      with tf.variable_scope(name):
      #        shape = [N] + self.var[i].shape[:]
      #        self.posterior[name] = ed.models.Empirical(params = tf.zeros(shape))
      self.session.run(tf.global_variables_initializer())

  '''
    Read input from file.
  '''
  def read_input_from_files(self, filename = 'input_preprocessed.h5'):
    self.file = h5py.File(filename)
    self.n_dimensions = self.file['train'].shape[1]-2
    self.col_signal = 0
    self.col_weight = 1
    self.col_data = 2

  '''
  Generate test sample.
  :param adjust_signal_weights: If True, weights the signal by the ratio of signal to background weights, so that the training considers both equally.
  '''
  def prepare_input(self, filename = 'input_preprocessed.h5', adjust_signal_weights = True, set_unit_weights = True):
    # make input file
    N = 10000
    self.file = h5py.File(filename, 'w')
    x = {}
    for t in ['train', 'test']:
      all_data = np.zeros(shape = (0, 2+2))
      signal = np.random.normal(loc = -1.0, scale = 0.5, size = (N, 2)).astype(np.float32)
      bkg    = np.random.normal(loc =  1.0, scale = 0.5, size = (N, 2)).astype(np.float32)
      data   = np.append(signal, bkg, axis = 0)
      data_t = np.append(np.ones(N, dtype = np.float32), np.zeros(N, dtype = np.float32))
      data_w = np.append(np.ones(N, dtype = np.float32), np.ones(N, dtype = np.float32))
      add_all_data = np.concatenate( (data_t[:,np.newaxis], data_w[:,np.newaxis], data), axis=1)
      all_data = np.concatenate((all_data, add_all_data), axis = 0)
      print('Checking nans in %s' % t)
      self.check_nans(all_data)
      self.file.create_dataset(t, data = all_data)
      #self.file[t].attrs['columns'] = ['signal', 'weight', '0', '1']

      signal = all_data[:, 0] == 1
      bkg = all_data[:, 0] == 0
      self.file.create_dataset('%s_%s' % (t, 'bkg'), data = bkg)
      self.file.create_dataset('%s_%s' % (t, 'signal'), data = signal)

    self.file.close()


  def check_nans(self, x):
    print("Dump of NaNs:")
    nan_idx = np.where(np.isnan(x))
    print(x[nan_idx])

    assert len(x[nan_idx]) == 0

  def plot_input_correlations(self, filename):
    import matplotlib.pyplot as plt
    import seaborn as sns

    x = self.file['test'][:, self.col_data:]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    sns.heatmap(np.corrcoef(x, rowvar = 0),
                cmap="YlGnBu", cbar = True, linewidths=.5, square = True,
                xticklabels = np.arange(0, x.shape[1]), yticklabels = np.arange(0, x.shape[1]),
                annot=True, fmt=".2f")
    ax.set(xlabel = '', ylabel = '', title = 'Correlation between input variables');
    plt.savefig(filename)
    plt.close("all")

  def plot_scatter_input(self, var1, var2, filename):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    x = self.file['test'][:, (self.col_data+var1, self.col_data+var2)]
    y = self.file['test'][:, self.col_signal]
    g = sns.scatterplot(x = x[:, 0], y = x[:, 1], hue = y,
                        hue_order = [0, 1], markers = ["^", "v"], legend = "brief", ax = ax)
    g.axes.get_legend().texts[0] = "Background"
    g.axes.get_legend().texts[1] = "Signal"
    ax.set(xlabel = var1, ylabel = var2, title = 'Scatter plot')
    plt.savefig(filename)
    plt.close("all")

  def plot_discriminator_output(self, filename):
    import matplotlib.pyplot as plt
    import seaborn as sns
    out_signal = {}
    out_bkg = {}
    N = 5
    nnout = self.nn(self.input)
    for i in range(N):
      out_signal[i] = []
      out_bkg[i] = []
      for x,w,y in self.get_batch(origin = 'test', signal = True):
        out_signal[i].extend(self.session.run(nnout, {self.input: x}))
      out_signal[i] = np.array(out_signal[i])
      for x,w,y in self.get_batch(origin = 'test', signal = False):
        out_bkg[i].extend(self.session.run(nnout, {self.input: x}))
      out_bkg[i] = np.array(out_bkg[i])
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    bins = np.linspace(np.amin(out_signal[0]), np.amax(out_signal[0]), 10)
    for i in range(N):
      label_s = None
      label_b = None
      if i == 0:
        label_s = "Test signal"
        label_b = "Test bkg."
      sns.distplot(out_signal[i], bins = bins,
                   kde = False, label = label_s, norm_hist = True, hist = True,
                   hist_kws={"histtype": "step", "linewidth": 2, "color": "r"})
      sns.distplot(out_bkg[i], bins = bins,
                   kde = False, label = label_b, norm_hist = True, hist = True,
                   hist_kws={"histtype": "step", "linewidth": 2, "color": "b"})
    ax.set(xlabel = 'NN output', ylabel = 'Events', title = '');
    ax.legend(frameon = False)
    plt.savefig(filename)
    plt.close("all")

  def get_batch(self, origin = 'train', **kwargs):
    filt = np.ones(self.file[origin].shape[0], dtype = 'bool')
    if 'signal' in kwargs and kwargs['signal']:
      filt = filt & self.file['%s_%s' % (origin, 'signal')][:]
    elif 'signal' in kwargs and not kwargs['signal']:
      filt = filt & self.file['%s_%s' % (origin, 'bkg')][:]

    filt = np.where(filt)[0]

    rows = np.random.permutation(filt)
    N = len(rows)

    for i in range(0, int(N/self.n_batch)):
      r = rows[i*self.n_batch : (i+1)*self.n_batch]
      r = sorted(r)
      x_batch = self.file[origin][r, self.col_data:]
      x_batch_w = self.file[origin][r, self.col_weight]
      y_batch = self.file[origin][r, self.col_signal]
      yield x_batch, x_batch_w, y_batch

  def get_batch_train(self):
    N = len(self.file['train'])
    rows = np.random.permutation(N)

    r = rows[0 : self.n_batch]
    r = sorted(r)
    x_batch = self.file['train'][r, self.col_data:]
    x_batch_w = self.file['train'][r, self.col_weight]
    y_batch = self.file['train'][r, self.col_signal]
    return x_batch, x_batch_w, y_batch

  def train(self, prefix, result_dir, network_dir):
    N = len(self.file['train'])

    first_state = []
    for i in sorted(self.var.keys()):
      first_state.append(tf.zeros(self.var[i].shape, dtype = tf.float32))

    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                          target_log_prob_fn = self.target_log_prob_fn,
                          step_size          = 0.01,
                          num_leapfrog_steps = 100)
    states, kernel_results = tfp.mcmc.sample_chain(
                          num_results        = self.n_posterior,
                          current_state      = first_state,
                          kernel             = hmc_kernel,
                          num_burnin_steps   = 500)

    with self.session.as_default():
      self.posterior, results_out = self.session.run([states, kernel_results])

    print(self.posterior)
    print(results_out)

    self.save("%s/%s_discriminator" % (network_dir, prefix))
    #gc.collect()
    print("============ End of training ===============")

  def save(self, discriminator_filename):
    outputs = {'output': self.nn(self.input)}
    for i in self.var:
      outputs[i] = self.var
    tf.save_model.simple_save(self.session,
                              discriminator_filename,
                              inputs = {'input': self.input},
                              outputs = outputs)

  '''
  Load stored network
  '''
  def load(self, discriminator_filename):
    self.session = tf.Session()
    with self.session.as_default():
      tf.saved_model.loader.load(self.session, [tf.saved_model.tag_constants.SERVING], discriminator_filename)
      with tf.variable_scope('model'):
        graph = tf.get_default_graph()
        self.input = graph.get_tensor_by_name("input:0")
        allVars = [op.name for op in graph.get_operations() if op.op_def and (op.op_def.name == 'Variable' or op.op_def.name == 'VariableV2')]
        allVars = sorted(allVars)
        self.count = 0
        for i in allVars:
          if ("qW_" in i[:3] or "qb_" in i[:3]) and "/loc" in i:
            name = i[:-4]
            self.var[name] = ed.Normal(loc = graph.get_tensor_by_name(name + '/loc:0'),
                                       scale = graph.get_tensor_by_name(name + '/scale:0'),
                                       name = name)
            n = int(name[3:])
            if n > self.count:
              self.count = n

    self.create_model()

def main():
  import argparse

  parser = argparse.ArgumentParser(description = 'Train a Bayesian NN to classify signal versus background.')
  parser.add_argument('--network-dir', dest='network_dir', action='store',
                    default='network',
                    help='Directory where networks are saved during training. (default: "network")')
  parser.add_argument('--result-dir', dest='result_dir', action='store',
                    default='result',
                    help='Directory where results are saved. (default: "result")')
  parser.add_argument('--input-file', dest='input', action='store',
                    default='input.h5',
                    help='Name of the file from where to read the input. If the file does not exist, create it. (default: "input.h5")')
  parser.add_argument('--prefix', dest='prefix', action='store',
                    default='bnn',
                    help='Prefix to be added to filenames when producing plots. (default: "bnn")')
  parser.add_argument('--mode', metavar='MODE', choices=['train', 'plot_input', 'plot_disc'],
                     default = 'train',
                     help='The mode is either "train" (a neural network), "plot_input" (plot input variables and correlations), "plot_disc" (plot discriminator output). (default: train)')
  args = parser.parse_args()
  prefix = args.prefix

  if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)
  if not os.path.exists(args.network_dir):
    os.makedirs(args.network_dir)

  network = BNN()
  # apply pre-processing if the preprocessed file does not exist
  if not os.path.isfile(args.input):
    network.prepare_input(filename = args.input)

  # read it from disk
  network.read_input_from_files(filename = args.input)

  # when training make some debug plots and prepare the network
  if args.mode == 'train':
    print("Plotting correlations.")
    network.plot_input_correlations("%s/%s_corr.pdf" % (args.result_dir, prefix))
    print("Plotting scatter plots.")
    network.plot_scatter_input(0, 1, "%s/%s_scatter_%d_%d.png" % (args.result_dir, prefix, 0, 1))

    # create network
    network.create_model()

    # for comparison: make a plot of the NN output value before any training
    # this will just be random!
    # try to predict if the signal or bkg. events in the test set are really signal or bkg.
    print("Plotting discriminator output.")
    network.plot_discriminator_output("%s/%s_discriminator_output_before_training.pdf" % (args.result_dir, prefix))

    # train it
    print("Training.")
    network.train(prefix, args.result_dir, args.network_dir)

    # plot training evolution
    print("Plotting train metrics.")
    network.plot_train_metrics("%s/%s_training.pdf" % (args.result_dir, prefix))

    print("Plotting discriminator output after training.")
    network.plot_discriminator_output("%s/%s_discriminator_output.pdf" % (args.result_dir, prefix))
  elif args.mode == 'plot_disc':
    network.load("%s/%s_discriminator_%s" % (args.network_dir, prefix, trained))
    network.plot_discriminator_output("%s/%s_discriminator_output.pdf" % (args.result_dir, prefix))
  elif args.mode == 'plot_input':
    network.plot_input_correlations("%s/%s_corr.pdf" % (args.result_dir, prefix))
    network.plot_scatter_input(0, 1, "%s/%s_scatter_%d_%d.png" % (args.result_dir, prefix, 0, 1))
  else:
    print('Option mode not understood.')

if __name__ == '__main__':
  main()

