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

from graphviz import Digraph

def tf_to_dot(graph):
    dot = Digraph()
    for n in graph.as_graph_def().node:
        dot.node(n.name, label=n.name)
        for i in n.input:
            dot.edge(i, n.name)
    return dot

class BNN(object):
  '''
  Implementation of a test BNN.
  '''

  def __init__(self, n_iteration = 1050,
               n_batch = 32,
               n_eval = 50,
               n_posterior = 10000,
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
    self.n_posterior_run = 5

    self.variational_inference = variational_inference

    self.session = None
    self.graph = None

    self.input = None
    self.output = None

    self.posterior = {} # posterior samples
    self.posterior_mean = {} # posterior mean
    self.posterior_std = {} # posterior std

  '''
    Model log. prob. (input x, output y, prior parameters for W and b, W, b)
  '''
  def model_logprob(self, x, y, p_Wmu, p_Wstd, p_bmu, p_bstd, **param):
    logprob = tf.zeros([])
    rv_W = []
    rv_b = []
    for i in range(0, len(self.layers)-1):
      rv_W.append(tfp.distributions.Normal(loc = p_Wmu[i], scale = p_Wstd[i]))
      rv_b.append(tfp.distributions.Normal(loc = p_bmu[i], scale = p_bstd[i]))

      logprob += tf.reduce_sum(rv_W[i].log_prob(param["W_%d" % i]))
      logprob += tf.reduce_sum(rv_b[i].log_prob(param["b_%d" % i]))

      x = tf.matmul(x, param["W_%d" % i]) + param["b_%d" %i]
      if i < len(self.layers)-2:
        x = tf.nn.leaky_relu(x, alpha = 0.2)
      else:
        x = tf.nn.sigmoid(x)
    rv_observation = tfp.distributions.Normal(loc = x, scale = 1.0)
    logprob += tf.reduce_sum(rv_observation.log_prob(y), axis = [0, 1])
    return logprob

  '''
    Return log. prob. in model_logprob feeding it the data.
    The only parameters in param are the W and b being sampled.
    Models the log. posterior log p(w,b|x,y).
  '''
  def weight_inference_logprob(self, *param):
    kwargs = {}
    c = 0
    for i in range(0, len(self.layers)-1):
      kwargs["W_%s" % i] = param[c]
      c += 1
      kwargs["b_%s" % i] = param[c]
      c += 1
    p_Wmu = [tf.zeros([self.layers[i], self.layers[i+1]]) for i in range(0, len(self.layers)-1)]
    p_Wstd = [tf.ones([self.layers[i], self.layers[i+1]]) for i in range(0, len(self.layers)-1)]
    p_bmu = [tf.zeros([self.layers[i+1]]) for i in range(0, len(self.layers)-1)]
    p_bstd = [tf.ones([self.layers[i+1]]) for i in range(0, len(self.layers)-1)]
    return self.model_logprob(x = self.input, y = self.output, p_Wmu = p_Wmu, p_Wstd = p_Wstd, p_bmu = p_bmu, p_bstd = p_bstd, **kwargs)

  '''
    Return log. prob. in model_logprob fixing the posterior W and b and inputting new data.
    The parameters in param are y, W and b being sampled
    Models log p(y|w,b,x) p(w) p(b).
  '''
  def run_logprob(self, *param):
    kwargs = {}
    c = 0
    kwargs["y"] = param[c]
    c += 1
    for i in range(0, len(self.layers)-1):
      kwargs["W_%s" % i] = param[c]
      c += 1
      kwargs["b_%s" % i] = param[c]
      c += 1
    p_Wmu =  [tf.convert_to_tensor(self.posterior_mean["W_%d" % (i)], dtype = tf.float32) for i in range(0, len(self.layers)-1)]
    p_Wstd = [tf.convert_to_tensor(self.posterior_std["W_%d" % (i)], dtype = tf.float32) for i in range(0, len(self.layers)-1)]
    p_bmu =  [tf.convert_to_tensor(self.posterior_mean["b_%d" % (i)], dtype = tf.float32) for i in range(0, len(self.layers)-1)]
    p_bstd = [tf.convert_to_tensor(self.posterior_std["b_%d" % (i)], dtype = tf.float32) for i in range(0, len(self.layers)-1)]
    return self.model_logprob(x = self.input, p_Wmu = p_Wmu, p_Wstd = p_Wstd, p_bmu = p_bmu, p_bstd = p_bstd, **kwargs)

  def sample(self, logprob, first_state, n_posterior):
    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                        target_log_prob_fn = logprob,
                        step_size          = 0.01,
                        num_leapfrog_steps = 10)

    states, kernel_results = tfp.mcmc.sample_chain(
                        num_results        = n_posterior,
                        current_state      = first_state,
                        kernel             = hmc_kernel,
                        num_burnin_steps   = 500)

    return states

  def set_sample_run(self, n_posterior):
    first_state = []
    first_state.append(tf.random_normal(shape = (1,), dtype = tf.float32))
    for i in range(0, len(self.layers)-1):
      first_state.append(tf.random_normal(shape = self.posterior_mean["W_%d" % i].shape, dtype = tf.float32))
      first_state.append(tf.random_normal(shape = self.posterior_mean["b_%d" % i].shape, dtype = tf.float32))
    self.sample_run = self.sample(self.run_logprob, first_state, n_posterior)

  def set_sample_infer(self, n_posterior):
    first_state = []
    for i in range(0, len(self.layers)-1):
      first_state.append(tf.random_normal(shape = self.posterior_mean["W_%d" % i].shape, dtype = tf.float32))
      first_state.append(tf.random_normal(shape = self.posterior_mean["b_%d" % i].shape, dtype = tf.float32))
    self.sample_infer = self.sample(self.weight_inference_logprob, first_state, n_posterior)

  def run(self, x):
    posterior = self.session.run(self.sample_run, {self.input: x})
    return posterior[0].astype(np.float32)

  def infer(self, x, y):
    posterior = self.session.run(self.sample_infer, {self.input: x, self.output: y})
    c = 0
    for i in range(0, len(self.layers)-1):
      self.posterior["W_%d" % i] = posterior[c].astype(np.float32)
      c += 1
      self.posterior["b_%d" % i] = posterior[c].astype(np.float32)
      c += 1

  '''
    Create network.
  '''
  def create_model(self):
    self.session = tf.Session()
    with self.session.as_default():
      self.layers = [self.n_dimensions, 10, 5, 1]
      if len(self.posterior_mean) == 0:
        for i in range(0, len(self.layers)-1):
          self.posterior_mean["W_%d" % i] = np.zeros( [self.layers[i], self.layers[i+1]], dtype = np.float32 )
          self.posterior_mean["b_%d" % i] = np.zeros( [self.layers[i+1]], dtype = np.float32 )
          self.posterior_std["W_%d" % i] = np.ones( [self.layers[i], self.layers[i+1]], dtype = np.float32 )
          self.posterior_std["b_%d" % i] = np.ones( [self.layers[i+1]], dtype = np.float32 )

      if not self.input:
        self.input = tf.placeholder(tf.float32, shape = (None, self.n_dimensions), name = 'input')
      if not self.output:
        self.output = tf.placeholder(tf.float32, shape = (None), name = 'output')

      self.init_op = tf.global_variables_initializer()

      self.set_sample_infer(self.n_posterior)
      self.set_sample_run(self.n_posterior_run)



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
      signal = np.random.normal(loc = -1.0, scale = 0.5, size = (N, 2))
      bkg    = np.random.normal(loc =  1.0, scale = 0.5, size = (N, 2))
      data   = np.append(signal, bkg, axis = 0)
      data_t = np.append(np.ones(N), np.zeros(N))
      data_w = np.append(np.ones(N), np.ones(N))
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
    out_signal = np.zeros(shape = (self.n_posterior_run,0))
    out_bkg = np.zeros(shape = (self.n_posterior_run,0))
    for x,w,y in self.get_batch(origin = 'test', signal = True):
      out_signal = np.append(out_signal, self.run(x), axis = 1)
    for x,w,y in self.get_batch(origin = 'test', signal = False):
      out_bkg = np.append(out_bkg, self.run(x), axis = 1)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    print(out_signal.shape)
    bins = np.linspace(np.amin(out_signal[0,:]), np.amax(out_signal[0,:]), 10)
    N = out_signal.shape[0]
    for i in range(N):
      label_s = None
      label_b = None
      if i == 0:
        label_s = "Test signal"
        label_b = "Test bkg."
      sns.distplot(out_signal[i,:], bins = bins,
                   kde = False, label = label_s, norm_hist = True, hist = True,
                   hist_kws={"histtype": "step", "linewidth": 2, "color": "r"})
      sns.distplot(out_bkg[i,:], bins = bins,
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
      x_batch = self.file[origin][r, self.col_data:].astype(np.float32)
      x_batch_w = self.file[origin][r, self.col_weight].astype(np.float32)
      y_batch = self.file[origin][r, self.col_signal].astype(np.float32)
      yield x_batch, x_batch_w, y_batch

  def get_batch_train(self):
    N = len(self.file['train'])
    rows = np.random.permutation(N)

    r = rows[0 : self.n_batch]
    r = sorted(r)
    x_batch = self.file['train'][r, self.col_data:].astype(np.float32)
    x_batch_w = self.file['train'][r, self.col_weight].astype(np.float32)
    y_batch = self.file['train'][r, self.col_signal].astype(np.float32)
    return x_batch, x_batch_w, y_batch

  def get_full_train(self):
    x_batch = self.file['train'][:, self.col_data:].astype(np.float32)
    x_batch_w = self.file['train'][:, self.col_weight].astype(np.float32)
    y_batch = self.file['train'][:, self.col_signal].astype(np.float32)
    return x_batch, x_batch_w, y_batch

  def train(self, prefix, result_dir, network_dir):
    self.session.run(self.init_op)

    self.graph = tf.get_default_graph()

    dot = tf_to_dot(self.graph)
    dot.render('%s/graph.gv' % result_dir, view=False) 

    x,w,y = self.get_full_train()
    self.infer(x, y)

    for i in self.posterior:
      self.posterior_mean[i] = np.mean(self.posterior[i], axis = 0)
      self.posterior_std[i] = np.std(self.posterior[i], axis = 0)

    self.save("%s/%s_discriminator" % (network_dir, prefix))
    #gc.collect()
    print("============ End of training ===============")

  def save(self, discriminator_filename):
    f = h5py.File(discriminator_filename + '_posterior_samples.h5', 'w')
    for k in self.posterior:
      f.create_dataset(k, data = self.posterior[k])
    f.close()

  '''
  Load stored network
  '''
  def load(self, discriminator_filename):
    f = h5py.File(discriminator_filename + '_posterior_samples.h5')
    self.posterior = {}
    for k in list(f):
      self.posterior[k] = f[k][:].astype(np.float32)
    f.close()

    for i in sorted(self.posterior):
      self.posterior_mean[i] = np.mean(self.posterior[i], axis = 0)
      self.posterior_std[i] = np.std(self.posterior[i], axis = 0)

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

    print("Plotting discriminator output after training.")
    network.plot_discriminator_output("%s/%s_discriminator_output.pdf" % (args.result_dir, prefix))
  elif args.mode == 'plot_disc':
    network.load("%s/%s_discriminator" % (args.network_dir, prefix))
    network.plot_discriminator_output("%s/%s_discriminator_output.pdf" % (args.result_dir, prefix))
  elif args.mode == 'plot_input':
    network.plot_input_correlations("%s/%s_corr.pdf" % (args.result_dir, prefix))
    network.plot_scatter_input(0, 1, "%s/%s_scatter_%d_%d.png" % (args.result_dir, prefix, 0, 1))
  else:
    print('Option mode not understood.')

if __name__ == '__main__':
  main()

