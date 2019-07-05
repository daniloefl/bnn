#!/usr/bin/env python

import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
import scipy
import pandas as pd

import tensorflow as tf
import tensorflow_probability as tfp

class PF(object):
    def __init__(self, name, add_variable_fn, shape, dtype):
        self.name = name
        self.add_variable_fn = add_variable_fn
        self.shape = shape
        self.dtype = dtype
        self.pf_w = self.add_variable_fn("%s_w" % name, shape = shape, dtype = dtype, trainable = True)
        self.pf_u = self.add_variable_fn("%s_u" % name, shape = shape, dtype = dtype, trainable = True)
        self.pf_b = self.add_variable_fn("%s_b" % name, shape = [1],   dtype = dtype, trainable = True)
    def forward(self, inputs):
        z = inputs + self.pf_u*tf.sigmoid(tf.reduce_sum(tf.multiply(inputs, self.pf_w)) + self.pf_b)
        return z
    def logdetjac(self, inputs):
        scalar = tf.reduce_sum(tf.multiply(inputs, self.pf_w)) + self.pf_b
        psi = tf.sigmoid(scalar)*(1 - tf.sigmoid(scalar))*self.pf_w
        ldj = tf.log(tf.abs(1 + tf.reduce_sum(tf.multiply(self.pf_u, psi))))
        return ldj

class GaussianParam(object):
    def __init__(self, name, add_variable_fn, shape, dtype):
        self.name = name
        self.loc_initializer = tf.compat.v1.initializers.random_normal(stddev=0.1)
        self.scale_initializer = tf.compat.v1.initializers.random_normal(mean=-3., stddev=0.1)
        self.add_variable_fn = add_variable_fn
        self.shape = shape
        self.dtype = dtype
        self.m = self.add_variable_fn("%s_m" % self.name, shape = self.shape, dtype = self.dtype, initializer = self.loc_initializer,     trainable = True)
        self.s = self.add_variable_fn("%s_s" % self.name, shape = self.shape, dtype = self.dtype, initializer = self.scale_initializer,   trainable = True)
        self.prior = tfp.distributions.Normal(loc = tf.zeros(self.shape), scale = tf.ones(self.shape))
        self.posterior = tfp.distributions.Normal(loc = self.mean(), scale = self.stddev())
    def mean(self):
        return self.m
    def stddev(self):
        return (1e-6 + tf.math.softplus(self.s))
    def sample(self):
        return self.posterior.sample()

class PlanarFlowDense(tf.keras.layers.Layer):
    def __init__(self, units, activation, n_planar_flows):
        super(PlanarFlowDense, self).__init__()
        self.units = int(units)
        self.activation = tf.keras.activations.get(activation)
        self.n_planar_flows = int(n_planar_flows)
        self.built = False
    def W_posterior(self):
        # sample from weights
        zW = self.W.sample()
        # forward the weights through the norm. flow
        for i in range(1, self.n_planar_flows+1):
            zW = self.pf_W[i].forward(zW)
        return zW
    def b_posterior(self):
        # sample from weights
        zb = self.b.sample()
        # forward the weights through the norm. flow
        for i in range(1, self.n_planar_flows+1):
            zb = self.pf_b[i].forward(zb)
        return zb
    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())

        self.shape_W = [int(input_shape[-1]), self.units]
        self.shape_b = [1, self.units]

        # trainable parameters: mean and std. dev.
        self.W = GaussianParam("W", add_variable_fn = self.add_variable, shape = self.shape_W, dtype = dtype)
        self.b = GaussianParam("b", add_variable_fn = self.add_variable, shape = self.shape_b, dtype = dtype)

        ## initialise PlanarFlow
        self.pf_W = {}
        self.pf_b = {}
        for i in range(1, self.n_planar_flows+1):
            self.pf_W[i] = PF("pfW%d" % i, self.add_variable, self.shape_W, dtype)
            self.pf_b[i] = PF("pfb%d" % i, self.add_variable, self.shape_b, dtype)

        self.built = True
    def call(self, inputs, training = False):
        inputs = tf.convert_to_tensor(value=inputs, dtype=self.dtype)

        # generate weights from posterior
        zW = {}
        zb = {}
        # sample from weights
        zW[0] = self.W.sample()
        zb[0] = self.b.sample()
        # forward the weights through the norm. flow
        for i in range(1, self.n_planar_flows+1):
            zW[i] = self.pf_W[i].forward(zW[i-1])
            zb[i] = self.pf_b[i].forward(zb[i-1])

        # use weights to propagate input
        out = tf.matmul(inputs, zW[self.n_planar_flows]) + zb[self.n_planar_flows]
        out = self.activation(out)

        # add KL divergence between prior and posterior Gaussians using weights before the flow
        batch_ndims = tf.size(self.W.prior.batch_shape_tensor())
        kl_div  = tfp.distributions.kl_divergence(tfp.distributions.Independent(self.W.posterior, reinterpreted_batch_ndims = batch_ndims),
                                                  tfp.distributions.Independent(self.W.prior, reinterpreted_batch_ndims = batch_ndims))
        kl_div += tfp.distributions.kl_divergence(tfp.distributions.Independent(self.b.posterior, reinterpreted_batch_ndims = batch_ndims),
                                                  tfp.distributions.Independent(self.b.prior, reinterpreted_batch_ndims = batch_ndims))
        self.add_loss(kl_div, inputs = False)
        # include effect of the flow with the - log det Jacobian for each step of the flow
        ldj = 0
        for i in range(1, self.n_planar_flows+1):
            ldj += - self.pf_W[i].logdetjac(zW[i-1])
            ldj += - self.pf_b[i].logdetjac(zb[i-1])
        self.add_loss(ldj, inputs = True)

        return out

class VIBNN_NF(object):
    def __init__(self,
                 network_dir = 'network',
                 result_dir = 'result',
                 batch_size = 128,
                 n_planar_flows = 2):
        self.network_dir = network_dir
        self.result_dir = result_dir
        self.batch_size = batch_size
        self.n_planar_flows = n_planar_flows
        self.Nepoch = 2
    def load_data(self, filename):
        self.file = pd.HDFStore(filename, 'r')
        self.N = self.file['df'].shape[0]
    def create_model(self):
        self.graph = tf.Graph()
        self.session = tf.Session(graph = self.graph)
        with self.graph.as_default() as g:
            x = tf.placeholder(tf.float32, shape=(None, 2), name = 'x')
            y = tf.placeholder(tf.float32, shape=(None, 1), name = 'y')
            with tf.variable_scope('model'):
                nn = tf.keras.Sequential([
                                        PlanarFlowDense(20, activation = tf.nn.relu, n_planar_flows = self.n_planar_flows),
                                        PlanarFlowDense(10, activation = tf.nn.relu, n_planar_flows = self.n_planar_flows),
                                        PlanarFlowDense( 2, activation = None,       n_planar_flows = self.n_planar_flows),
                                        ], name = 'nn')
                logits = tf.identity(nn(x), name = 'logits')
                pred_distribution = tfp.distributions.Categorical(logits = logits)
                logprob = tf.reduce_mean(pred_distribution.log_prob(y[:,0]), name = 'logprob')
                pred = tf.cast(tf.argmax(logits, axis = -1), tf.float32, name = 'prediction')
                #pred_distribution = tfp.distributions.Normal(loc = logits[:,0], scale = 1.0) 
                #logprob = tf.reduce_mean(pred_distribution.log_prob(y[:,0]), name = 'logprob')
                #pred = tf.identity(logits[:,0], name = 'prediction')
                mse = tf.reduce_mean(tf.square(y[:,0] - pred), name = 'mse')
                acc = tf.reduce_mean(tf.cast(tf.equal(pred, y[:,0]), tf.float32), name = 'acc')
                kldiv = tf.identity(sum(nn.losses)/self.N, name = 'kldiv')
                elbo = tf.identity(logprob - kldiv, name = 'elbo')
                Wmu = {}
                Wstd = {}
                bmu = {}
                bstd = {}
                for i, layer in enumerate(nn.layers):
                    try:
                        Wmu[i] = layer.W.mean()
                        Wstd[i] = layer.W.stddev()
                        bmu[i] = layer.b.mean()
                        bstd[i] = layer.b.stddev()
                    except AttributeError:
                        continue
                opt = tf.train.AdamOptimizer(learning_rate = 0.01)
                train_op = opt.minimize(-elbo, name = 'train_op')
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), name = 'init_op')
            self.session.run(init_op)
            #g.finalize()
    def get_batch(self, **kwargs):
        filt = np.ones(self.N, dtype = 'bool')
        if 'signal' in kwargs and kwargs['signal']:
            filt = filt & self.file['df'].loc[:, 'sig'].values.astype(bool)
        elif 'signal' in kwargs and not kwargs['signal']:
            filt = filt & (~(self.file['df'].loc[:, 'sig'].values.astype(bool)))
        filt = np.where(filt)[0]
        rows = np.random.permutation(filt)
        for i in range(0, int(self.N/self.batch_size)):
            r = rows[i*self.batch_size : (i+1)*self.batch_size]
            r = sorted(r)
            if len(r) == 0: break
            df = self.file.select('df', where = 'index = r')
            x_batch = df.drop(['w', 'sig'], axis = 1)
            y_batch = df.loc[:, 'sig'].to_frame()
            yield x_batch, y_batch

    def train(self):
        with self.graph.as_default() as g:
            x = g.get_tensor_by_name('x:0')
            y = g.get_tensor_by_name('y:0')
            pred = g.get_tensor_by_name('model/prediction:0')
            kldiv = g.get_tensor_by_name('model/kldiv:0')
            elbo = g.get_tensor_by_name('model/elbo:0')
            mse = g.get_tensor_by_name('model/mse:0')
            acc = g.get_tensor_by_name('model/acc:0')
            logprob = g.get_tensor_by_name('model/logprob:0')
            train_op = g.get_operation_by_name('model/train_op')
            batch_count = 0
            for i in range(0, self.Nepoch):
                for x_batch, y_batch in self.get_batch():
                    _ = self.session.run(train_op,
                                         feed_dict = {x: x_batch,
                                                      y: y_batch})
                    batch_count += 1
                    if batch_count % 10 == 0:
                        elbo_, logprob_, kldiv_, acc_, mse_ = 0.0, 0.0, 0.0, 0.0, 0.0
                        c = 0
                        for x_batch, y_batch in self.get_batch():
                            elbo_t, logprob_t, kldiv_t, acc_t, mse_t = self.session.run([elbo, logprob, kldiv, acc, mse], feed_dict = {x: x_batch, y: y_batch})
                            elbo_ += elbo_t
                            logprob_ += logprob_t
                            kldiv_ += kldiv_t
                            acc_ += acc_t
                            mse_ += mse_t
                            c += 1
                        elbo_ /= c
                        logprob_ /= c
                        kldiv_ /= c
                        acc_ /= c
                        mse_ /= c
                        print("Epoch {:2d}, batch cum. count {:2d}: -ELBO = -(logprob - kldiv): {:>6.3f}, logprob: {:>6.3f}, kldiv: {:>6.3f}, accuracy: {:>6.5f}, mse(out[0], y): {:>6.5f}".format(i, batch_count, -elbo_, logprob_, kldiv_, acc_, mse_))
                        self.save('nn', batch_count)
    def run(self, x_batch, n_posterior_run = 10):
        with self.graph.as_default() as g:
            x = g.get_tensor_by_name('x:0')
            logits = g.get_tensor_by_name('model/logits:0')
            pred = g.get_tensor_by_name('model/prediction:0')
            pred_ = np.zeros((len(x_batch), n_posterior_run))
            logits_ = np.zeros((len(x_batch), 2, n_posterior_run))
            for i in range(0, n_posterior_run):
                pred_[:, i], logits_[:, :, i] = self.session.run([pred, logits], feed_dict = {x: x_batch})
            prob_ = scipy.special.softmax(logits_, axis = 1)
            return pred_, prob_
    def save(self, filename, step):
        with self.graph.as_default() as g:
            saver = tf.train.Saver()
            saver.save(self.session, '{}/{}'.format(self.network_dir, filename), global_step = step)
    def load(self, filename, step):
        self.graph = tf.Graph()
        self.session = tf.Session(graph = self.graph)
        with self.graph.as_default() as g:
            saver = tf.train.import_meta_graph('{}/{}-{}.meta'.format(self.network_dir, filename, step))
            saver.restore(self.session, tf.train.latest_checkpoint('{}/'.format(self.network_dir)))
    def plot_nn_output(self, filename):
        n_posterior_run = 10
        out_signal = np.zeros(shape = (0, n_posterior_run))
        out_bkg = np.zeros(shape = (0, n_posterior_run))
        for x,y in self.get_batch(signal = True): out_signal = np.append(out_signal, self.run(x, n_posterior_run)[1][:,1,:], axis = 0)
        for x,y in self.get_batch(signal = False): out_bkg = np.append(out_bkg, self.run(x, n_posterior_run)[1][:,1,:], axis = 0)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        bins = np.linspace(np.amin(out_signal[:,0]), np.amax(out_signal[:,0]), 10)
        N = out_signal.shape[1]
        for i in range(N):
            label_s = None
            label_b = None
            if i == 0:
                label_s = "signal"
                label_b = "bkg."
            sns.distplot(out_signal[:,i], bins = bins,
                         kde = False, label = label_s, norm_hist = True, hist = True,
                         hist_kws={"histtype": "step", "linewidth": 2, "color": "r"})
            sns.distplot(out_bkg[:,i], bins = bins,
                         kde = False, label = label_b, norm_hist = True, hist = True,
                         hist_kws={"histtype": "step", "linewidth": 2, "color": "b"})
        ax.set(xlabel = 'NN output', ylabel = 'Events', title = '');
        ax.legend(frameon = False)
        plt.savefig('{}/{}'.format(self.result_dir, filename))
        plt.close("all")
    def plot_scatter_output(self, A, B, filename_mean, filename_std):
        n_posterior_run = 100

        df = self.file.select('df')

        # make countour
        mins = [df[A].min(), df[B].min()]
        maxs = [df[A].max(), df[B].max()]
        step = [(maxs[0] - mins[0])/50.0, (maxs[1] - mins[1])/50.0]
        bx,by = np.mgrid[mins[0]:(maxs[0]+0.5*step[0]):step[0], mins[1]:(maxs[1]+0.5*step[0]):step[1]]
        inputs = np.vstack([bx.flatten(), by.flatten()]).T
        inputs = inputs.astype(np.float32)

        prob_ = self.run(inputs, n_posterior_run)[1][:,1,:]
        prob_mean = np.mean(prob_, axis = 1)
        prob_std = np.std(prob_, axis = 1)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        cmap = sns.diverging_palette(250, 12, s=85, l=25, as_cmap=True)

        contour = ax.contourf(bx, by, prob_std.reshape((-1, bx.shape[1])), cmap=cmap)

        ax.scatter(df[df['sig'] == 0][A], df[df['sig'] == 0][B], color = 'b', marker = 's', s = 5, label='Background')
        ax.scatter(df[df['sig'] == 1][A], df[df['sig'] == 1][B], color = 'r', marker = 's', s = 5, label='Signal')

        ax.set_xlim([mins[0], maxs[0]])
        ax.set_ylim([mins[1], maxs[1]])
        ax.set(xlabel=A, ylabel=B, title='');
        cbar = plt.colorbar(contour, ax=ax)
        cbar.ax.set_ylabel('Output (sample of the posterior)');
        ax.legend()

        plt.savefig('{}/{}'.format(self.result_dir, filename_std))
        plt.close("all")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        cmap = sns.diverging_palette(250, 12, s=85, l=25, as_cmap=True)

        contour = ax.contourf(bx, by, prob_mean.reshape((-1, bx.shape[1])), cmap=cmap)

        ax.scatter(df[df['sig'] == 0][A], df[df['sig'] == 0][B], color = 'b', marker = 's', s = 5, label='Background')
        ax.scatter(df[df['sig'] == 1][A], df[df['sig'] == 1][B], color = 'r', marker = 's', s = 5, label='Signal')

        ax.set_xlim([mins[0], maxs[0]])
        ax.set_ylim([mins[1], maxs[1]])
        ax.set(xlabel=A, ylabel=B, title='');
        cbar = plt.colorbar(contour, ax=ax)
        cbar.ax.set_ylabel('Output (sample of the posterior)');
        ax.legend()

        plt.savefig('{}/{}'.format(self.result_dir, filename_mean))
        plt.close("all")

def main():
    import argparse

    parser = argparse.ArgumentParser(description = 'Train a Bayesian NN to classify signal versus background using variational inference in TFP.')
    parser.add_argument('--network-dir', dest='network_dir', action='store',
                      default='network',
                      help='Directory where networks are saved during training. (default: "network")')
    parser.add_argument('--result-dir', dest='result_dir', action='store',
                      default='result',
                      help='Directory where results are saved. (default: "result")')
    parser.add_argument('--input-file', dest='input', action='store',
                      default='data.h5',
                      help='Name of the file from where to read the input. If the file does not exist, create it. (default: "input.h5")')
    parser.add_argument('--load', dest = 'load', action = 'store',
                       default = 300,
                       help='Batch count to use when loading for testing.')
    parser.add_argument('--train', dest = 'train', action = 'store_true',
                       default = False,
                       help='Train the network.')
    parser.add_argument('--test', dest = 'test', action = 'store_true',
                       default = False,
                       help='Test the network.')
    args = parser.parse_args()
    
    vibnn = VIBNN_NF(network_dir = args.network_dir,
                  result_dir = args.result_dir)
    vibnn.load_data(args.input)

    if args.train:
        vibnn.create_model()
        vibnn.train()
    elif args.test:
        vibnn.load('nn', int(args.load))
        vibnn.plot_nn_output('nn_output.png')
        vibnn.plot_scatter_output('A', 'B', 'scatter_mean.png', 'scatter_std.png')
    else:
        print('Please provide an action to be executed.')
        sys.exit(-1)

if __name__ == '__main__':
    main()

