'''
An example of distribution approximation using Generative Adversarial Networks in TensorFlow.

Based on the blog post by Eric Jang:
    http://blog.evjang.com/2016/06/generative-adversarial-nets-in.html,
and of course the original GAN paper by Ian Goodfellow et. al.:
    https://arxiv.org/abs/1406.2661.

The minibatch discrimination technique is taken from Tim Salimans et. al.:
    https://arxiv.org/abs/1606.03498.
'''

import argparse
import numpy as np
from scipy.stats import norm
import tensorflow as tf

import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns

sns.set(color_codes=True)

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)


class DataDistribution(object):
    def __init__(self, mu=4, sigma=0.5):
        self.mu = mu
        self.sigma = sigma

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples


class GeneratorDistribution(object):
    def __init__(self, d_range):
        self.d_range = d_range

    def sample(self, N):
        return np.linspace(-self.d_range, self.d_range, N) + \
            np.random.random(N) * 0.01


def linear(input_vec, out_size, scope=None, stddev=1.0):
    norm = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope or 'linear'):
        weights = tf.get_variable(
            'weights',
            [input_vec.get_shape()[1], out_size],
            initializer=norm)
        bias = tf.get_variable(
            'bias',
            [out_size],
            initializer=const)
        return tf.matmul(input_vec, weights) + bias


def generator(input_vec, hidden_size):
    h0 = tf.nn.softplus(linear(input_vec, hidden_size, 'g0'))
    h1 = linear(h0, 1, 'g1')
    return h1


def discriminator(input_vec, hidden_size, minibatch_layer=True):
    h0 = tf.tanh(linear(input_vec, hidden_size, 'd0'))
    h1 = tf.tanh(linear(h0, hidden_size, 'd1'))

    # without the minibatch layer, the discriminator needs an additional layer
    # to have enough capacity to separate the two distributions correctly
    if minibatch_layer:
        h2 = minibatch(h1)
    else:
        h2 = tf.tanh(linear(h1, hidden_size, scope='d2'))

    h3 = tf.sigmoid(linear(h2, 1, scope='d3'))
    return h3


def minibatch(input, num_kernels=5, kernel_dim=3):
    x = linear(input, num_kernels * kernel_dim, scope='minibatch', stddev=0.02)
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    eps = tf.expand_dims(np.eye(int(input.get_shape()[0]), dtype=np.float32), 1)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2) + eps
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    return tf.concat(1, [input, minibatch_features])


def update(loss, var_list,
           initial_learning_rate=0.005, decay=0.95, num_decay_steps=150):
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_steps,
        decay,
        staircase=True
    )

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    update_step = optimizer.minimize(
        loss,
        global_step=batch,
        var_list=var_list
    )

    return update_step


class GAN(object):
    def __init__(self, data, gen, minibatch):
        self.data = data
        self.gen = gen
        self.minibatch = minibatch
        self.mlp_hidden_size = 4
        self.anim_frames = []
        self._create_model()

    def _create_model(self):
        # In order to make sure that the discriminator is providing useful gradient
        # information to the generator from the start, we're going to pretrain the
        # discriminator using a maximum likelihood objective. We define the network
        # for this pretraining step scoped as D_pre.
        with tf.variable_scope('D_pre'):
            self.pre_input = tf.placeholder(tf.float32, shape=(None, 1))
            self.pre_labels = tf.placeholder(tf.float32, shape=(None, 1))
            D_pre = discriminator(self.pre_input, self.mlp_hidden_size, self.minibatch)
            self.pre_loss = tf.reduce_mean(tf.square(D_pre - self.pre_labels))
            self.pre_opt = update(self.pre_loss, None)

        # This defines the generator network - it takes samples from a noise
        # distribution as input, and passes them through an MLP.
        with tf.variable_scope('G'):
            self.z = tf.placeholder(tf.float32, shape=(None, 1))
            self.G = generator(self.z, self.mlp_hidden_size)

        # The discriminator tries to tell the difference between samples from the
        # true data distribution (self.x) and the generated samples (self.z).
        #
        # Here we create two copies of the discriminator network (that share parameters),
        # as you cannot use the same network with different inputs in TensorFlow.
        with tf.variable_scope('D') as scope:
            self.x = tf.placeholder(tf.float32, shape=(None, 1))
            self.D_real = discriminator(self.x, self.mlp_hidden_size, self.minibatch)
            scope.reuse_variables()
            self.D_gen = discriminator(self.G, self.mlp_hidden_size, self.minibatch)

        # Define the loss for discriminator and generator networks (see the original
        # paper for details), and create optimizers for both
        self.loss_d = tf.reduce_mean(-tf.log(self.D_real) - tf.log(1 - self.D_gen))
        self.loss_g = tf.reduce_mean(-tf.log(self.D_gen))

        tf_vars = tf.trainable_variables()
        self.d_pre_params = [v for v in tf_vars if v.name.startswith('D_pre/')]
        self.d_params = [v for v in tf_vars if v.name.startswith('D/')]
        self.g_params = [v for v in tf_vars if v.name.startswith('G/')]

        self.opt_d = update(self.loss_d, self.d_params)
        self.opt_g = update(self.loss_g, self.g_params)

    def train(self, batch_size, num_steps, num_pretrain_steps,
              log_interval=100, anim_path=None):
        with tf.Session() as session:
            tf.initialize_all_variables().run()

            # pretraining discriminator
            for step in range(num_pretrain_steps):
                d = (np.random.random(batch_size) - 0.5) * 10.0
                labels = norm.pdf(d, loc=self.data.mu, scale=self.data.sigma)
                pretrain_loss, _ = session.run([self.pre_loss, self.pre_opt], {
                    self.pre_input: np.reshape(d, (batch_size, 1)),
                    self.pre_labels: np.reshape(labels, (batch_size, 1))
                })
            self.weightsD = session.run(self.d_pre_params)

            # copy weights from pre-training over to new D network
            for i, v in enumerate(self.d_params):
                session.run(v.assign(self.weightsD[i]))

            for step in range(num_steps):
                # update discriminator
                x = self.data.sample(batch_size)
                z = self.gen.sample(batch_size)
                loss_d, _ = session.run([self.loss_d, self.opt_d], {
                    self.x: np.reshape(x, (batch_size, 1)),
                    self.z: np.reshape(z, (batch_size, 1))
                })

                # update generator
                z = self.gen.sample(batch_size)
                loss_g, _ = session.run([self.loss_g, self.opt_g], {
                    self.z: np.reshape(z, (batch_size, 1))
                })

                if step % log_interval == 0:
                    print('{}:\t{:.3}\t{:.3}'.format(step, loss_d, loss_g))

                if anim_path:
                    self.anim_frames.append(
                        self._samples(session, batch_size))

            if anim_path:
                self._save_animation(anim_path)
            else:
                self._plot_distributions(session, batch_size)

    def _samples(self, session, batch_size, num_points=1000, num_bins=100):
        '''
        Return a tuple (db, pd, pg), where db is the current decision
        boundary, pd is a histogram of samples from the data distribution,
        and pg is a histogram of generated samples.
        '''
        xs = np.linspace(-self.gen.d_range, self.gen.d_range, num_points)
        bins = np.linspace(-self.gen.d_range, self.gen.d_range, num_bins)

        # decision boundary
        db = np.zeros((num_points, 1))
        for i in range(num_points // batch_size):
            db[batch_size * i:batch_size * (i + 1)] = session.run(self.D_real, {
                self.x: np.reshape(
                    xs[batch_size * i:batch_size * (i + 1)],
                    (batch_size, 1)
                )
            })

        # data distribution
        d = self.data.sample(num_points)
        pd, _ = np.histogram(d, bins=bins, density=True)

        # generated samples
        zs = np.linspace(-self.gen.d_range, self.gen.d_range, num_points)
        g = np.zeros((num_points, 1))
        for i in range(num_points // batch_size):
            g[batch_size * i:batch_size * (i + 1)] = session.run(self.G, {
                self.z: np.reshape(
                    zs[batch_size * i:batch_size * (i + 1)],
                    (batch_size, 1)
                )
            })
        pg, _ = np.histogram(g, bins=bins, density=True)

        return db, pd, pg

    def _plot_distributions(self, session, batch_size):
        db, pd, pg = self._samples(session, batch_size)
        db_x = np.linspace(-self.gen.d_range, self.gen.d_range, len(db))
        p_x = np.linspace(-self.gen.d_range, self.gen.d_range, len(pd))
        f, ax = plt.subplots(1)
        ax.plot(db_x, db, label='decision boundary')
        ax.set_ylim(0, 1)
        plt.plot(p_x, pd, label='real data')
        plt.plot(p_x, pg, label='generated data')
        plt.title('1D Generative Adversarial Network')
        plt.xlabel('Data values')
        plt.ylabel('Probability density')
        plt.legend()
        plt.show()

    def _save_animation(self, anim_path):
        f, ax = plt.subplots(figsize=(6, 4))
        f.suptitle('1D Generative Adversarial Network', fontsize=15)
        plt.xlabel('Data values')
        plt.ylabel('Probability density')
        ax.set_xlim(-6, 6)
        ax.set_ylim(0, 1.4)
        line_db, = ax.plot([], [], label='decision boundary')
        line_pd, = ax.plot([], [], label='real data')
        line_pg, = ax.plot([], [], label='generated data')
        frame_number = ax.text(
            0.02,
            0.95,
            '',
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes
        )
        ax.legend()

        db, pd, _ = self.anim_frames[0]
        db_x = np.linspace(-self.gen.d_range, self.gen.d_range, len(db))
        p_x = np.linspace(-self.gen.d_range, self.gen.d_range, len(pd))

        def init():
            line_db.set_data([], [])
            line_pd.set_data([], [])
            line_pg.set_data([], [])
            frame_number.set_text('')
            return line_db, line_pd, line_pg, frame_number

        def animate(i):
            frame_number.set_text(
                'Frame: {}/{}'.format(i, len(self.anim_frames))
            )
            db, pd, pg = self.anim_frames[i]
            line_db.set_data(db_x, db)
            line_pd.set_data(p_x, pd)
            line_pg.set_data(p_x, pg)
            return (line_db, line_pd, line_pg, frame_number)

        anim = animation.FuncAnimation(
            f,
            animate,
            init_func=init,
            frames=len(self.anim_frames),
            blit=True
        )
        anim.save(anim_path, fps=30, extra_args=['-vcodec', 'libx264'])


def main(args):
    model = GAN(
        DataDistribution(),
        GeneratorDistribution(d_range=8),
        args.minibatch
    )

    model.train(
        args.batch_size, args.num_steps, args.num_pretrain_steps,
        args.log_interval, args.anim)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=1200,
                        help='the number of training steps to take')
    parser.add_argument('--num-pretrain-steps', type=int, default=0,
                        help='the number of pretraining steps to take')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='the batch size')
    parser.add_argument('--minibatch', type=bool, default=False,
                        help='use minibatch discrimination')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='print loss after this many steps')
    parser.add_argument('--anim', type=str, default=None,
                        help='name of the output animation file (default: none)')
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
