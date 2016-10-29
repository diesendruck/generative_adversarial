# Helper functions for normal_logistic_ga.py

import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime


def visualize(plot_dict, var_set, runs_per_theta, num_unique_thetas):
    """Plot test figures."""
    # Create plot with subplots.
    fig, axes = plt.subplots(runs_per_theta, num_unique_thetas, figsize=(8,8))
    axes = axes.ravel()

    # Make title using the fixed parameters of the test.
    # This is correct, only if all cases in var_set use same p1, p2, q.
    vars1 = var_set[0]
    plt.suptitle((r'$A$-Matrix: '+r'$p_1={}$, '+r'$p_2={}$, '+r'$q={}$').format(
                  vars1[1], vars1[2], vars1[3]))

    # Create subplots with several trials per unique theta.
    for i in xrange(num_unique_thetas):
        for j in xrange(runs_per_theta):
            start_index = i
            plot_index = start_index+j*num_unique_thetas
            # For mixed_theta, don't show theta titles.
            #axes[start_index].set_title(r'$\theta={}$'.format(var_set[i][4]))
            axes[plot_index].imshow(plot_dict[i][j],interpolation='none', cmap='GnBu')
            axes[plot_index].tick_params(labelsize=6)

    # Save figures to directory.
    path = '/Users/mauricediesendruck/Google Drive/0-LIZHEN ' \
           'RESEARCH/sbm-multiplex networks/sbm/'
    os.chdir(path)
    plt.savefig('fig-'+time.strftime('%Y%m%d_%H:%M:%S')+'.png', format='png',
                dpi=1200)


def sample_a(n, p_pos, p_neg, p_btwn, theta_fill_value, verbose):
    theta = np.empty([n, n]); theta.fill(theta_fill_value)
    mixed_theta = True
    if mixed_theta:
        # Make arbitrary theta matrix to see effect on sampled A-matrices.
        theta = make_mixed_theta(theta)
    z = sample_ising(theta)
    q = build_q_matrix(z, p_pos, p_neg, p_btwn)
    a = sample_sbm(q, n)
    a = a[:, np.argsort(z)][np.argsort(z), :]

    if verbose == True:
        summarize(n, p_pos, p_neg, p_btwn, theta_fill_value, z, q, a)
    return a


def make_mixed_theta(theta):
    """ Define mixed theta matrix."""
    n = len(theta)
    theta.fill(0)
    if False:
        np.fill_diagonal(theta, 5)
        rng = np.arange(n-1); theta[rng, rng+1] = 5
        rng = np.arange(n-2); theta[rng, rng+2] = 5
        rng = np.arange(n-3); theta[rng, rng+3] = 5
    if True:
        theta[:, [0, 1, 2, 3, 4]] = 5
    theta = sym_matrix(theta)
    return theta


def summarize(n, p_pos, p_neg, p_btwn, theta_fill_value, z, q, a):
    print('N: ', n)
    print('Pr(1): ', p_pos)
    print('Pr(-1): ', p_neg)
    print('Pr(between): ', p_btwn)
    print('Theta fill value: ', theta_fill_value)
    print
    print("Z vector: ")
    print(z)
    print("q matrix:")
    print(q)
    print("For q: ", check_symmetry(q))
    print("a matrix:")
    print(a)
    print("For a: ", check_symmetry(a))


def sample_adj_matrix(n, p):
    """Builds random adjacency matrix.

    Creates nxn adjacency matrix (1s and 0s) representing edges between nodes.
    Each edge is sampled as an independent Bernoulli random variable with
    probability p.

    Args:
        n: Number of nodes, and size of matrix adjacency matrix.
        p: Bernoulli probability for each edge.

    Returns:
        adj: Adjacency matrix.
    """
    adj = np.asarray([[rbern(p) for j in range(n)] for i in range(n)])
    adj = sym_matrix(adj)
    np.fill_diagonal(adj, 0)
    return adj


def build_q_matrix(z, p_pos, p_neg, p_btwn):
    """Builds q matrix from stochastic block model.

    Compares each element in z to every other element in z, assigning link
    probabilities according to the agreement between pairs of elements.

    Args:
        z: Vector of ising assignments.
        p_pos: Link probability for pair of elements in cluster +1.
        p_neg: Link probability for pair of elements in cluster -1.
        p_btwn: Link probability for pair of elements in opposite clusters.

    Returns:
        q: Q matrix of pairwise link probabilities, given the stochastic block
            model.
    """

    def cond(i, j):
        """Determines which probability value applies for a given pair.

        Args:
            i: Reference index of z vector.
            j: Comparison index of z vector.

        Returns:
            p: Probability value.
        """
        # Probability to return, which gets reassigned given conditions.
        p = 0

        # A point and itself gets a zero, so q has zeros on diagonal.
        if i == j:
            p = 0
        else:
            # When reference element is 1, return within-cluster-1 probability
            # or cross prob.
            if z[i] == 1:
                if z[i] == z[j]:  # if pair is equal, give cluster 1 probability
                    p = p_pos
                else:
                    p = p_btwn
            # When reference element is -1, return within-cluster-(-1)
            # probability or cross prob.
            elif z[i] == -1:
                if z[i] == z[j]:
                    p = p_neg
                else:
                    p = p_btwn
            else:
                p = "z[i] not in [1, -1]"
        return p

    n = len(z)
    # Evaluate over all z indices; here, indices are the range 0 to n-1.
    q = np.asarray([[cond(i, j) for j in range(n)] for i in range(n)])
    return q


def check_symmetry(q): return("Symmetry: ", (q.transpose() == q).all())


def sample_sbm(q, n):
    """Samples from the Stochastic Block Model (SBM) link probability matrix.

    Args:
        q: The link probability matrix.
        n: The number of rows (and equivalently, columns) of the matrix q.

    Returns:
        a: An instance of the link matrix, based on SBM probability matrix.
    """
    a = np.asarray([[rbern(q[i, j]) for j in range(n)] for i in range(n)])
    a = sym_matrix(a)
    return a


def rbern(p):
    r = np.random.binomial(1, p)
    return r


def sym_matrix(matrix, part="upper"):
    """Makes square, symmetric matrix, from matrix and upper/lower flag.

    Requires: import numpy as np

    Supply a square matrix and a flag like "upper" or "lower", and copy the
    chosen matrix part, symmetrically, to the other part. Diagonals are left
    alone. For example:
    matrix <- [[8, 1, 2],
               [0, 8, 4],
               [0, 0, 8]]
    sym_matrix(matrix, "upper") -> [[8, 1, 2],
                                    [1, 8, 4],
                                    [2, 4, 8]]

    Args:
        matrix: Square matrix.
        part: String indicating "upper" or "lower".

    Returns:
        m: Symmetric matrix, with either upper or lower copied across the
            diagonal.
    """
    matrix = np.asarray(matrix)
    n = matrix.shape[0]
    upper_indices = np.triu_indices(n, k=1)
    lower_indices = upper_indices[1], upper_indices[0]
    m = np.copy(matrix)
    if part=="upper":
        m[lower_indices] = m[upper_indices]
    elif part=="lower":
        m[upper_indices] = m[lower_indices]
    else:
        print("Give a good 'part' definition, e.g. 'upper' or 'lower'.")
    return m


def sample_ising(theta):
    """Given a matrix of agreement parameters, samples binary ising vector.

    Samples vector of 1's and -1's from a Gibbs sampled Ising Distribution.

    Args:
        theta: Agreement parameter matrix; one agreement coefficient for each
            pair of nodes.

    Returns:
        z_sample: Vector of n values, each either 1 or -1.
    """
    # Set up parameters and variable storage.
    n = len(theta)  # Number of nodes in graph.
    num_trials = 500  # Number of times to run the Gibbs sampler.
    burn_in = 300  # Number of Gibbs samples to discard; must be < num_trials.
    z_chain = np.zeros([num_trials, n])  # Storage for Gibbs samples, by row.

    # Initialize and store first configuration of z's.
    z0 = np.random.choice([-1, 1], n)  # Initialize z's.
    z_chain[0,:] = z0  # Store initial values as first row of z_chain.

    # Run Gibbs.
    for t in range(1, num_trials):
        z = z_chain[t-1,:]
        for i in range(n):
            # Sample each z from its full Ising model conditional.
            # pi(z_i|z_not_i) = (1/C)*exp(sum(theta*z_i*z_j)), for j's with
            #     edges to i [...actually, edge condition irrelevant here].
            # Evaluate for z_i=-1 and z_i=1, normalize, then sample.
            summation_terms_neg1 = [theta[i, j]*(-1)*z[j] if j!=i else 0 for j in range(n)]
            summation_terms_pos1 = [theta[i, j]*(1)*z[j] if j!=i else 0 for j in range(n)]
            pn = unnorm_prob_neg1 = np.exp(sum(summation_terms_neg1))
            pp = unnorm_prob_pos1 = np.exp(sum(summation_terms_pos1))
            # Normalize probabilities.
            pr_neg1 = pn/(pn+pp)
            pr_pos1 = pp/(pn+pp)
            # Sample z_i
            z_i_value = np.random.choice([-1, 1], p=[pr_neg1, pr_pos1])
            # Store z_i value in z_chain.
            z_chain[t, i] = z_i_value

    # Sample a z from the z_chain.
    sample_index = np.random.randint(burn_in, len(z_chain))
    z_sample = z_chain[sample_index,:]

    return z_sample

def evaluate_value_fn(current_beta, current_mu, x_batch, z_batch, batch_size,
                      true_data_mean):
    """
    Evaluates value function.

    Args:
        current_beta: The latest beta parameter value.
        current_mu: The latest mu parameter value.
        x_batch: Random sample, with replacement, from true data.
        z_batch: Random sample, with replacement, from noise data.
        batch_size: Number of points to sample from data.
        true_data_mean: Value of x in N(x, 1), distribution of true data.

    Returns:
         value: Value of value function.
    """
    value = 0.0
    for i in xrange(batch_size):
        value += safe_ln(discriminate_datapoint(x_batch[i], current_beta))
        value += safe_ln(1 - discriminate_datapoint(
            generate_from_noise(z_batch[i], current_mu), current_beta))
        # REGULARIZATION
        #value += abs(current_mu - true_data_mean)
    value /= batch_size
    return value


def evaluate_value_fn_second_term(current_beta, current_mu, z_batch,
                                  batch_size, true_data_mean):
    """
    Evaluates only the second term of the value function.

    Args:
        current_beta: The latest beta parameter value.
        current_mu: The latest mu parameter value.
        z_batch: Random sample, with replacement, from noise data.
        batch_size: Number of points to sample from data.
        true_data_mean: Value of x in N(x, 1), distribution of true data.

    Returns:
         value: Value of value function.
    """
    value = 0.0
    for i in xrange(batch_size):
        value += safe_ln(1 - discriminate_datapoint(
            generate_from_noise(z_batch[i], current_mu), current_beta))
        # REGULARIZATION
        #value += abs(current_mu - true_data_mean)
    value /= batch_size
    return value


def safe_ln(x, minval=0.0000000001):
    return np.log(x.clip(min=minval))


def discriminate_datapoint(datapoint, current_beta):
    """
    Produces probability of being true data, using logistic model.

    Args:
        datapoint: True or generated datapoint.
        current_beta: The latest beta parameter value.

    Returns:
        probability_true_data: Probability of being from the true data.
    """
    probability_true_data = (
        1 / (1 + np.exp(-1 * (datapoint * current_beta[1] + current_beta[0]))))
    return probability_true_data


def generate_from_noise(noise, current_mu):
    """
    Generates from normal model, using noise and current mu.

    Args:
        noise: Standard Gaussian noise, in dimension of original data.
        current_mu: The latest mu parameter value.

    Returns:
        generated_datapoint: Single datapoint from generative model.
    """
    generated_datapoint = noise + current_mu
    return generated_datapoint


# NOT CURRENTLY BEING USED.
def sample_batch(batch_size, data):
    """
    Generates batch from full data.

    Args:
        batch_size: Number of points to sample from data.
        data: Numpy array of data (e.g. true or generated).

    Returns:
         batch: Numpy array of size (batch_size, 2), where each element [x, y]
           indicates value x of class y.
    """
    num_elements = data.shape[0]
    if batch_size > num_elements:
        sys.exit("Batch size exceed data size.")
    batch_indices = np.random.choice(range(num_elements), batch_size)
    batch = [data[i] for i in batch_indices]
    batch = np.array(batch)
    return batch