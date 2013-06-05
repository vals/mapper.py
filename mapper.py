""" Python implementation of mapper: http://comptop.stanford.edu/u/preprints/mapperPBG.pdf
"""
from collections import defaultdict

import numpy as np
from scipy.cluster import hierarchy


def mapper(d, filter_fcn, resolution=0.2, overlap=50, magic_fudge=10):
    """ Mapper

    d: Distance matrix
    filter_fcn: Filter function used for decomposing space.
    (The 'Filter function' appear to be an array of values for each point
        in the matlab implementation.)
    """
    step = (filter_fcn.max() - filter_fcn.min()) * resolution

    level = 1
    node_info = defaultdict(dict)
    level_idx = {}
    adja = {}

    print('Filter range: [{:.2f}-{:.2f}]'.format(filter_fcn.min(), filter_fcn.max()))
    print('Interval length: {:.2f}'.format(step))
    print('Overlap: {:.2f}'.format(overlap))
    print('Magic fudge: {:.2f}'.format(magic_fudge))

    r_step = step * (1. - overlap / 100.)
    for i in np.arange(filter_fcn.min(), filter_fcn.max() + r_step, r_step):
        print('Filter indices from range [{:.2f}-{:.2f}]'.format(i, i + step))
        # Select the points in this filter range
        idx = np.nonzero((i <= filter_fcn) & (filter_fcn <= i + step))[0]
        num_points = len(idx)

        # Don't grow graph if we have too few points
        if num_points <= 5:
            continue

        I, R, simp1, F = barcode_linkage_noplex(d[:, idx][idx])

        lens = I[1, :] - I[0, :]
        lens[lens == np.inf] = R

        num_bin, _ = np.histogram(lens, magic_fudge)

        z = np.nonzero(num_bin == 0)[0]
        if len(z) == 0:
            num_clusters = 1
        else:
            num_clusters = num_bin[z[0]:len(num_bin)].sum()

        # Get the largest value(s) in the lens
        long_idx = np.argsort(lens)
        long_idx = long_idx[::-1][:num_clusters]

        min_death_time = I[1, long_idx].min()
        if min_death_time == np.inf:
            si = np.argsort(I[1, :])[::-1]
            min_death_time = I[1, :][si][1] + np.spacing(1)

        simp1 = simp1[:, F < min_death_time]

        G = np.eye(num_points)
        num_simps = simp1.shape[1]
        for j in range(num_simps):
            G[simp1[0, j], simp1[1, j]] = 1
            G[simp1[1, j], simp1[0, j]] = 1

        node_colors = color_graph(G)

        num_graph_nodes = len(node_info)
        level_idx[level] = []
        for j in set(node_colors):
            curr_color = j + num_graph_nodes
            level_idx[level].append(curr_color)
            node_info[curr_color]['level'] = level
            node_info[curr_color]['fnval'] = i
            node_info[curr_color]['set'] = set(idx[node_colors == j])
            node_info[curr_color]['filter'] = filter_fcn[idx[node_colors == j]].max()

        if level > 1:
            prev_lvl_idx = level_idx[level - 1]
            this_lvl_idx = level_idx[level]
            for i1 in range(1, len(prev_lvl_idx)):
                for i2 in range(1, len(this_lvl_idx)):
                    a = prev_lvl_idx[i1]
                    b = this_lvl_idx[i2]
                    if len(node_info[a]['set'] & node_info[b]['set']) > 0:
                        adja[(a, b)] = 1
                        adja[(b, a)] = 1

        level += 1

    print('done')

    return adja, node_info, level_idx


def barcode_linkage_noplex(dist):
    true = np.ones_like(dist) == 1
    LM_cond = dist[np.tril(true, -1).T]
    Z = hierarchy.linkage(LM_cond)
    R = dist.max()
    N = len(dist)

    # The distance between clusters Z[i, 0] and Z[i, 1] is given by Z[i, 2]
    I = np.zeros((2, len(Z[:, 2]) + 1))
    I[1, :len(Z[:, 2])] = Z[:, 2]
    I[1, len(Z[:, 2])] = np.inf

    D = np.triu(dist <= R)

    II, JJ = np.nonzero(D)
    Ca = np.zeros((2, len(II)), dtype=int)
    Ca[0, :] = II
    Ca[1, :] = JJ

    ind = np.ravel_multi_index((Ca[0, :], Ca[1, :]), dims=(N, N))

    F = dist.flat[ind]

    return I, R, Ca, F


def color_graph(G):
    num_nodes = G.shape[0]
    node_colors = np.zeros(num_nodes)
    color = 1
    q = [0]
    node_colors[q[0]] = color

    while (node_colors == 0).any():
        while len(q) > 0:
            current_node = q.pop(0)
            connected_nodes = G[current_node].nonzero()[0]
            for i in range(len(connected_nodes)):
                if node_colors[connected_nodes[i]] == 0:
                    node_colors[connected_nodes[i]] = color
                    q.extend(list(connected_nodes))

        colorless = (node_colors == 0).nonzero()[0]
        if len(colorless) > 0:
            q = [colorless[0]]
            color += 1
            node_colors[q[0]] = color

    return node_colors
