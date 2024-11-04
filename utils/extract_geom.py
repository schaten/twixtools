#!/usr/bin/env python
import argparse
import numpy as np

from bart import cfl
import twixtools

tol = 1e-6

def bart_geom(geom, radial):

    fov = [geom.fov[0], geom.fov[0] if radial else geom.fov[1]]

    # geom[0][j] = fov_readout * normalize(readout vector)
    # geom[1][j] = fov_phase * normalize(phase enc. vector)
    # geom[2][j] = slice offset
    # basis: patient coordinates

    vec_x = geom.prs_to_pcs()[:,0]
    vec_y = geom.prs_to_pcs()[:,1]

    assert(abs(np.dot(vec_x, vec_y)) < tol)
    assert(abs(1 - np.linalg.norm(vec_x)) < tol)
    assert(abs(1 - np.linalg.norm(vec_y)) < tol)

    geom = np.array([fov[0] * geom.prs_to_pcs()[:,0],
                     fov[1] * geom.prs_to_pcs()[:,1],
                     geom.offset])
    return geom

def bart_geom_slices(geom_array, slice_dim = 13, slice_order = None, radial = False):
    if slice_order is None:
        slice_order = list(range(len(geom_array)))
    return np.reshape(np.stack(
        [bart_geom(geom_array[i], radial) for i in slice_order],\
        axis=-1), tuple([3,3] + (slice_dim - 2) * [1] + [len(geom_array)]))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("datfile")
    parser.add_argument("outfile")
    parser.add_argument("--radial", action='store_true')
    args = parser.parse_args()

    twix = twixtools.read_twix(args.datfile, parse_geometry=True, parse_data=False)

    order = None
    if '-' != twix[-1]['hdr']['Config']['chronSliceIndices'][0]:
        order = []
        for x in twix[-1]['hdr']['Config']['chronSliceIndices']:
            if len(order) == len(twix[-1]['geometry']):
                break
            if x == ' ':
                continue
            val = int(x)
            order.append(val)

    geom = bart_geom_slices(twix[-1]['geometry'], slice_order = order, radial = args.radial)

    cfl.writecfl(args.outfile, geom)
