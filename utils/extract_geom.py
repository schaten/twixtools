#!/usr/bin/env python
import argparse
import numpy as np

from bart import cfl
import twixtools

radial = True


def bart_geom(geom):

    return np.array([geom.fov[1] * geom.prs_to_pcs()[:,1],
                     geom.fov[0] * geom.prs_to_pcs()[:,0],
                     geom.offset / np.linalg.norm(geom.offset)]).T

def bart_geom_slices(geom_array, slice_dim = 13):
    return np.reshape(np.stack(
        [bart_geom(s) for s in geom_array],\
        axis=-1), tuple([3,3] + (slice_dim - 2) * [1] + [2]))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("datfile")
    parser.add_argument("outfile")
    args = parser.parse_args()

    twix = twixtools.read_twix(args.datfile, parse_geometry=True, parse_data=False)

    geom = bart_geom_slices(twix[-1]['geometry'])

    cfl.writecfl(args.outfile, geom)
