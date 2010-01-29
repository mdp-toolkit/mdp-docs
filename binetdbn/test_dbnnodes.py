
import unittest

import mdp
n = mdp.numx

import binet

import dbn_binodes


class TestDBNBiNode(unittest.TestCase):

    def test_dbn(self):
        n_layers = 2
        nodes = [binet.SenderBiNode()]
        dbn_ids = ["dbn_%d" % (i+1) for i in range(n_layers)]
        nodes += [dbn_binodes.DBNLayerBiNode(node_id=dbn_id)
                  for dbn_id in dbn_ids]
        nodes.append(dbn_binodes.DBNMasterBiNode(dbn_ids=dbn_ids,
                                                 node_id="dbn_master"))
        flow = binet.BiFlow(nodes)
        
        # TODO: finish this unittest
        
        error = flow["dbn_master"].error 
        