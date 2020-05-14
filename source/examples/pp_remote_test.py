"""
Test the NetworkPPScheduler.

This script only works on *nix systems. On Windows systems you can create
the slave servers and the pp scheduler instance manually and then use it with
MDP's standard PPScheduler.
"""

import numpy as np

import mdp
import mdp.parallel as parallel

# start a single pp slave server on localhost with support for two cores 
remote_slaves = [("localhost", 2)]

# IMPORTANT: modify these paths to make sure that MDP and the ppserver.py
#    module (included in pp) can be importet on the slave machines.
sys_paths = ["/home/niko/win_workspace/mdp-toolkit",
             "/home/niko/win_workspace/Parallel\ Python/src"]

def main():
    # create the scheduler, which in turn starts the remote slave servers
    with parallel.pp_support.NetworkPPScheduler(
                                    remote_slaves=remote_slaves,
                                    source_paths=sys_paths,
                                    verbose=False) as scheduler:
        # test simple tasks
        for i in range(30):
            scheduler.add_task(i, parallel.SqrTestCallable())
        results = scheduler.get_results()
        # check result
        results.sort()
        results = np.array(results)
        assert np.all(results[:6] == np.array([0,1,4,9,16,25]))
        print "simple test done, now testing with MDP flow..."
        # simple test with MDP flow
        flow = mdp.parallel.ParallelFlow([mdp.nodes.PCANode()])
        data = [np.random.random([50, 10]) for _ in range(5)]
        flow.train([data], scheduler)
        print "done."
   
if __name__ == "__main__":
    print mdp.config.info()
    main()

