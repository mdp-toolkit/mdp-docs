import mdp
dir(mdp.helper_funcs)
# ['__builtins__', '__doc__', '__file__', '__name__',
# 'cubica', 'fastica', 'get_eta', 'mdp', 'pca', 'sfa', 'whitening']
pcanode1 = mdp.nodes.PCANode()
pcanode1
# PCANode(input_dim=None, output_dim=None, typecode='None')
pcanode2 = mdp.nodes.PCANode(output_dim = 10)
pcanode2
# PCANode(input_dim=None, output_dim=10, typecode='None')
pcanode3 = mdp.nodes.PCANode(output_dim = 0.8)
pcanode3.desired_variance
# 0.80000000000000004
pcanode4 = mdp.nodes.PCANode(typecode = 'f')
pcanode4
# PCANode(input_dim=None, output_dim=None, typecode='f')
pcanode4.get_supported_typecodes()
# ['f', 'd']
expnode = mdp.nodes.PolynomialExpansionNode(3)
x = mdp.numx_rand.random((100, 25))  # 25 variables, 100 observations
pcanode1.train(x)
pcanode1
# PCANode(input_dim=25, output_dim=None, typecode='d')
for i in range(100):
    x = mdp.numx_rand.random((100, 25))
    pcanode1.train(x)
# >>>
expnode.is_trainable()
# 0
pcanode1.stop_training()
pcanode3.train(x)
pcanode3.stop_training()
pcanode3.output_dim
# 16
pcanode3.explained_variance
# 0.85261144755506446
avg = pcanode1.avg            # mean of the input data
v = pcanode1.get_projmatrix() # projection matrix
x = mdp.numx_rand.random((100, 25))
y_pca = pcanode1.execute(x)
y_pca = pcanode1(x)
x = mdp.numx_rand.random((100, 5))
y_exp = expnode(x)
pcanode1.is_invertible()
# 1
x = pcanode1.inverse(y_pca)
expnode.is_invertible()
# 0
class TimesTwoNode(mdp.Node):
    def is_trainable(self): return 0
    def _execute(self, x):
        return self._scast(2)*x
    def _inverse(self, y):
        return y/self._scast(2)
# ...
# >>>
node = TimesTwoNode(typecode = 'i')
x = mdp.numx.array([[1.0, 2.0, 3.0]])
y = node(x)
print x, '* 2 =  ', y
# [ [ 1.  2.  3.]] * 2 =   [ [2 4 6]]
print y, '/ 2 =', node.inverse(y)
# [ [2 4 6]] / 2 = [ [1 2 3]]
class PowerNode(mdp.Node):
    def __init__(self, power, input_dim=None, typecode=None):
        super(PowerNode, self).__init__(input_dim=input_dim, typecode=typecode)
        self.power = power
    def is_trainable(self): return 0
    def is_invertible(self): return 0
    def get_supported_typecodes(self):
        return ['f', 'd']
    def _execute(self, x):
        return self._refcast(x**self._scast(self.power))
# ...
# >>>
node = PowerNode(3)
x = mdp.numx.array([[1.0, 2.0, 3.0]])
y = node.execute(x)
print x, '**', node.power, '=', node(x)
# [ [ 1.  2.  3.]] ** 3 = [ [  1.   8.  27.]]
class MeanFreeNode(mdp.Node):
    def __init__(self, input_dim=None, typecode=None):
        super(MeanFreeNode, self).__init__(input_dim=input_dim,
                                           typecode=typecode)
        self.avg = None
        self.tlen = 0
    def _train(self, x):
        # Initialize the mean vector with the right
        # size and typecode if necessary:
        if self.avg is None:
            self.avg = mdp.numx.zeros(self.get_input_dim(),
                                      typecode=self.get_typecode())
        self.avg += sum(x, 0)
        self.tlen += x.shape[0]
    def _stop_training(self):
        self.avg /= self._scast(self.tlen)
    def _execute(self, x):
        return self._refcast(x - self.avg)
    def _inverse(self, y):
        return self._refcast(y + self.avg)
# ...
# >>>
node = MeanFreeNode()
x = mdp.numx_rand.random((10,4))
node.train(x)
y = node.execute(x)
print 'Mean of y (should be zero): ', mdp.utils.mean(y, 0)
# Mean of y (should be zero):  [  0.00000000e+00   2.22044605e-17
# -2.22044605e-17   1.11022302e-17]
class TwiceNode(mdp.Node):
    def is_trainable(self): return 0
    def is_invertible(self): return 0
    def _set_input_dim(self, n):
        self._input_dim = n
        self._output_dim = 2*n
    def _set_output_dim(self, n):
        raise mdp.NodeException, "Output dim can not be explicitly set!"
    def _execute(self, x):
        return mdp.numx.concatenate((x, x),1)
# ...
# >>>
node = TwiceNode()
x = mdp.numx.zeros((5,2))
x
# array([[0, 0],
# [0, 0],
# [0, 0],
# [0, 0],
# [0, 0]])
node.execute(x)
# array([[0, 0, 0, 0],
# [0, 0, 0, 0],
# [0, 0, 0, 0],
# [0, 0, 0, 0],
# [0, 0, 0, 0]])
plot = scipy.gplt.plot
class VisualizeNode(mdp.Node):
    def is_trainable(self): return 0
    def is_invertible(self): return 0
    def execute(self, x):
        mdp.Node.execute(self,x)
        self._refcast(x)
        plot(x)
        return x
# >>>
inp = mdp.numx_rand.random((1000,20))
inp = (inp - mdp.utils.mean(inp,0))/mdp.utils.std(inp,0)
inp[:,5:] /= 10.0
x = mdp.utils.mult(inp,mdp.numx_rand.random((20,20)))
plot(x)
pca = mdp.nodes.PCANode(output_dim=5)
pca.train(x)
out1 = pca.execute(x)
plot(out1)
ica = mdp.nodes.CuBICANode()
ica.train(out1)
out2 = ica.execute(out1)
plot(out2)
flow = mdp.Flow([VisualizeNode(),
                       mdp.nodes.PCANode(output_dim=5),
                       VisualizeNode(),
                       mdp.nodes.CuBICANode(),
                       VisualizeNode()])
# ...
flow.train(x)
out = flow.execute(x)
cov = mdp.utils.amax(abs(mdp.utils.cov(inp[:,:5],out)))
print cov
# [ 0.99324451  0.99724133  0.99247439  0.99049607  0.994309  ]
rec = flow[1::2].inverse(out)
cov = mdp.utils.amax(abs(mdp.utils.cov(x/mdp.utils.std(x,0),
                                       rec/mdp.utils.std(rec,0))))
print cov
# [ 0.99839606  0.99744461  0.99616208  0.99772863  0.99690947
# 0.99864056  0.99734378  0.98722502  0.98118101  0.99407939
# 0.99683096  0.99756988  0.99664384  0.99723419  0.9985529
# 0.99829763  0.9982712   0.99721741  0.99682906  0.98858858]
for node in flow:
    print repr(node)
# ...
# VisualizeNode(input_dim=20, output_dim=20, typecode='d')
# PCANode(input_dim=20, output_dim=5, typecode='d')
# VisualizeNode(input_dim=5, output_dim=5, typecode='d')
# CuBICANode(input_dim=5, output_dim=5, typecode='d')
# VisualizeNode(input_dim=5, output_dim=5, typecode='d')
# >>>
len(flow)
# 5
nodetoberemoved = flow.pop(-1)
nodetoberemoved
# VisualizeNode(input_dim=5, output_dim=5, typecode='d')
len(flow)
# 4
dummyflow = flow[3:].copy()
longflow = flow + dummyflow
len(longflow)
# 5
flow
# Flow([VisualizeNode(input_dim=20, output_dim=20, typecode='d'),
# PCANode(input_dim=20, output_dim=5, typecode='d'),
# VisualizeNode(input_dim=5, output_dim=5, typecode='d'),
# CuBICANode(input_dim=5, output_dim=5, typecode='d')])
flow.pop(1)
# Traceback (most recent call last):
# File "<stdin>", line 1, in ?
# [...]
# ValueError: dimensions mismatch: 20 != 5
class BogusExceptNode(mdp.Node):
   def train(self,x):
       self.bogus_attr = 1
       raise Exception, "Bogus Exception"
   def execute(self,x):
       raise Exception, "Bogus Exception"
# ...
flow = mdp.Flow([BogusExceptNode()])
flow.set_crash_recovery(1)
flow.train([[None]])
# Traceback (most recent call last):
# File "<stdin>", line 1, in ?
# [...]
# mdp.linear_flows.FlowExceptionCR:
# ----------------------------------------
# ! Exception in node #0 (BogusExceptNode):
# Node Traceback:
# Traceback (most recent call last):
# [...]
# Exception: Bogus Exception
# ----------------------------------------
# A crash dump is available on: "/tmp/MDPcrash_LmISO_.pic"
flow.set_crash_recovery('/home/myself/mydumps/MDPdump.pic')
BogusNode = mdp.IdentityNode
class BogusNode2(mdp.IdentityNode):
    """This node does nothing. but it's not trainable and not invertible.
    """
    def is_trainable(self): return 0
    def is_invertible(self): return 0
# ...
# >>>
def gen_data(blocks):
    progressbar = mdp.utils.ProgressBar(0,blocks)
    progressbar.update(0)
    for i in xrange(blocks):
        block_x = mdp.utils.atleast_2d(mdp.numx.arange(2,1001,2))
        block_y = mdp.utils.atleast_2d(mdp.numx.arange(1,1001,2))
        # put variables on columns and observations on rows
        block = mdp.numx.transpose(mdp.numx.concatenate([block_x,block_y]))
        progressbar.update(i+1)
        yield block
    print '\n'
    return
# ...
# >>>
flow = mdp.Flow([BogusNode(),BogusNode()],verbose=1)
flow.train([gen_data(5000),gen_data(3000)])
# Training node #0 (IdentityNode)
# [===================================100%==================================>]
flow = mdp.Flow([BogusNode(),BogusNode()])
block_x = mdp.utils.atleast_2d(mdp.numx.arange(2,1001,2))
block_y = mdp.utils.atleast_2d(mdp.numx.arange(1,1001,2))
single_block = mdp.numx.transpose(mdp.numx.concatenate([block_x,block_y]))
flow.train(single_block)
flow = mdp.Flow([BogusNode2(),BogusNode()], verbose=1)
flow.train([None,gen_data(5000)])
# Training node #0 (BogusNode2)
# Training finished
# Training node #1 (IdentityNode)
# [===================================100%==================================>]
flow = mdp.Flow([BogusNode2(),BogusNode()], verbose=1)
flow.train(single_block)
# Training node #0 (BogusNode2)
# /.../linear_flows.py:94: MDPWarning:
# ! Node 0 in not trainable
# You probably need a 'None' generator for this node. Continuing anyway.
# warnings.warn(wrnstr, mdp.MDPWarning)
# Training finished
# Training node #1 (IdentityNode)
# Training finished
# Close the training phase of the last node
import warnings
warnings.filterwarnings("ignore",'.*',mdp.MDPWarning)
flow = mdp.Flow([BogusNode2(),BogusNode()], verbose=1)
flow.train(single_block)
# Training node #0 (BogusNode2)
# Training finished
# Training node #1 (IdentityNode)
# Training finished
# Close the training phase of the last node
warnings.filterwarnings("always",'.*',mdp.MDPWarning)
flow = mdp.Flow([BogusNode(),BogusNode()], verbose=1)
flow.train([gen_data(1), gen_data(1)])
# Training node #0 (BogusNode2)
# Training finished
# Training node #1 (IdentityNode)
# [===================================100%==================================>]
output = flow.execute(single_block)
output = flow.inverse(single_block)
def gen_data(blocks,dims):
    mat = mdp.numx_rand.random((dims,dims))-0.5
    for i in xrange(blocks):
        # put variables on columns and observations on rows
        block = mdp.utils.mult(mdp.numx_rand.random((1000,dims)), mat)
        yield block
    return
# ...
# >>>
pca = mdp.nodes.PCANode(output_dim=0.9)
exp = mdp.nodes.PolynomialExpansionNode(2)
sfa = mdp.nodes.SFANode()
class PCADimensionExceededException(Exception):
    """Exception base class for PCA exceeded dimensions case."""
    pass
# ...
# >>>
class CheckPCA(mdp.CheckpointFunction):
    def __init__(self,max_dim):
        self.max_dim = max_dim
    def __call__(self,node):
        node.stop_training()
        act_dim = node.get_output_dim()
        if act_dim > self.max_dim:
            errstr = 'PCA output dimensions exceeded maximum '+\
                     '(%d > %d)'%(act_dim,self.max_dim)
            raise PCADimensionExceededException, errstr
        else:
            print 'PCA output dimensions = %d'%(act_dim)
# ...
# >>>
flow = mdp.CheckpointFlow([pca, exp, sfa])
flow.train([gen_data(10, 50), None, gen_data(10, 50)],
           [CheckPCA(10), None, None])
# Traceback (most recent call last):
# File "<stdin>", line 2, in ?
# [...]
# __main__.PCADimensionExceededException: PCA output dimensions exceeded maximum (25 > 10)
flow[0] = mdp.nodes.PCANode(output_dim=0.9)
flow.train([gen_data(10, 12), None, gen_data(10, 12)],
           [CheckPCA(10), None, None])
# PCA output dimensions = 6
pca = mdp.nodes.PCANode(output_dim=0.9)
exp = mdp.nodes.PolynomialExpansionNode(2)
sfa = mdp.nodes.SFANode()
flow = mdp.CheckpointFlow([pca, exp, sfa])
flow.train([gen_data(10, 12), None, gen_data(10, 12)],
           [CheckPCA(10),
            None,
            mdp.CheckpointSaveFunction('dummy.pic',
                                       stop_training = 1,
                                       protocol = 0)])
# ...
# PCA output dimensions = 7
fl = file('dummy.pic')
import cPickle
sfa_reloaded = cPickle.load(fl)
sfa_reloaded
# SFANode(input_dim=35, output_dim=35, typecode='d')
fl.close()
import os
os.remove('dummy.pic')
p2 = mdp.numx.pi*2
t = mdp.utils.linspace(0,1,10000,endpoint=0) # time axis 1s, samplerate 10KHz
dforce = mdp.numx.sin(p2*5*t) + mdp.numx.sin(p2*11*t) + mdp.numx.sin(p2*13*t)
def logistic_map(x,r):
    return r*x*(1-x)
# ...
# >>>
series = mdp.numx.zeros((10000,1),'d')
series[0] = 0.6
for i in range(1,10000):
    series[i] = logistic_map(series[i-1],3.6+0.13*dforce[i])
# ...
# >>>
sequence = [mdp.nodes.EtaComputerNode(),
            mdp.nodes.TimeFramesNode(10),
            mdp.nodes.PolynomialExpansionNode(3),
            mdp.nodes.SFANode(output_dim=1),
            mdp.nodes.EtaComputerNode()]
# ...
# >>>
flow = mdp.Flow(sequence, verbose=1)
flow.train(series)
slow = flow.execute(series)
resc_dforce = (dforce - mdp.utils.mean(dforce,0))/mdp.utils.std(dforce,0)
mdp.utils.cov(resc_dforce[:-9],slow)
# 0.99992501533859179
print 'Eta value (time-series): ', flow[0].get_eta(t=10000)
# Eta value (time-series):  [ 3002.53380245]
print 'Eta value (slow feature): ', flow[-1].get_eta(t=9996)
# Eta value (slow feature):  [ 10.2185087]
mdp.numx_rand.seed(1266090063, 1644375755)
def uniform(min_, max_, dims):
    """Return a random number between min_ and max_ ."""
    return mdp.numx_rand.random(dims)*(max_-min_)+min_
# ...
def circumference_distr(center, radius, n):
    """Return n random points uniformly distributed on a circumference."""
    phi = uniform(0, 2*mdp.numx.pi, (n,1))
    x = radius*mdp.numx.cos(phi)+center[0]
    y = radius*mdp.numx.sin(phi)+center[1]
    return mdp.numx.concatenate((x,y), axis=1)
# ...
def circle_distr(center, radius, n):
    """Return n random points uniformly distributed on a circle."""
    phi = uniform(0, 2*mdp.numx.pi, (n,1))
    sqrt_r = mdp.numx.sqrt(uniform(0, radius*radius, (n,1)))
    x = sqrt_r*mdp.numx.cos(phi)+center[0]
    y = sqrt_r*mdp.numx.sin(phi)+center[1]
    return mdp.numx.concatenate((x,y), axis=1)
# ...
def rectangle_distr(center, w, h, n):
    """Return n random points uniformly distributed on a rectangle."""
    x = uniform(-w/2., w/2., (n,1))+center[0]
    y = uniform(-h/2., h/2., (n,1))+center[1]
    return mdp.numx.concatenate((x,y), axis=1)
# ...
N = 2000
cf1 = circumference_distr([6,-0.5], 2, N)
cf2 = circumference_distr([3,-2], 0.3, N)
cl1 = circle_distr([-5,3], 0.5, N/2)
cl2 = circle_distr([3.5,2.5], 0.7, N)
# - Rectangles:
# ::
x = mdp.numx.concatenate([cf1, cf2, cl1, cl2, r1,r2,r3,r4], axis=0)
x = mdp.numx.take(x,mdp.numx_rand.permutation(x.shape[0]))
gng = mdp.nodes.GrowingNeuralGasNode(max_nodes=75)
STEP = 500
for i in range(0,x.shape[0],STEP):
    gng.train(x[i:i+STEP])
    # [...] plotting instructions
# ...
gng.stop_training()
n_obj = len(gng.graph.connected_components())
# 5
