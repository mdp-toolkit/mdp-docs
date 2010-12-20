import mdp
import pylab
from matplotlib import ticker, axes3d



#################################################
# Testing Functions
#################################################

def s_distr(npoints, hole=False):
    """Return a 3D S-shaped surface. If hole is True, the surface has
    a hole in the middle."""
    t = mdp.numx_rand.random(npoints)
    y = mdp.numx_rand.random(npoints)*5.
    theta = 3.*mdp.numx.pi*(t-0.5)
    x = mdp.numx.sin(theta)
    z = mdp.numx.sign(theta)*(mdp.numx.cos(theta) - 1.)
    if hole:
        indices = mdp.numx.where( ((0.3>t) | (0.7<t)) | ((1.0>y) | (4.0<y)) )
        return x[indices], y[indices], z[indices], t[indices]
    else:
        return x, y, z, t

def scatter_2D(x,y,t=None,cmap=pylab.cm.jet):
    #fig = pylab.figure()
    pylab.subplot(212)
    if t==None:
        pylab.scatter(x,y)
    else:
        pylab.scatter(x,y,c=t,cmap=cmap)

    pylab.xlabel('x')
    pylab.ylabel('y')

def scatter_3D(x,y,z,t=None,cmap=pylab.cm.jet):
    fig = pylab.figure

    if t==None:
        ax.scatter3D(x,y,z)
    else:
        ax.scatter3D(x,y,z,c=t,cmap=cmap)

    if x.min()>-2 and x.max()<2:
        ax.set_xlim(-2,2)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    # elev, az
    ax.view_init(10, -80)

n, k = 1000, 15
x, y, z, t = s_distr(n, hole=False)
data = mdp.numx.array([x,y,z]).T
lle_projected_data = mdp.nodes.LLENode(k, output_dim=2)(data)

#plot input in 3D
fig = pylab.figure(1, figsize=(6,4))
pylab.clf()
ax = axes3d.Axes3D(fig)
ax.scatter3D(x, y, z, c=t, cmap=pylab.cm.jet)
ax.set_xlim(-2, 2)
ax.view_init(10, -80)
pylab.draw()
pylab.savefig('s_shape_3D.png')

#plot projection in 2D
pylab.clf()
projection = lle_projected_data
pylab.scatter(projection[:,0],\
              projection[:,1],\
              c=t,cmap=pylab.cm.jet)
pylab.savefig('s_shape_lle_proj.png')


# ### with hole
x, y, z, t = s_distr(n, hole=True)
data = mdp.numx.array([x,y,z]).T
lle_projected_data = mdp.nodes.LLENode(k, output_dim=2)(data)

#plot input in 3D
fig = pylab.figure(1, figsize=(6,4))
pylab.clf()
ax = axes3d.Axes3D(fig)
ax.scatter3D(x, y, z, c=t, cmap=pylab.cm.jet)
ax.set_xlim(-2, 2)
ax.view_init(10, -80)
pylab.draw()
pylab.savefig('s_shape_hole_3D.png')

#plot projection in 2D
pylab.clf()
projection = lle_projected_data
pylab.scatter(projection[:,0],\
              projection[:,1],\
              c=t,cmap=pylab.cm.jet)
pylab.savefig('s_shape_hole_lle_proj.png')

hlle_projected_data = mdp.nodes.HLLENode(k, output_dim=2)(data)

#plot projection in 2D
pylab.clf()
projection = hlle_projected_data
pylab.scatter(projection[:,0],\
              projection[:,1],\
              c=t,cmap=pylab.cm.jet)
pylab.savefig('s_shape_hole_hlle_proj.png')
