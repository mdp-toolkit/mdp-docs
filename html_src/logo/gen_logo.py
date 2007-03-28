import mdp, numpy, pylab, Image, ImageDraw, ImageFont

fontPath = "Verdana.ttf"
fontSize = 300
imgsize = (865, 400)
imgorig = (50, 0)
text = "MDP"
N = 10000 # data points
filling = 0.9 # probability that a data point belongs to the foreground
max_nodes = 500

# set random seed
numpy.random.seed(1)
print 'Generate image.'
image = Image.new("F", imgsize)
draw = ImageDraw.Draw(image)
draw.text(imgorig, text, font=ImageFont.truetype(fontPath, fontSize), fill=1)
image = image.transpose(Image.FLIP_TOP_BOTTOM)

print 'Create 2-D data distribution.'
# read image as array (1 is black, 0 is white)
x = numpy.fromstring(image.tostring(), numpy.float32)
x.shape = (image.size[1], image.size[0])
del image, draw

# create 2-D data distribution correspondent to the image

# shuffled lists of indices of the foreground and background pixels 
idx_fg = x.nonzero()
idx_bg = (1.-x).nonzero()
idx_fg = zip(idx_fg[1], idx_fg[0])
idx_bg = zip(idx_bg[1], idx_bg[0])
numpy.random.shuffle(idx_fg)
numpy.random.shuffle(idx_bg)
N_fg = int(N*filling) 

# choose N_fg pixels from fg and N-N_fg pixels from bg
idx = idx_fg[:N_fg] + idx_bg[:N-N_fg]
numpy.random.shuffle(idx)
data = numpy.array(idx).astype('d')

print 'Learning neural gas.'
# neural gas node
gng = mdp.nodes.GrowingNeuralGasNode(max_nodes=max_nodes)
gng.train(data)
gng.stop_training()

# plot resulting network
lines = []
for e in gng.graph.edges:
    x0, y0 = e.head.data.pos
    x1, y1 = e.tail.data.pos
    lines.extend(([x0,x1], [y0,y1], "r-",
                  [x0,x1], [y0,y1], "r."))

print 'Plot.'
pylab.plot(*lines)
pylab.plot(data[::10,0], data[::10,1], "k.")
pylab.axis('scaled')
raw_input('Press ENTER to quit!\n')
