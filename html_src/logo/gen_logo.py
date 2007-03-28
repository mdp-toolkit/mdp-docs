import mdp, numpy, pylab

fontPath = "Verdana.ttf"
fontSize = 300
imgsize = (865, 400)
imgorig = (50, 0)
text = "MDP"
N = 10000 # data points
filling = 0.9 # probability that a data point belongs to the foreground
max_nodes = 500
new_image = False

# set random seed
numpy.random.seed(1)

if new_image:
    import Image, ImageDraw, ImageFont
    print 'Generate image.'
    image = Image.new("F", imgsize)
    draw = ImageDraw.Draw(image)
    draw.text(imgorig,text,font=ImageFont.truetype(fontPath,fontSize),fill=1)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    fl = file('text2.raw', 'wb')
    fl.write(image.tostring())
    fl.close()
    del image, draw

print 'Load image.'
fl = file('text.raw', 'rb')
imgstr = fl.read()
fl.close()

print 'Create 2-D data distribution.'
# read image as array (1 is black, 0 is white)
x = numpy.fromstring(imgstr, numpy.float32)
x.shape = (imgsize[1], imgsize[0])


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
