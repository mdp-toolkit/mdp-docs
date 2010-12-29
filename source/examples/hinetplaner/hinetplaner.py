
# TODO: make data exchange with simulations possible,
#    maybe use same format, so that one can simply cutnpaste into a config file

# TODO: svg creation returns list of element ids,
#    problem: return value is already occupied by the svg xml

# TODO: copy MDP hinet style over the local style here
#    check at startup if file is present, if not create it

# ------

# TODO: add checkbox for ignore_cover

# TODO: register click callback globaly for each SVG, the use the event
#    target attrubute to identify the element and send that id to the
#    server

# TODO: use div reference when using ids in jQuery for speedup
#    also store direct references to the SVG elements in arrays for speedup

# TODO: make SFA, PCA plugin for layer, can combine with any switchboard,
#    simply have two selectors, one for board, one for algorithm

# TODO: mid-term: make configs exchangeable with model?
#    make even model flows accessible or instead integrate GUI into model? 

# TODO: longterm: deal with state problem for multiple frontends,
#    spawn threads or processes, each serving one frontend?
#    or just a dict with the flows depending on the ids?
#    assign an id to each frontend at the start?
#
#    use a timeout (10min) to delete state if fronend does not request,
#    can then still use show_flow in the fronend to recreate the state?

# Note:  throbbers created at http://www.loadinfo.net
#    throbber 16_cycle_one_24

import cStringIO as StringIO
import json
import os
import fnmatch

import mdp
from mdp.hinet import (
    ChannelSwitchboard, Rectangular2dSwitchboard, DoubleRect2dSwitchboard,
    DoubleRhomb2dSwitchboard, get_2d_image_switchboard
)

import switchboard_svg
mdp.activate_extensions(["switchboard_factory", "svg"])

# keys are nice switchboard standard names, values are the types
SWITCHBOARD_TYPES_DICT = {
    "standard rect": Rectangular2dSwitchboard,
    "double rect": DoubleRect2dSwitchboard,
    "double rhomb": DoubleRhomb2dSwitchboard,
    "single node": ChannelSwitchboard
}
# ordered list of switchboard names for the UI
SWITCHBOARD_NAMES = ["standard rect", "double rect",
                     "double rhomb", "single node"] 

# SVG style
SVG_BASE_SIZE = 3  # size of the little channel rectangles
SVG_GAP_SIZE = 1  # size of the gap between them
SVG_CLASS_NAME = "cov_normal"  # default class name
SVG_ID_PREFIX = "lcov"

# path where the JSON hinet configs are stored
CONFIG_PATH = "configs"  

# state variables
flow = None

## JSON RPC functions

def get_layer_params():
    """Return the layer type paramter information."""
    layer_names = []
    layer_params = {}
    for switchboard_name in SWITCHBOARD_NAMES:
        layer_names.append(switchboard_name)
        params = SWITCHBOARD_TYPES_DICT[switchboard_name].free_parameters[:]
        if "ignore_cover" in params:
            params.remove("ignore_cover")
        params += ["sfa_dim", "sfa2_dim"]
        layer_params[switchboard_name] = params
    return {"layer_names": layer_names, "layer_params": layer_params}

def get_hinet(hinet_config):
    """Return all the network information for a given parameter set.
    
    hinet_config -- Dict containing all the configuration info:
        'image_size': Size of the input image.
        'layer_configs': List of parameter dictionaries.
    """
    global flow
    layers = []
    prev_layer = None
    image_size = hinet_config["image_size"]
    for layer_config in hinet_config["layer_configs"]:
        if not prev_layer:
            # create fake layer, corresponding to image
            prev_layer = mdp.hinet.FlowNode(mdp.Flow(
                [get_2d_image_switchboard(image_size)]))
        layers.append(get_sfa_layer(prev_layer, layer_config))
        prev_layer = layers[-1]
    flow = mdp.Flow(layers)
    ## create HTML view
    xhtml_file = StringIO.StringIO()
    hinet_translator = mdp.hinet.HiNetXHTMLTranslator()
    hinet_translator.write_flow_to_file(flow=flow, xhtml_file=xhtml_file)
    ## create coverage layer representation
    coverage_svgs = []
    coverage_ids = []
    coverage_svgs.append(switchboard_svg.image_svg_representation(
                                             image_size=image_size,
                                             id_prefix=SVG_ID_PREFIX + "_0",
                                             element_size=SVG_BASE_SIZE,
                                             element_gap=SVG_GAP_SIZE,
                                             element_class=SVG_CLASS_NAME))
    coverage_ids.append([])  # the pixels do not support coverage
    # now the layers
    for i_layer, layer in enumerate(layers):
        svg_id_prefix = SVG_ID_PREFIX + "_%d" % (i_layer+1)
        coverage_svgs.append(layer[0].svg_representation(
                                id_prefix=svg_id_prefix,
                                element_size=SVG_BASE_SIZE * (i_layer+2),
                                element_gap=SVG_GAP_SIZE,
                                element_class=SVG_CLASS_NAME))
        coverage_ids.append([svg_id_prefix + "_%d" % i
                             for i in range(layer[0].output_channels)])
    return {
        "html_view": xhtml_file.getvalue(),
        "hinet_coverage_svgs": coverage_svgs,
        "hinet_coverage_ids": coverage_ids,
        "hinet_config_str": json.dumps(hinet_config, sort_keys=True, indent=4)
    }

def get_layer_coverage(layer, channel):
    """Return the id's of the elements which are affected by the given channel.
    
    layer -- Index of the layer.
    channel -- Inndex of the channel.
    """
    active_channels = [[] for _ in range(len(flow) + 1)]
    active_channels[layer] = [channel] 
    for i in range(layer-1, -1, -1):
        active_channels[i] = flow[i][0].get_out_channels_input_channels(
                                                        active_channels[i+1])
    return ["lcov_%d_%d" % (i,j) for i in range(len(active_channels))
                                 for j in active_channels[i]]
    
def get_available_hinet_configs():
    config_names = []
    for path, dirs, files in os.walk(CONFIG_PATH):
        dirs.sort()
        files.sort()
        for filename in fnmatch.filter(files, '*.json'):
            # TODO: incorporate path?
            config_names.append(filename[:-5])
    return config_names

def save_hinet_config(layer_configs, config_name):
    try:
        os.makedirs(CONFIG_PATH)
    except:
        pass
    config_file = open(os.path.join(CONFIG_PATH, config_name + '.json'), 'w')
    try:
        json.dump(layer_configs, config_file, sort_keys=True, indent=4)
    finally:
        config_file.close()
    return get_available_hinet_configs()
    
def get_hinet_config(config_name):
    config_file = open(os.path.join(CONFIG_PATH, config_name + '.json'),
                       'r')
    result = json.load(config_file)
    return result
    

## helper functions ##

def get_sfa_layer(prev_layer, layer_config):
    switchboard_class = SWITCHBOARD_TYPES_DICT[layer_config["layer_type"]]
    switchboard_params = {}
    for key in switchboard_class.free_parameters:
        if key in layer_config:
            switchboard_params[key] = layer_config[key]
    switchboard = switchboard_class.create_switchboard(
                                    free_params=switchboard_params,
                                    prev_switchboard=prev_layer[0],
                                    prev_output_dim=prev_layer.output_dim)
    sfa_input_dim = switchboard.out_channel_dim
    sfa_node = mdp.nodes.SFANode(input_dim=sfa_input_dim, 
                                 output_dim=layer_config["sfa_dim"])
    sfa2_node = mdp.nodes.SFA2Node(input_dim=layer_config["sfa_dim"], 
                                   output_dim=layer_config["sfa2_dim"])
    flownode = mdp.hinet.FlowNode(mdp.Flow([sfa_node, sfa2_node]))
    sfa_layer = mdp.hinet.CloneLayer(flownode, 
                                     n_nodes=switchboard.output_channels)
    return mdp.hinet.FlowNode(mdp.Flow([switchboard, sfa_layer]))
