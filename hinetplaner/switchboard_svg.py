"""
Extension for SVG representations of switchboards in a 2d hierarchical network.
"""

import mdp
from mdp.hinet import (
    ChannelSwitchboard, Rectangular2dSwitchboard, DoubleRect2dSwitchboard,
    DoubleRhomb2dSwitchboard
)


class SVGExtensionChannelSwitchboard(mdp.ExtensionNode, ChannelSwitchboard):
    """Extension node for SVG representations of channel switchboards."""
    
    extension_name = "svg"
    
    def svg_representation(self, id_prefix, element_size=1, element_gap=1,
                           element_class="coverage_element"):
        """Return an SVG representation of the switchboard.
        
        id_prefix -- Prefix string for the id values of the output elements.
        element_size -- Pixel size of the output elements (default 1). 
        gap_size -- Pixel size of the gap between output elements (default 1).
        element_class -- CSS class name for the output elements.
        
        This is the default implementation where the output elements are
        simply placed next to each other.
        """
        es = element_size
        gs = element_gap
        svg_width = self.output_channels * (es + gs)
        svg_height = es
        code_lines = ['<svg width="%dpx" height="%dpx"' %
                          (svg_width, svg_height) + 
                      ' xmlns="http://www.w3.org/2000/svg" version="1.1">']
        for i_channel in range(self.output_channels):
            code_lines.append(
                 '<rect id="%s_%d"' % (id_prefix, i_channel) +
                 ' class="%s"' % element_class +
                 ' width="%s" height="%s"' % (es,es) +
                 ' x="%d" y="%d"' % (i_channel*(es+gs), 0) +
                 ' />' )
        code_lines.append('</svg>')
        return "\n".join(code_lines)


def image_svg_representation(image_size, id_prefix, element_size=1,
                             element_gap=1, element_class="coverage_element"):
    """Return an SVG representation of the image.
    
    image_size -- Tuple with the width and height.
    
    This is a helper function to be used together with svg_representation
    for the following network layers.
    """
    es = element_size
    gs = element_gap
    svg_size = (image_size[0] * (es+gs), image_size[1] * (es+gs))
    code_lines = ['<svg width="%dpx" height="%dpx"' % svg_size + 
                  ' xmlns="http://www.w3.org/2000/svg" version="1.1">']
    for y in range(image_size[1]):
        for x in range(image_size[0]):
            i_channel = x + (y * image_size[0])
            code_lines.append('<rect id="%s_%d"' % (id_prefix, i_channel) +
                              ' class="%s"' % element_class +
                              ' width="%s" height="%s"' % (es, es) +
                              ' x="%d" y="%d"' % (x*(es+gs), y*(es+gs)) +
                              ' />' )
    code_lines.append('</svg>')
    return "\n".join(code_lines)
    
    
@mdp.extension_method("svg", Rectangular2dSwitchboard, "svg_representation")
@mdp.extension_method("svg", DoubleRhomb2dSwitchboard, "svg_representation")
def _rect2d_switchoard_svd(self, id_prefix, element_size=1, element_gap=1,
                           element_class="coverage_element"):
    es = element_size
    gs = element_gap
    svg_width = self.out_channels_xy[0] * (es + gs)
    svg_height = self.out_channels_xy[1] * (es + gs)
    code_lines = ['<svg width="%dpx" height="%dpx"' %
                  (svg_width, svg_height) + 
                  ' xmlns="http://www.w3.org/2000/svg" version="1.1">']
    for y in range(self.out_channels_xy[1]):
        for x in range(self.out_channels_xy[0]):
            i_channel = x + (y * self.out_channels_xy[0])
            code_lines.append(
                      '<rect id="%s_%d"' % (id_prefix, i_channel) +
                      ' class="%s"' % element_class +
                      ' width="%s" height="%s"' % (es, es) +
                      ' x="%d" y="%d"' % (x*(es+gs), y*(es+gs)) +
                       ' />' )
    code_lines.append('</svg>')
    return "\n".join(code_lines)

@mdp.extension_method("svg", DoubleRect2dSwitchboard, "svg_representation")
def _double_rect2d_switchoard_svg(self, id_prefix, element_size=1,
                                  element_gap=1,
                                  element_class="coverage_element"):
    es = element_size
    gs = element_gap
    svg_width = self.long_out_channels_xy[0] * (es + gs) * 2
    svg_height = self.long_out_channels_xy[1] * (es + gs) * 2
    code_lines = ['<svg width="%dpx" height="%dpx"' %
                  (svg_width, svg_height) + 
                  ' xmlns="http://www.w3.org/2000/svg" version="1.1">']
    for y in range(self.long_out_channels_xy[1]):
        for x in range(self.long_out_channels_xy[0]):
            i_channel = x + (y * self.long_out_channels_xy[0])
            code_lines.append(
                     '<rect id="%s_%d"' % (id_prefix, i_channel) +
                     ' class="%s"' % element_class +
                     ' width="%s" height="%s"' % (es, es) +
                     ' x="%d" y="%d"' % (2*x*(es+gs), 2*y*(es+gs)) +
                     ' />' )
    offset = self.long_out_channels_xy[0] * self.long_out_channels_xy[1]
    for y in range(self.long_out_channels_xy[1] - 1):
        for x in range(self.long_out_channels_xy[0] - 1):
            i_channel = offset + x + (y * (self.long_out_channels_xy[0]-1))
            code_lines.append(
                     '<rect id="%s_%d"' % (id_prefix, i_channel) +
                     ' class="%s"' % element_class +
                     ' width="%s" height="%s"' % (es, es) +
                     ' x="%d" y="%d"' % ((2*x+1)*(es+gs), (2*y+1)*(es+gs)) +
                     ' />' )
    code_lines.append('</svg>')
    return "\n".join(code_lines)
