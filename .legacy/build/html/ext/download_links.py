# -*- coding: utf-8 -*-
from docutils import nodes
from docutils.parsers.rst import Directive

from mdp import __version__

class DownloadLinkDirective(Directive):
    has_content = False
    required_arguments = 1
    final_argument_whitespace = True

    def run(self):
        prefix = self.arguments[0]
        dl = self.state_machine.document.settings.env.config.download_link
        if len(dl) == 0:
            raise Exception('You must set the variable "download_link".')
        text = ' '.join((prefix, dl))
        return [nodes.literal_block(text, text)]

def setup(app):
    app.add_directive('download-link', DownloadLinkDirective)
    app.add_config_value('download_link', '', 'env')
