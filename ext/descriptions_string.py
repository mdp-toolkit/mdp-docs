# -*- coding: utf-8 -*-
from docutils import nodes, core
from docutils.parsers.rst import Directive

from mdp import __short_description__ as short_description
from mdp import __doc__ as long_description
from mdp import __medium_description__ as middle_description

class LongDescriptionStringDirective(Directive):
    has_content = False
    def run(self):
        document = core.publish_doctree(long_description)
        return document.children

class MiddleDescriptionStringDirective(Directive):
    has_content = False
    def run(self):
        document = core.publish_doctree(middle_description)
        return document.children

class ShortDescriptionStringDirective(Directive):
    has_content = False
    def run(self):
        document = core.publish_doctree(short_description)
        return document.children

def setup(app):
    app.add_directive('long-description-string',
                      LongDescriptionStringDirective)
    app.add_directive('short-description-string',
                      ShortDescriptionStringDirective)
    app.add_directive('middle-description-string',
                      MiddleDescriptionStringDirective)
