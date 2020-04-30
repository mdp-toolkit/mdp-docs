# -*- coding: utf-8 -*-
from docutils import nodes
from docutils.parsers.rst import Directive

from mdp import __version__, __authors__, __homepage__, __contact__

TAGS = (('Authors', __authors__),
        ('Copyright', 'This document has been placed in the public domain.'),
        ('Homepage', __homepage__),
        ('Contact', __contact__),
        ('Version', __version__))

class VersionStringDirective(Directive):
    has_content = False

    def run(self):
        field_list = nodes.field_list()
        for name, value in TAGS:
            fieldname = nodes.field_name()
            fieldname += nodes.Text(name)
            fieldbody = nodes.field_body()
            para = nodes.paragraph()
            para += nodes.Text(value)
            fieldbody += para
            field = nodes.field('', fieldname, fieldbody)
            field_list += field
        return [field_list]

def setup(app):
    app.add_directive('version-string', VersionStringDirective)
