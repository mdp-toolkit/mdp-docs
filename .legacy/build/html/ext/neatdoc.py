from sphinx.ext.autosummary import Autosummary
from sphinx.ext.autosummary import get_documenter
from sphinx import addnodes
from docutils.parsers.rst import directives, Directive
from docutils.nodes import Text
from sphinx.util.inspect import safe_getattr,safe_getmembers
import re
import sphinx.ext.autodoc
import sys
import inspect

class Docsummary(Autosummary):
    has_content = True
    required_arguments = 1
    final_argument_whitespace = False
    option_spec = {
        'methods': directives.unchanged,
        'attributes': directives.unchanged,
        'modulecontent': directives.unchanged,
        'inherited': directives.unchanged
    }


    @staticmethod
    def get_members(obj, typ):
        items = []
        for name in dir(obj):
            try:
                member = safe_getattr(obj, name)
                if type(member) is str or type(member) is list:
                    documenter = None
                else:
                    documenter = get_documenter(member, obj)
            except AttributeError:
                continue
            if documenter is not None:
                if documenter.objtype == typ:
                    items.append((name,member))

        return items
    @staticmethod
    def get_mod_member(obj):
        members = safe_getmembers(obj)
        ret =[]
        for mname in members:
            try:
                ret.append((mname, safe_getattr(obj, mname[0])))
            except:
                pass
        return ret

  
    def run(self):
        app = self.state.document.settings.env.app
        self.content = []

        if ('methods' in self.options) or ('attributes' in self.options):
            clazz = self.arguments[0]
            (module_name,class_name) = re.findall(re.compile("^(.*)\.([^.]*)$"),clazz)[0]
            class_name = class_name#.encode("utf-8")
            module_name = module_name#.encode("utf-8")
            try:
                m = __import__(module_name, globals(), locals(), [class_name])
                c = getattr(m, class_name)
            except ImportError:
                print("Warning: Could not import %s's class %s" % (module_name,class_name))
                return []

            if 'methods' in self.options:
                self.content_inherited = []
                methods = self.get_members(c, 'method')

                for method in methods:
                    if method[0] not in c.__dict__:
                        self.content_inherited.append("~%s.%s" % (clazz, method[0]))
                    else:
                        self.content.append("~%s.%s" % (clazz, method[0]))

                if 'inherited' in self.options:
                    self.content = self.content_inherited
                    return super(Docsummary, self).run()
                else:
                    return super(Docsummary, self).run()



            elif 'attributes' in self.options:

                attribs = self.get_members(c, 'attribute')
                self.content = ["~%s.%s" % (clazz, attrib[0]) for attrib in attribs ]


        elif ('modulecontent' in self.options):
            self.content = []
            module = self.arguments[0]
            m = __import__(module, globals,locals(),[],0)
            impmodulenames = set(sys.modules)&set(dir(m))
            dmodulenames = [mod for mod in dir(m) if mod not in impmodulenames]
            for name in dmodulenames:
                obj = getattr(m,name)                
                if hasattr(obj,'__module__'):
#                    if (module == obj.__module__):
                    self.content.append("~%s.%s" % (module, name))

        return super(Docsummary, self).run()

def run_apidoc(app):
    from sphinx.ext.apidoc import main
    import os
    import sys

    module_path_list = app.config.neatdoc_module_path_list
    module_path_rst_target = app.config.neatdoc_module_path_rst_target
    apidoc_options = app.config.neatdoc_apidoc_options
       
    for module_path in module_path_list:
        foldername = (re.search(r'.+/([^/]+)',module_path)).group(1)
        main(apidoc_options+[ '-o', module_path_rst_target+'/'+foldername, module_path, '--force'])



def createClassToc(app, what, name, obj, options, lines):
    # adds autoautosummary directive to end of docstring to create a class-toc

    create_class_toc = app.config.neatdoc_create_class_toc
    create_module_toc = app.config.neatdoc_create_module_toc
    if (what =="class")and create_class_toc:
        classtoc0 = ['**Standart attributes:**','','.. docsummary:: %s'% (name),'    :attributes:','']
        classtoc1 = ['**Methods:**','','*Non-inherited*','','.. docsummary:: %s'% (name),'    :methods:','',
                                       '*Inherited*','','.. docsummary:: %s'% (name),'    :inherited:','    :methods:', '']
        classtoc0.extend(classtoc1)
        lines.extend( classtoc0)

    if (what =="module")and create_module_toc:
        moduletoc0 = ['**Module content:**','','.. docsummary:: %s'% (name),'    :modulecontent:']
        lines.extend(moduletoc0)

def setup(app):
    app.connect('autodoc-process-docstring', createClassToc)
    app.connect('builder-inited', run_apidoc)

    app.add_directive('docsummary', Docsummary)

    app.add_config_value('neatdoc_module_path_list', None, 'env')
    app.add_config_value('neatdoc_module_path_rst_target', app.srcdir, 'env')
    app.add_config_value('neatdoc_apidoc_options', ['-e', '-P', '-d 5'], 'env')
    app.add_config_value('neatdoc_create_class_toc', True, 'env')
    app.add_config_value('neatdoc_create_module_toc', True, 'env')
