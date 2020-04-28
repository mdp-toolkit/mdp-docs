# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = PYTHONPATH=$(PYTHONPATH) sphinx-build
PAPER         = a4
BUILDDIR      = build

# mdp specific modifications
MDPTOOLKIT    = ../mdp-toolkit
PYTHONPATH   := $(PYTHONPATH):ext:$(MDPTOOLKIT):source/examples/binetdbn   # use hard assignment to avoid recursion
EPYDOC        = PYTHONPATH=$(PYTHONPATH) epydoc
APIBUILD      = build_api/api
APICSS	      = source/_static/API.css
CODEDIR       = source/code

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) -n source
LINKS           = absolute

.PHONY: help clean html htmllocal dirhtml singlehtml pickle json htmlhelp qthelp devhelp epub latex latexpdf text man changes linkcheck doctest

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "  epydoc     to make API with epydoc"
	@echo "  dirhtml    to make HTML files named index.html in directories"
	@echo "  singlehtml to make a single large HTML file"
	@echo "  pickle     to make pickle files"
	@echo "  json       to make JSON files"
	@echo "  htmlhelp   to make HTML files and a HTML help project"
	@echo "  qthelp     to make HTML files and a qthelp project"
	@echo "  devhelp    to make HTML files and a Devhelp project"
	@echo "  epub       to make an epub"
	@echo "  latex      to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  latexpdf   to make LaTeX files and run them through pdflatex"
	@echo "  text       to make text files"
	@echo "  man        to make manual pages"
	@echo "  changes    to make an overview of all changed/added/deprecated items"
	@echo "  linkcheck  to check all external links for integrity"
	@echo "  doctest    to run all doctests embedded in the documentation (if enabled)"
	@echo "  codesnippet  to create python modules from doctests embedded in the documentation (if enabled)"
	@echo "  website    make MDP website"
	@echo "        add MDPTOOLKIT=../... to specify path to mdp-toolkit (default ../mdp-toolkit)"
	@echo "        add LINKS=local to generate local relative links in html"
	@echo "        add PAPER=letter to set page size (default a4)"

clean:
	-rm -rf $(BUILDDIR)/* $(APIBUILD)

html: html$(LINKS)

htmlabsolute:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

htmllocal: htmlabsolute
	for file in `grep -l -I mdp-toolkit.sourceforge.net -r $(BUILDDIR)/html`; do\
		repl=`echo $$file|sed -r "s|build/html/||; s|[^/]+/|../|g; s|[^/]+$$||"`; \
		sed -r -i "s|http://mdp-toolkit.sourceforge.net/|$$repl|g" $$file; \
        done

epydoc:
	mkdir -p $(APIBUILD)
	mkdir -p $(BUILDDIR)/html
	$(EPYDOC) --debug \
	--html -o $(APIBUILD) --name="Modular toolkit for Data Processing MDP" \
	--url="http://mdp-toolkit.sourceforge.net"  \
	--css=$(APICSS) \
	--show-frames \
	--introspect-only \
        --no-sourcecode \
        --no-imports \
        --redundant-details \
	--inheritance=grouped \
	--verbose \
        --graph all --graph-font-size 9 \
	--docformat=plaintext \
	--external-api=numpy \
	$(MDPTOOLKIT)/mdp/__init__.py
	cp -r $(APIBUILD)  $(BUILDDIR)/html/

dirhtml:
	$(SPHINXBUILD) -b dirhtml $(ALLSPHINXOPTS) $(BUILDDIR)/dirhtml
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/dirhtml."

singlehtml:
	$(SPHINXBUILD) -b singlehtml $(ALLSPHINXOPTS) $(BUILDDIR)/singlehtml
	@echo
	@echo "Build finished. The HTML page is in $(BUILDDIR)/singlehtml."

pickle:
	$(SPHINXBUILD) -b pickle $(ALLSPHINXOPTS) $(BUILDDIR)/pickle
	@echo
	@echo "Build finished; now you can process the pickle files."

json:
	$(SPHINXBUILD) -b json $(ALLSPHINXOPTS) $(BUILDDIR)/json
	@echo
	@echo "Build finished; now you can process the JSON files."

htmlhelp:
	$(SPHINXBUILD) -b htmlhelp $(ALLSPHINXOPTS) $(BUILDDIR)/htmlhelp
	@echo
	@echo "Build finished; now you can run HTML Help Workshop with the" \
	      ".hhp project file in $(BUILDDIR)/htmlhelp."

qthelp:
	$(SPHINXBUILD) -b qthelp $(ALLSPHINXOPTS) $(BUILDDIR)/qthelp
	@echo
	@echo "Build finished; now you can run "qcollectiongenerator" with the" \
	      ".qhcp project file in $(BUILDDIR)/qthelp, like this:"
	@echo "# qcollectiongenerator $(BUILDDIR)/qthelp/MDP-toolkit.qhcp"
	@echo "To view the help file:"
	@echo "# assistant -collectionFile $(BUILDDIR)/qthelp/MDP-toolkit.qhc"

devhelp:
	$(SPHINXBUILD) -b devhelp $(ALLSPHINXOPTS) $(BUILDDIR)/devhelp
	@echo
	@echo "Build finished."
	@echo "To view the help file:"
	@echo "# mkdir -p $$HOME/.local/share/devhelp/MDP-toolkit"
	@echo "# ln -s $(BUILDDIR)/devhelp $$HOME/.local/share/devhelp/MDP-toolkit"
	@echo "# devhelp"

epub:
	$(SPHINXBUILD) -b epub $(ALLSPHINXOPTS) $(BUILDDIR)/epub
	@echo
	@echo "Build finished. The epub file is in $(BUILDDIR)/epub."

latex:
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo
	@echo "Build finished; the LaTeX files are in $(BUILDDIR)/latex."
	@echo "Run \`make' in that directory to run these through (pdf)latex" \
	      "(use \`make latexpdf' here to do that automatically)."

latexpdf:
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo "Running LaTeX files through pdflatex..."
	make -C $(BUILDDIR)/latex all-pdf > /dev/null 2>&1
	@echo "pdflatex finished; the PDF files are in $(BUILDDIR)/latex."

text:
	$(SPHINXBUILD) -b text $(ALLSPHINXOPTS) $(BUILDDIR)/text
	@echo
	@echo "Build finished. The text files are in $(BUILDDIR)/text."

man:
	$(SPHINXBUILD) -b man $(ALLSPHINXOPTS) $(BUILDDIR)/man
	@echo
	@echo "Build finished. The manual pages are in $(BUILDDIR)/man."

changes:
	$(SPHINXBUILD) -b changes $(ALLSPHINXOPTS) $(BUILDDIR)/changes
	@echo
	@echo "The overview file is in $(BUILDDIR)/changes."

linkcheck:
	$(SPHINXBUILD) -b linkcheck2 $(ALLSPHINXOPTS) $(BUILDDIR)/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in $(BUILDDIR)/linkcheck/output.txt."

doctest:
	$(SPHINXBUILD) -b doctest $(ALLSPHINXOPTS) $(BUILDDIR)/doctest
	@echo "Testing of doctests in the sources finished, look at the " \
	      "results in $(BUILDDIR)/doctest/output.txt."

codesnippet:
	$(SPHINXBUILD) -b codesnippet $(ALLSPHINXOPTS) $(BUILDDIR)/codesnippet
	rm -rf $(CODEDIR)
	mkdir -p $(CODEDIR)
	cp -a $(BUILDDIR)/codesnippet/* $(CODEDIR)
	@echo "Generation of module from doctest finished, look at the" \
	      "results in $(BUILDDIR)/codesnippet/."

website: epydoc codesnippet html$(LINKS) latexpdf

legacyapi:
	mkdir -p $(BUILDDIR)/html
	cp -a api $(BUILDDIR)/html/

legacywebsite: legacyapi codesnippet html
legacywebsitelocal: legacyapi codesnippet htmllocal
