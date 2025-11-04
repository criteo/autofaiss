# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

install:
	pip install sphinx sphinxcontrib-napoleon sphinx-autodoc-typehints sphinx_rtd_theme nbsphinx

doc: pre copy-notebook html
	echo "building doc done"

venv-doc: ## [Continuous integration] Install in venv and build doc
	python3 -m venv .env && . .env/bin/activate && cd .. && make install && cd docs && make install pre doctest html && rm -rf .env
	
run:
	cd _build/html && python -m http.server 8010

pre:
	sphinx-apidoc --ext-autodoc --extensions sphinxcontrib.napoleon -f -o _source/ ../autofaiss

clean:
	rm -r _build
	rm -r _source

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
