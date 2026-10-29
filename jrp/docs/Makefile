# Minimal Sphinx makefile.  -W is not optional here: the whole point of the
# docs build in CI is that warnings fail it, and a local build that is laxer
# than CI is worse than no local build.
SPHINXOPTS  ?= -W
SPHINXBUILD ?= python -m sphinx
SOURCEDIR    = .
BUILDDIR     = _build

.PHONY: help html clean
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
clean:
	rm -rf "$(BUILDDIR)"
