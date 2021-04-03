# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "Playing Awale with MCTS"
copyright = "2020, Nikita Marchant"
author = "Nikita Marchant"

# The short X.Y version
version = ""
# The full version, including alpha/beta/rc tags
release = ""


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

import sys, os

sys.path.append(os.path.abspath("./extensions/"))

extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.ifconfig",
    "sphinxcontrib.bibtex",
    "nbsphinx",
    "sphinxcontrib.proof",
    # "numsec",
]

mathjax_config = {
    "tex2jax": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
        "processEscapes": True,
        "processEnvironments": True,
    }
}

todo_include_todos = True
nbsphinx_allow_errors = True
nbsphinx_execute = "never"

numfig = True
numfig_format = {
    "figure": "Figure %s",
    "table": "Table %s",
    "code-block": "Listing %s",
    "section": "Section %s",
    "proof": "Theorem %s",
}
numfig_secnum_depth = 0

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects htmlstatic_path and html_extra_path .
exclude_patterns = ["**.ipynb_checkpoints", "**.ipynb"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"
highlight_language = "none"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "alabaster"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {"show_powered_by": False, "sidebar_width": 0}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static", "_static/custom.css"]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
html_sidebars = {}

html_js_files = [
    ("js/custom.js", {"type": "module"}),
    (
        "https://cdn.jsdelivr.net/npm/pseudocode@latest/build/pseudocode.min.js",
        {"id": "pseudocode-script"},
    ),
    "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS_CHTML",
]

html_css_files = [
    "https://cdn.jsdelivr.net/npm/pseudocode@latest/build/pseudocode.min.css"
]

html_favicon = "_static/favicon.ico"

mathjax_path = " "

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "mancala-thesisdoc"
html_title = "Playing Awale with MCTS"


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    "papersize": "a4paper",
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "mancala-thesis.tex",
        "mancala-thesis Documentation",
        "Nikita Marchant",
        "manual",
    ),
]

nitpicky = True

# -- Extension configuration -------------------------------------------------


proof_theorem_types = {
    "algorithm": "Algorithm",
    "conjecture": "Conjecture",
    "corollary": "Corollary",
    "definition": "Definition",
    "example": "Example",
    "lemma": "Lemma",
    "observation": "Observation",
    "proof": "Proof",
    "property": "Property",
    "theorem": "Theorem",
    "application": "Application",
}

proof_html_nonumbers = ["application"]

from pybtex.plugin import register_plugin  # noqa
from pybtex.style.formatting.unsrt import Style as UnsrtStyle  # noqa
from pybtex.style.labels.alpha import LabelStyle as AlphaLabelStyle  # noqa


class CustomLabelStyle(AlphaLabelStyle):
    def format_label(self, entry):
        label = entry.persons["author"][0].last()[-1]
        if "year" in entry.fields:
            label += ", " + entry.fields["year"]
        return label


class CustomStyle(UnsrtStyle):
    default_label_style = "customlabel"


register_plugin("pybtex.style.labels", "customlabel", CustomLabelStyle)
register_plugin("pybtex.style.formatting", "custom", CustomStyle)
