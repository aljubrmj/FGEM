# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'FGEM'
copyright = '2024, Mohammad Aljubran'
author = 'Mohammad Aljubran'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    #'sphinx.ext.autodoc',
    #'sphinx.ext.autosummary',
    "myst_parser",
    # "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.graphviz",
    # "sphinxcontrib.bibtex",
    #'sphinx.ext.pngmath',
    #'sphinxcontrib.tikz',
    #'rinoh.frontend.sphinx',
    "sphinx.ext.imgconverter",  # for SVG conversion
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

html_theme_options = {
    "repository_url": "https://github.com/aljubrmj/FGEM",
    "use_repository_button": True,
    "show_navbar_depth": 1,
}

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
# html_title = "FGEM"

# A shorter title for the navigation bar.  Default is the same as html_title.
# html_short_title = "FGEM"

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "./_static/fgem-logo.png"

# -- Options for EPUB output
# epub_show_urls = 'footnote'
