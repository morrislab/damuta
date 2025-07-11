# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'DAMUTA'
copyright = '2022, Cait Harrigan'
author = 'Cait Harrigan'

# The full version, including alpha/beta/rc tags
release = "1.0.7"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    'sphinx.ext.autosummary',
    "sphinx.ext.napoleon", 
    "sphinx_rtd_theme",
    "nbsphinx",
    'sphinx_gallery.load_style',
    "nbsphinx_link",
]

# Mock imports for packages that might not be available on RTD or are heavy dependencies
autodoc_mock_imports = [
    'wandb',
    'pymc3',  # Also mock pymc3 as it's heavy
    'theano',
]

autodoc_default_options = {"autosummary": True}
autodoc_default_flags = ['members']
add_module_names = False

nbsphinx_thumbnails = {
    'examples/quickstart': '_static/quickstart.png',
    'examples/data': '_static/data.png', 
    'examples/models': '_static/models.png',
    'examples/wandb': '_static/wandb.png',
    'examples/estimate_signatures_and_activities': '_static/sigs.png', 
}



# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '_templates', 'stubs', 'damuta.rst']

# Skip notebooks if pandoc is not available (for local builds)
import shutil
if not shutil.which('pandoc'):
    print("Warning: pandoc not found, skipping notebook examples")
    exclude_patterns.extend(['examples/*.ipynb'])
else:
    nbsphinx_allow_errors = True


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['thumbnails']