[![](docs/_static/images/python310.svg)](https://www.python.org/downloads/) [![](docs/_static/images/arXiv.svg)](https://arxiv.org/abs/2511.00165) [![Licence: MIT](docs/_static/images/license.svg)](LICENSE)
# opbasis
This Python package allows the derivation of a minimal operator basis for typical use cases of lattice gauge theory practitioners.
- **Full documentation** and some introductory examples see the [GitHub Pages](https://nikolai-husung.github.io/opbasis/).
- **Installation** see the instructions below.

## Installation
There are three ways to make this package available on your client:
- Use pip to install the latest version supplied in the git via
```bash
pip3 install opbasis-1.0.0b1-py3-none-any.whl
```
- Run the Makefile to build and install the latest version yourself.
- Add the *opbasis* folder containing all the Python source files to your work directory or to `sys.path`.

The package requires Python in version 3.10 or higher.

## Possible plans for the future
- [ ] Add support for multiple gauge groups including Abelian ones and allow for non-traceless algebra generators. This is of particular interest for QCD+QED but it will require a major overhaul of the way `_AlgebraBlock` and EOMs are being dealt with right now.
