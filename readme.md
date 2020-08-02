# Gravity inversion script example using [GRAFEN](https://github.com/AlexIII/GRAFEN)

## Basic description

lang: python 3.7

Files:
- `cg.py` - main program file. Implements conjugate gradient method for solving system of linear equations.
- `Aop.py` - a wrapper over GRAFEN and an assortment of grd-related operations.
- `pygrid.py` - a class to work with Surfer grd7 files. Represents grd7 file as NumPy array.

## License

This software is distributed under MIT License. © Alexander Chernoskutov, Denis Byzov