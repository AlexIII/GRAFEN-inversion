# Gravity inversion script example using [GRAFEN](https://github.com/AlexIII/GRAFEN/tree/topo+layered)

## Usage

Workload is defined as a directory with config, measured gravity field data and a height map.

```
./grafen-inversion.py workload_dir [init | solve | continue]
    workload_dir/
        target_field*.grd
        topo_hieghtmap*.grd
        config.txt
            n_layers = 81
            [l0 = 60]
            [pprr = 100] # Point Potential Replace Radius, -1 for no PPRR
```

## Basic description

lang: python 3.7

Files:
- `grafen-inversion.py` - main program file.
- `cg.py` - Conjugate gradient method for solving system of linear equations.
- `Aop.py` - a wrapper over GRAFEN and an assortment of grd-related operations.
- `pygrid.py` - a class to work with Surfer grd7 files. Represents grd7 file as NumPy array.

## License

This software is distributed under MIT License. © Alexander Chernoskutov, Denis Byzov