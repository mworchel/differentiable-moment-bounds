<p align="center">

  <h1 align="center"><a href="https://dl.acm.org/doi/10.1145/3730899">Moment Bounds are Differentiable:<br> Efficiently Approximating Measures in Inverse Rendering</a></h1>

  <div  align="center">
    <a href="https://dl.acm.org/doi/10.1145/3730899">
      <img src="assets/teaser.jpg" alt="Logo" width="100%">
    </a>
  </div>

  <p align="center">
    <i>ACM Transactions on Graphics (SIGGRAPH 2025)</i>
    <br />
    <a href="https://mworchel.github.io/"><strong>Markus Worchel</strong></a>
    ·
    <a href="https://www.cg.tu-berlin.de/people/marc-alexa"><strong>Marc Alexa</strong></a>
  </p>
</p>

## About

This repository contains the official implementation of the paper "Moment Bounds are Differentiable: Efficiently Approximating Measures in Inverse Rendering", which introduces a family of techniques to differentiable rendering that can be used to approximate various effects of light attenuation, such as shadows, transmittance, and transparency.

In particular, this repository contains the core algorithms for computing and differentiating moment-based bounds, which are the key component of these approximations. The algorithms are implement in C++, targeting the CPU and the GPU (using CUDA). The `diffmoments` package exposes these algorithms to Python, where they are readily usable with PyTorch, NumPy, and [Dr.Jit](https://github.com/mitsuba-renderer/drjit).

Note that this repository does *not* contain code for specific applications like shadow mapping or transmittance mapping. Moment shadow maps will soon be available in the [`diffshadow`](https://github.com/mworchel/differentiable-shadow-mapping) package, which exposes low-level primitives for differentiable shadow mapping.

## Installation

The easiest way to install the Python package is via `pip`

```bash
pip install diffmoments
```

### Optional: Test the Installation

To test the installation, run

```bash
pip install numpy pytest
python -m pytest .\tests -v
```

Some tests will be skipped, depending on the availability of packages (NumPy is required, PyTorch/Dr.Jit are optional).

## Usage

Please see the [Getting Started](tutorials/getting_started.ipynb) tutorial to learn about the functions exposed by the `diffmoments` package and how to use them with the different frameworks.

## Development

This section is intended for developers who want to modify the source code or who are trying to use parts of this project in a non-standard way (e.g. not via the Python bindings).

### Cloning

The only external dependency is [`nanobind`](https://github.com/wjakob/nanobind) for the Python bindings, which is included as a git submodule. 

To clone the repository and its dependencies use

```bash
git clone --recursive git@github.com:mworchel/differentiable-moment-bounds.git
```

### Compilation

The project uses C++20 core and library features, so a compatible compiler is required (GCC ≥ 11, Clang ≥ 14, MSVC ≥ 19.27).

If the goal is to modify the source code and access the algorithms through the Python bindings, it is advisable to set up an editable build. Please see the root [`CMakeLists.txt`](CMakeLists.txt) for instructions.

Alternatively, the project can be directly built using CMake (≥ 3.25.2). 

If the goal is to access functions or algorithms from another C++ project, using this project as a dependency, see the notes on architecture below.

#### CUDA

A potential "gotcha" is that executing the CUDA variants not necessarily reflects changes made to the source code. The reason is that compiling the CUDA code is optional, and must be enabled via the CMake option `DM_BUILD_KERNELS`. This requires the CUDA toolkit, specifically a C++20-compatible CUDA compiler (NVCC ≥ 12.0).

### Architecture

The Python interface is just a shallow wrapper around C++ code, which is organized to be accessible by external projects. The following diagram gives an overview of the project structure:

```
    +-------------------------+
    |  PyTorch/NumPy/Dr.Jit   |
    +------------+------------+
                 |
                 v
    +------------+------------+
    |  diffmoments (Python)   | 
    +------------+------------+
                 |
                 v
    +------------+------------+
    |  diffmoments_ext (C++)  |
    +------------+------------+
                 |
                 v
  +--------------+---------------+
  |  diffmoments_dispatch (C++)  |
  +-----------+---------------+--+
              |               ^
              |               |
              |               v
              |    +----------+------------------+
              |    |  diffmoments_kernels (CUDA) |
              |    +----+------------------------+
              |         |                    
              v         v
  +-----------+---------+---+     
  |  diffmoments (C++/CUDA) |            
  +-------------------------+
```

All entities labeled with `C++`/`C++/CUDA` are CMake targets. A brief description, from bottom to top:

* `diffmoments`: contains low-level building blocks (e.g. Cholesky factorization and polynomial root finding), and the primal and adjoint algorithms (`compute_moment_bound`/`compute_moment_bound_backward` in [`moment_problem.hpp`](src/moment_problem.hpp))
* `diffmoments_dispatch`/`diffmoments_kernels`: both targets are tightly coupled (see remarks below) and define high-level functions that, for example, apply the core algorithms to multiple inputs (potentially in parallel) or detect errors in a list of outputs.
* `diffmoments_ext`: exposes the high-level functions to Python (the `diffmoments` Python package is just a shallow wrapper around this extension module).

#### Remarks on CUDA kernels

The target `diffmoments_kernels` has a single source file, [`src/kernels.cu`](src/kernels.cu), which contains the CUDA implementation of the high-level functions. The code is not directly compiled to machine code, but instead translated to PTX (an intermediate language), producing the file [`src/generated/kernels.ptx`](src/generated/kernels.ptx). Since PTX can be JIT-compiled on a user system, the user's hardware doesn't need to be known in advance, ensuring forward-compatibility. The `diffmoments_dispatch` library loads and executes the PTX code using the CUDA driver API.

The PTX file is tracked by git and is only updated when the CMake option `DM_BUILD_KERNELS` is enabled. This ensures that the project compiles on systems missing the CUDA toolkit (e.g., the GitHub runners that build the PyPI wheels).

## License and Copyright

The code in this repository is provided under a 3-clause BSD license. 

## Citation

If you find this code or our method useful for your research, please cite our paper

```bibtex
@article{worchel:2025:diffmoments,
author = {Worchel, Markus and Alexa, Marc},
title = {Moment Bounds are Differentiable: Efficiently Approximating Measures in Inverse Rendering},
year = {2025},
issue_date = {August 2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {44},
number = {4},
issn = {0730-0301},
url = {https://doi.org/10.1145/3730899},
doi = {10.1145/3730899},
journal = {ACM Trans. Graph.},
month = jul,
articleno = {80},
numpages = {21},
keywords = {differentiable rendering, shadows}
}
```
