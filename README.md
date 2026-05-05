# MA402 Final Project - Rosenbrock Minimization with PETSc TAO

**Course:** MA402 - Mathematics for Scientific Computing, Spring '26
**Domain:** Data Science & Optimization

---

## Problem Overview

This project applies PETSc's **TAO (Toolkit for Advanced Optimization)** library through `petsc4py` to minimize the classical **Rosenbrock function** in *n* dimensions:

$$f(\mathbf{x}) = \sum_{i=0}^{n-2} \left[ 100\,(x_{i+1} - x_i^2)^2 + (1 - x_i)^2 \right]$$

The function has a unique global minimum at $\mathbf{x}^* = (1, 1, \ldots, 1)$ with $f(\mathbf{x}^*) = 0$.

It is also known as Rosenbrock's Banana function or Rosenbrock's Valley because of its 'U' shape. The global minimum is inside a narrow, parabolic valley. This makes finding the valley easy, but the global minimum difficult.

The solver uses the **L-BFGS quasi-Newton method** (`tao_type = 'lmvm'`), which builds a limited-memory approximation to the inverse Hessian from the last *m* gradient differences.

The C tutorial for the function can be found here: https://petsc.org/main/src/tao/unconstrained/tutorials/rosenbrock1.c.html

---

## Repository Structure

```
MA402-final-project/
├── README.md                     ← this file
├── tutorial_module.py            ← petsc4py solver script
├── tutorial_presentation.ipynb   ← Jupyter Notebook demo
└── docs/
    ├── setObjective.md           ← Documentation for Tao.setObjective
    ├── setGradient.md            ← Documentation for Tao.setGradient
    └── getConvergedReason.md     ← Documentation for Tao.getConvergedReason
```

---

## Documented Functions

| Python Method | C Function | PETSc Source |
|---|---|---|
| `Tao.setObjective()` | `TaoSetObjective` | [`src/tao/interface/taosolver.c`](https://gitlab.com/petsc/petsc/-/blob/main/src/tao/interface/taosolver.c) |
| `Tao.setGradient()` | `TaoSetGradient` | [`src/tao/interface/taosolver.c`](https://gitlab.com/petsc/petsc/-/blob/main/src/tao/interface/taosolver.c) |
| `Tao.getConvergedReason()` | `TaoGetConvergedReason` | [`src/tao/interface/taosolver.c`](https://gitlab.com/petsc/petsc/-/blob/main/src/tao/interface/taosolver.c) |

All three are declared in [`include/petsctao.h`](https://gitlab.com/petsc/petsc/-/blob/main/include/petsctao.h) and wrapped in [`src/binding/petsc4py/src/petsc4py/PETSc/TAO.pyx`](https://gitlab.com/petsc/petsc/-/blob/main/src/binding/petsc4py/src/petsc4py/PETSc/TAO.pyx).

---

## AI Translation Experience

**Model used:** Claude Sonnet 4.6 with Adaptive Thinking

**Process:**  
The starting point was the PETSc C tutorial `src/tao/unconstrained/tutorials/rosenbrock1.c`. I prompted Claude to translate the C TAO file into an equivalent `petsc4py` Python solution. The generated code compiled and the logic was correct, but I encountered several runtime errors that I needed to debug. I also used Claude to explain some of the more complex mathematics behind it all and to help me write the documentation properly in MD files as well as in the Jupyter notebook.

**Bugs encountered and fixed:**

1. **`getConvergedReason()` returns a plain `int`, not a named enum.**  
   The generated code used `reason.name` to print a readable convergence label, e.g.:
```python
   print(f"Converged reason: {reason}  ({reason.name})")
```
   In petsc4py 3.25+, `getConvergedReason()` returns a plain `int` rather than a named enum object, so `.name` does not exist and raised `AttributeError: 'int' object has no attribute 'name'`.  
   *Fix:* Removed the `.name` reference. The integer value alone is sufficient — `reason > 0` confirms convergence, and the value maps to the enum table in the docs (e.g. `3` = `TAO_CONVERGED_GTTOL`).

2. **`getGradientNorm()` returns a PETSc object, not a float.**  
   The generated code printed the gradient norm with a float format specifier:
```python
   gnorm = tao.getGradientNorm()
   print(f"Gradient norm: {gnorm:.6e}")
```
   In this version of petsc4py, `getGradientNorm()` returns a PETSc Mat object rather than a scalar, causing `TypeError: unsupported format string passed to petsc4py.PETSc.Mat.__format__`.  
   *Fix:* Removed the gradient norm print line entirely. The converged reason and final objective value are sufficient to confirm solver success.

3. **`plt.show()` fails in WSL2: no display available.**  
   The generated code called `plt.show()` to render plots interactively. WSL2 has no graphical display server, so this raised `UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown` and no plot appeared.  
   *Fix:* Replaced all `plt.show()` calls with `plt.savefig(...)` to write plots as PNG files, which can then be opened normally from Windows Explorer.

**Installation challenges:** 
The main issue I encountered was environment setup. `petsc4py` does not support Windows Python and requires a Linux environment. Claude directed me through WSL2 (Windows Subsystem for Linux) installation to create an Ubuntu virtual terminal on my device and install the correct dependencies for the code to run.

## Install and Run the solver

```bash
# Install Ubuntu into the Windows operating system
wsl --install

# Once installed, open the Ubuntu terminal and run the following commands
# These update the installation and install python
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv \
                   gcc g++ gfortran cmake libopenmpi-dev

# To install the needed dependencies
sudo apt install -y libopenblas-dev liblapack-dev gfortran python3-dev build-essential
pip3 install petsc4py --break-system-packages
pip3 install matplotlib numpy jupyter --break-system-packages

# Navigate to the file and run the solver (WSL can see Windows files at /mnt/c/)
cd /mnt/c/[File Directory]
python tutorial_module.py

# To run the Jupyter Notebook
pip3 install notebook --break-system-packages
python3 -m notebook --no-browser --ip=0.0.0.0
```

---

