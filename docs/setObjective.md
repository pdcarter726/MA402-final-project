# `Tao.setObjective` Documentation

## Source Mapping

| Layer | Location |
|---|---|
| **Python call** | `tao.setObjective(objective)` |
| **Cython wrapper** | [`src/binding/petsc4py/src/petsc4py/PETSc/TAO.pyx`](https://gitlab.com/petsc/petsc/-/blob/main/src/binding/petsc4py/src/petsc4py/PETSc/TAO.pyx) |
| **C function** | `TaoSetObjective` |
| **C header** | [`include/petsctao.h`](https://gitlab.com/petsc/petsc/-/blob/main/include/petsctao.h) |
| **C implementation** | [`src/tao/interface/taosolver.c`](https://gitlab.com/petsc/petsc/-/blob/main/src/tao/interface/taosolver.c) |

### Cython Bridge (TAO.pyx excerpt)

```cython
def setObjective(self, objective, args=None, kargs=None):
    if args  is None: args  = ()
    if kargs is None: kargs = {}
    context = (objective, args, kargs)
    self.set_attr('__objective__', context)
    CHKERR(TaoSetObjective(
        self.tao, TAO_Objective, <void*>context
    ))
    return self
```

The wrapper stores the Python callable in a tuple with its positional and keyword arguments, then passes a C function pointer (`TAO_Objective`) and the tuple as a void context pointer to `TaoSetObjective`. The C function pointer dereferences the context at each TAO iteration to invoke the Python callable.

### C Function Signature

```c
/* include/petsctao.h */
PetscErrorCode TaoSetObjective(
    Tao                 tao,
    PetscErrorCode    (*func)(Tao, Vec, PetscReal*, void*),
    void               *ctx
);
```

---

## NumPy-Style Docstring

```python
def setObjective(self, objective, args=None, kargs=None):
    """
    Register a callback that evaluates the scalar objective function f(x).

    TAO calls ``objective(tao, x, *args, **kargs)`` at each iteration to
    obtain the current function value.  The return value must be a Python
    ``float`` (or anything castable to ``PetscReal``).

    Internally this wraps ``TaoSetObjective`` from
    ``src/tao/interface/taosolver.c``.  The Python callable is stored as a
    Cython context pointer so that the C-level callback can invoke it without
    the GIL being held by the caller.

    Parameters
    ----------
    objective : callable
        A function with signature::

            f_val = objective(tao, x, *args, **kargs)

        where

        - **tao** (*PETSc.TAO*) – the solver context (passed automatically).
        - **x** (*PETSc.Vec*) – the current iterate; treat as read-only.
        - **\\*args**, **\\*\\*kargs** – forwarded from the ``args`` / ``kargs``
          arguments below.
        - **f_val** (*float*) – the scalar objective value f(x).

    args : tuple, optional
        Positional arguments forwarded to ``objective`` after ``tao`` and
        ``x``.  Defaults to ``()``.

    kargs : dict, optional
        Keyword arguments forwarded to ``objective``.  Defaults to ``{}``.

    Returns
    -------
    self : PETSc.TAO
        The solver context (for method chaining).

    Notes
    -----
    **Type mapping** — C → Python:

    | C type        | Python type   |
    |---------------|---------------|
    | `PetscReal`   | `float`       |
    | `Vec`         | `PETSc.Vec`   |
    | `Tao`         | `PETSc.TAO`   |

    The callback must *return* the objective value; it must **not** write
    it to an output argument (contrast with ``setGradient``, which writes
    in-place).

    If ``setObjectiveGradient`` is also set, TAO may prefer the combined
    callback to avoid redundant evaluations.

    Examples
    --------
    Minimize the 2-D Rosenbrock function:

    .. math::

        f(x_1, x_2) = 100\\,(x_2 - x_1^2)^2 + (1 - x_1)^2

    >>> from petsc4py import PETSc
    >>> import numpy as np
    >>>
    >>> def rosenbrock_obj(tao, x):
    ...     xa = x.getArray(readonly=True)
    ...     return float(100.0 * (xa[1] - xa[0]**2)**2 + (1.0 - xa[0])**2)
    >>>
    >>> tao = PETSc.TAO().create()
    >>> tao.setType('lmvm')
    >>> tao.setObjective(rosenbrock_obj)

    With extra parameters passed through ``args``:

    >>> def weighted_obj(tao, x, weight):
    ...     xa = x.getArray(readonly=True)
    ...     f = float(100.0 * (xa[1] - xa[0]**2)**2 + (1.0 - xa[0])**2)
    ...     return weight * f
    >>>
    >>> tao.setObjective(weighted_obj, args=(2.5,))

    See Also
    --------
    Tao.setGradient : register the gradient callback.
    Tao.setObjectiveGradient : register a combined f+∇f callback.
    """
```

---

## Mathematical Background

TAO solves:

$$\min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x})$$

where $f : \mathbb{R}^n \to \mathbb{R}$ is a smooth scalar-valued function.
`setObjective` registers the Python function that TAO calls to evaluate $f(\mathbf{x})$ at any given iterate $\mathbf{x}$.

For the Rosenbrock problem:

$$f(\mathbf{x}) = \sum_{i=0}^{n-2} \left[ 100\,(x_{i+1} - x_i^2)^2 + (1 - x_i)^2 \right]$$

The minimum $f(\mathbf{x}^*) = 0$ is attained at $\mathbf{x}^* = \mathbf{1}$.
