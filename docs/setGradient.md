# `Tao.setGradient` Documentation

## Source Mapping

| Layer | Location |
|---|---|
| **Python call** | `tao.setGradient(gradient, g)` |
| **Cython wrapper** | [`src/binding/petsc4py/src/petsc4py/PETSc/TAO.pyx`](https://gitlab.com/petsc/petsc/-/blob/main/src/binding/petsc4py/src/petsc4py/PETSc/TAO.pyx) |
| **C function** | `TaoSetGradient` |
| **C header** | [`include/petsctao.h`](https://gitlab.com/petsc/petsc/-/blob/main/include/petsctao.h) |
| **C implementation** | [`src/tao/interface/taosolver.c`](https://gitlab.com/petsc/petsc/-/blob/main/src/tao/interface/taosolver.c) |

### Cython Bridge (TAO.pyx excerpt)

```cython
def setGradient(self, gradient, Vec g=None, args=None, kargs=None):
    if args  is None: args  = ()
    if kargs is None: kargs = {}
    context = (gradient, args, kargs)
    self.set_attr('__gradient__', context)
    cdef PetscVec gvec = NULL
    if g is not None: gvec = g.vec
    CHKERR(TaoSetGradient(
        self.tao, gvec, TAO_Gradient, <void*>context
    ))
    return self
```

The key difference from `setObjective`: the C signature takes a pre-allocated `Vec g` for the output.  The Python callback must write the gradient **in-place** into this vector via `g.setArray(...)` rather than returning a new array.

### C Function Signature

```c
/* include/petsctao.h */
PetscErrorCode TaoSetGradient(
    Tao                 tao,
    Vec                 g,      /* pre-allocated output gradient vector */
    PetscErrorCode    (*func)(Tao, Vec, Vec, void*),
    void               *ctx
);
```

---

## Documentation

```python
def setGradient(self, gradient, g, args=None, kargs=None):
    """
    Register a callback that evaluates the gradient ∇f(x) of the objective.

    TAO calls ``gradient(tao, x, g, *args, **kargs)`` at each iteration.
    The callback must write the gradient vector **in-place** into ``g``
    using ``g.setArray(...)`` or equivalent; it has no return value.

    A pre-allocated PETSc vector ``g`` must be passed to this method so
    that TAO can manage its lifetime.  A convenient way to create it is
    ``g = x.duplicate()`` where ``x`` is the solution vector.

    Internally this wraps ``TaoSetGradient`` from
    ``src/tao/interface/taosolver.c``.

    Parameters
    ----------
    gradient : callable
        A function with signature::

            gradient(tao, x, g, *args, **kargs) -> None

        where

        - **tao** (*PETSc.TAO*) – the solver context (passed automatically).
        - **x** (*PETSc.Vec*) – current iterate; treat as read-only.
        - **g** (*PETSc.Vec*) – output vector; must be populated with ∇f(x).
        - **\\*args**, **\\*\\*kargs** – forwarded extra arguments.

        The callback must **not** return a value; write the gradient
        in-place with ``g.setArray(grad_array)``.

    g : PETSc.Vec
        A pre-allocated vector with the same parallel layout as the solution
        vector.  TAO uses this as the persistent storage for the gradient.

    args : tuple, optional
        Extra positional arguments forwarded to ``gradient``.
        Defaults to ``()``.

    kargs : dict, optional
        Extra keyword arguments forwarded to ``gradient``.
        Defaults to ``{}``.

    Returns
    -------
    self : PETSc.TAO
        The solver context (for method chaining).

    Notes
    -----
    **Type mapping** — C → Python:

        | C type       | Python type   |
        |--------------|---------------|
        | `Vec` (in)   | `PETSc.Vec`   |
        | `Vec` (out)  | `PETSc.Vec`   |
        | `Tao`        | `PETSc.TAO`   |

    **Common mistake:** returning a NumPy array instead of writing in-place.
    TAO ignores the return value of the gradient callback; the gradient
    *must* be written into the pre-allocated ``g`` vector.

    The gradient vector ``g`` passed here and the one received inside the
    callback are the **same object**; you do not need to call
    ``g.assemble()`` explicitly for dense updates via ``setArray``.

    Examples
    --------
    Rosenbrock gradient for n = 2:

    .. math::

        \\frac{\\partial f}{\\partial x_1} = -400\\,x_1(x_2 - x_1^2) - 2(1-x_1)

        \\frac{\\partial f}{\\partial x_2} = 200\\,(x_2 - x_1^2)

    >>> from petsc4py import PETSc
    >>> import numpy as np
    >>>
    >>> def rosenbrock_grad(tao, x, g):
    ...     xa = x.getArray(readonly=True)
    ...     ga = np.zeros_like(xa)
    ...     ga[:-1] += -400.0 * xa[:-1] * (xa[1:] - xa[:-1]**2) \\
    ...                - 2.0 * (1.0 - xa[:-1])
    ...     ga[1:]  +=  200.0 * (xa[1:] - xa[:-1]**2)
    ...     g.setArray(ga)          # write in-place — no return value
    >>>
    >>> tao = PETSc.TAO().create()
    >>> tao.setType('lmvm')
    >>>
    >>> x = PETSc.Vec().createSeq(2)
    >>> x.setArray([-1.2, 1.0])
    >>> g = x.duplicate()           # same layout, zero-initialised
    >>>
    >>> tao.setGradient(rosenbrock_grad, g)

    See Also
    --------
    Tao.setObjective : register the objective callback.
    Tao.setObjectiveGradient : combined f + ∇f callback (more efficient).
    """
```

---

## Mathematical Background

For an unconstrained problem $\min_{\mathbf{x}} f(\mathbf{x})$, the gradient $\nabla f(\mathbf{x}) \in \mathbb{R}^n$ is the vector of partial derivatives:

$$\left[\nabla f(\mathbf{x})\right]_i = \frac{\partial f}{\partial x_i}$$

L-BFGS (the `lmvm` TAO type) uses the last $m$ gradient differences $\{y_k\}$ and iterate differences $\{s_k\}$ to build the BFGS approximation to $H^{-1} = \left[\nabla^2 f\right]^{-1}$ via the two-loop recursion, without ever explicitly forming $H$.

For the Rosenbrock function the gradient components are:

$$\frac{\partial f}{\partial x_i} =
\begin{cases}
-400\,x_i(x_{i+1} - x_i^2) - 2(1 - x_i) + 200(x_i - x_{i-1}^2) & 0 < i < n-1 \\
-400\,x_0(x_1 - x_0^2) - 2(1 - x_0)                               & i = 0       \\
200\,(x_{n-1} - x_{n-2}^2)                                         & i = n-1
\end{cases}$$
