# `Tao.getConvergedReason` — Documentation

## Source Mapping

| Layer | Location |
|---|---|
| **Python call** | `tao.getConvergedReason()` |
| **Cython wrapper** | [`src/binding/petsc4py/src/petsc4py/PETSc/TAO.pyx`](https://gitlab.com/petsc/petsc/-/blob/main/src/binding/petsc4py/src/petsc4py/PETSc/TAO.pyx) |
| **C function** | `TaoGetConvergedReason` |
| **C header** | [`include/petsctao.h`](https://gitlab.com/petsc/petsc/-/blob/main/include/petsctao.h) |
| **C implementation** | [`src/tao/interface/taosolver.c`](https://gitlab.com/petsc/petsc/-/blob/main/src/tao/interface/taosolver.c) |

### Cython Bridge (TAO.pyx excerpt)

```cython
def getConvergedReason(self):
    cdef PetscTAOConvergedReason reason = TAO_CONTINUE_ITERATING
    CHKERR(TaoGetConvergedReason(self.tao, &reason))
    return reason
```

The C enum `PetscTAOConvergedReason` (typedef'd to `TaoConvergedReason`) is automatically mapped by Cython to a Python `IntEnum` subclass (`PETSc.TAO.ConvergedReason`).  Positive values indicate success; negative values indicate failure; zero means the solver is still running.

### C Function Signature

```c
/* include/petsctao.h */
PetscErrorCode TaoGetConvergedReason(
    Tao                    tao,
    TaoConvergedReason    *reason   /* output enum value */
);
```

### Convergence Reason Enum Values

| Name | Value | Meaning |
|---|---|---|
| `TAO_CONVERGED_GATOL` | 1 | `‖∇f‖ ≤ gatol` |
| `TAO_CONVERGED_GRTOL` | 2 | `‖∇f‖ ≤ grtol · ‖∇f₀‖` |
| `TAO_CONVERGED_GTTOL` | 3 | `‖∇f‖ ≤ gttol · ‖∇f₀‖` (relative-to-initial) |
| `TAO_CONVERGED_STEPTOL` | 4 | Step size below tolerance |
| `TAO_CONVERGED_MINF` | 5 | Objective below minimum bound |
| `TAO_CONVERGED_USER` | 6 | User-defined convergence |
| `TAO_CONTINUE_ITERATING` | 0 | Still iterating |
| `TAO_DIVERGED_MAXITS` | -2 | Maximum iterations exceeded |
| `TAO_DIVERGED_NAN` | -4 | Objective or gradient is NaN |
| `TAO_DIVERGED_MAXFCN` | -5 | Maximum function evaluations exceeded |
| `TAO_DIVERGED_LS_FAILURE` | -6 | Line search failed |
| `TAO_DIVERGED_TR_REDUCTION` | -7 | Trust-region reduction failed |
| `TAO_DIVERGED_USER` | -8 | User-defined divergence |

---

## Documentation

```python
def getConvergedReason(self):
    """
    Return the reason the TAO solver stopped iterating.

    Must be called **after** ``tao.solve()``.  The returned value tells you
    whether the optimiser converged to a solution satisfying the tolerances,
    ran out of iterations, or encountered a numerical failure.

    Internally this wraps ``TaoGetConvergedReason`` from
    ``src/tao/interface/taosolver.c``, which reads the ``reason`` field set
    by the convergence-test routine at each TAO iteration.

    Parameters
    ----------
    None

    Returns
    -------
    reason : PETSc.TAO.ConvergedReason
        An integer-valued enum.  Positive values indicate success;
        negative values indicate failure; zero (``TAO_CONTINUE_ITERATING``)
        means the solver has not yet run or is still iterating.

        Key values:

        | Name                        | Value |
|-----------------------------|-------|
| `TAO_CONVERGED_GATOL`       | 1     |
| `TAO_CONVERGED_GRTOL`       | 2     |
| `TAO_CONVERGED_STEPTOL`     | 4     |
| `TAO_CONTINUE_ITERATING`    | 0     |
| `TAO_DIVERGED_MAXITS`       | -2    |
| `TAO_DIVERGED_NAN`          | -4    |
| `TAO_DIVERGED_LS_FAILURE`   | -6    |

Notes
-----
**Type mapping** C → Python:

| C type                       | Python type                   |
|------------------------------|-------------------------------|
| `TaoConvergedReason` (int)   | `PETSc.TAO.ConvergedReason`   |

    A result of ``reason > 0`` is the idiomatic check for convergence.
    ``reason.name`` gives the human-readable enum string.

    The convergence tolerances tested against are those set by
    ``tao.setTolerances``; the specific reason indicates which tolerance
    was first satisfied.

    Examples
    --------
    Basic convergence check after solve:

    >>> from petsc4py import PETSc
    >>>
    >>> tao = PETSc.TAO().create()
    >>> tao.setType('lmvm')
    >>> # ... register callbacks, set tolerances, set solution ...
    >>> tao.solve()
    >>>
    >>> reason = tao.getConvergedReason()
    >>> print(reason)           # integer value, e.g. 2
    >>> print(f"  Converged reason  : {reason}")      # human-readable, e.g. 'TAO_CONVERGED_GRTOL'
    >>>
    >>> if reason > 0:
    ...     print("Optimisation succeeded.")
    ... else:
    ...     print(f"Optimisation failed: {reason.name}")

    Diagnosing divergence:

    >>> if reason == PETSc.TAO.ConvergedReason.DIVERGED_MAXITS:
    ...     print("Hit iteration limit — try increasing max_it or loosening tolerances.")
    >>> elif reason == PETSc.TAO.ConvergedReason.DIVERGED_NAN:
    ...     print("Objective or gradient is NaN — check the callback for numerical issues.")

    See Also
    --------
    Tao.setTolerances : set the convergence tolerances tested against.
    Tao.getIterationNumber : number of outer iterations performed.
    Tao.getObjectiveValue : final objective value f(x*).
    Tao.getGradientNorm : final gradient norm ‖∇f(x*)‖.
    """
```

---

## Mathematical Background

TAO declares convergence when one of the following criteria is satisfied:

| Criterion | Condition | Tolerance parameter |
|---|---|---|
| Absolute gradient norm | $\|\nabla f(\mathbf{x}_k)\| \leq \varepsilon_{\text{abs}}$ | `gatol` |
| Relative gradient norm | $\|\nabla f(\mathbf{x}_k)\| \leq \varepsilon_{\text{rel}} \cdot \|\nabla f(\mathbf{x}_0)\|$ | `grtol` |
| Gradient ratio | $\|\nabla f(\mathbf{x}_k)\| \leq \varepsilon_{\text{ratio}} \cdot \|\nabla f(\mathbf{x}_0)\|$ | `gttol` |
| Step size | $\|\mathbf{x}_{k+1} - \mathbf{x}_k\| \leq \delta_{\text{step}}$ | `steptol` |

For the Rosenbrock problem the expected convergence reason is `TAO_CONVERGED_GATOL` or `TAO_CONVERGED_GRTOL`, since both the gradient norm and objective drop to machine precision near $\mathbf{x}^* = \mathbf{1}$.
