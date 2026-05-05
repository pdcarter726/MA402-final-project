"""
Rosenbrock Function Minimization using PETSc TAO

This module solves the n-dimensional Rosenbrock (banana) optimization problem:

    min f(x) = sum_{i=0}^{n-2} [ 100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2 ]

The global minimum is located at x* = (1, 1, ..., 1) with f(x*) = 0.

The solver uses PETSc's TAO (Toolkit for Advanced Optimization) with the
L-BFGS quasi-Newton method ('lmvm'), which builds a limited memory
approximation to the inverse Hessian using gradient history.

Reference (C tutorial):
    https://petsc.org/main/src/tao/unconstrained/tutorials/rosenbrock1.c.html
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

#  PETSc initialisation 
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc


class RosenbrockSolver:
    """
    Solves the n-dimensional Rosenbrock minimization problem using PETSc TAO.

    The objective function is the classic non-convex test function:

        f(x) = sum_{i=0}^{n-2} [ 100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2 ]

    Its unique global minimum is at x* = (1, 1, ..., 1) with f(x*) = 0.
    The function is notoriously difficult for gradient-based methods due to
    its narrow, curved valley.

    Parameters
    ----------
    n : int
        Problem dimension (must be >= 2).
    x0 : array-like or None
        Initial guess vector of length n.  If None, the classic alternating
        start (-1.2, 1.0, -1.2, 1.0, ...) is used.
    tao_type : str
        TAO solver type string.  Defaults to 'lmvm' (L-BFGS).
        Other options: 'bncg', 'nls', 'cg'.

    Attributes
    ----------
    obj_history : list of float
        Objective value recorded at every function evaluation; populated
        after solve() is called.
    """

    def __init__(self, n: int = 2, x0=None, tao_type: str = "lmvm"):
        if n < 2:
            raise ValueError("Problem dimension n must be >= 2.")
        self.n = n
        self.tao_type = tao_type
        self.obj_history: list[float] = []

        if x0 is None:
            # Classic Moré–Garbow–Hillstrom starting point
            x0 = np.array([-1.2 if i % 2 == 0 else 1.0 for i in range(n)],
                          dtype=float)
        self.x0 = np.asarray(x0, dtype=float)
        if self.x0.shape != (n,):
            raise ValueError(f"x0 must have length {n}.")

    #  Callback: objective

    def objective(self, tao: "PETSc.TAO", x: "PETSc.Vec") -> float:
        """
        Evaluate the Rosenbrock objective f(x).

        This method is passed as a callback to ``tao.setObjective``.  TAO
        calls it at each iteration with the current iterate x.

        Parameters
        ----------
        tao : PETSc.TAO
            The nonlinear solver context (provided automatically by TAO).
        x : PETSc.Vec
            Current iterate, distributed across MPI ranks.

        Returns
        -------
        float
            Scalar objective value f(x).
        """
        xa = x.getArray(readonly=True)
        f = float(np.sum(100.0 * (xa[1:] - xa[:-1] ** 2) ** 2
                         + (1.0 - xa[:-1]) ** 2))
        self.obj_history.append(f)
        return f

    # Callback: gradient 

    def gradient(self, tao: "PETSc.TAO",
                 x: "PETSc.Vec", g: "PETSc.Vec") -> None:
        """
        Evaluate the Rosenbrock gradient ∇f(x) and write it into g.

        This method is passed as a callback to ``tao.setGradient``.

        The gradient components are:

        ∂f/∂x[i] = -400·x[i]·(x[i+1] - x[i]²) - 2·(1 - x[i])
                   + 200·(x[i] - x[i-1]²)         (boundary terms omitted)

        Parameters
        ----------
        tao : PETSc.TAO
            The nonlinear solver context (provided automatically by TAO).
        x : PETSc.Vec
            Current iterate.
        g : PETSc.Vec
            Output vector; this function must populate it with ∇f(x).
        """
        xa = x.getArray(readonly=True)
        ga = np.zeros(self.n, dtype=float)

        # Contributions from term i:  100*(x[i+1]-x[i]^2)^2 + (1-x[i])^2
        ga[:-1] += (-400.0 * xa[:-1] * (xa[1:] - xa[:-1] ** 2)
                    - 2.0 * (1.0 - xa[:-1]))
        ga[1:]  +=  200.0 * (xa[1:] - xa[:-1] ** 2)

        g.setArray(ga)

    # Solver

    def solve(self) -> tuple[np.ndarray, float]:
        """
        Configure and run the TAO optimiser.

        Creates a PETSc TAO context, registers the objective and gradient
        callbacks, sets tolerances, and calls ``tao.solve()``.

        Returns
        -------
        x_opt : numpy.ndarray, shape (n,)
            Optimal solution vector approximating x* = (1, ..., 1).
        fval : float
            Objective value at x_opt, approximating 0.
        """
        # solution vector
        x = PETSc.Vec().create(PETSc.COMM_WORLD)
        x.setSizes(self.n)
        x.setFromOptions()
        x.setArray(self.x0.copy())

        # gradient vector (pre-allocated; handed to TAO)
        g = x.duplicate()

        # TAO context
        tao = PETSc.TAO().create(PETSc.COMM_WORLD)
        tao.setType(self.tao_type)

        # Register callbacks (the three documented functions)
        tao.setObjective(self.objective)
        tao.setGradient(self.gradient, g)

        # Convergence criteria:
        #   gatol  – absolute gradient norm tolerance  ||∇f|| < gatol
        #   grtol  – relative gradient norm tolerance  ||∇f|| < grtol·||∇f₀||
        #   gttol  – gradient norm tolerance ratio (disabled with 0.0)
        tao.setTolerances(gatol=1e-8, grtol=1e-8, gttol=0.0)
        tao.setFromOptions()
        tao.setSolution(x)

        # solve
        tao.solve()

        # report
        reason = tao.getConvergedReason()   # ← documented function #3
        iters  = tao.getIterationNumber()
        fval   = tao.getObjectiveValue()
        

        print(f"  TAO type          : {self.tao_type}")
        print(f"  Converged reason  : {reason}")
        print(f"  Iterations        : {iters}")
        print(f"  Final f(x)        : {fval:.6e}")
        

        x_opt = x.getArray().copy()

        # clean up PETSc objects
        tao.destroy()
        x.destroy()
        g.destroy()

        return x_opt, fval

    # Visualisation

    def plot_convergence(self) -> None:
        """
        Plot objective value history on a log scale.

        Uses ``self.obj_history`` populated during ``solve()``.
        """
        if not self.obj_history:
            print("No history to plot.  Call solve() first.")
            return

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.semilogy(self.obj_history, marker="o", markersize=3,
                    linewidth=1.5, color="#2563EB")
        ax.set_xlabel("Objective Evaluation Count", fontsize=12)
        ax.set_ylabel("f(x)  [log scale]", fontsize=12)
        ax.set_title("Rosenbrock Convergence History  (TAO L-BFGS)", fontsize=13)
        ax.grid(True, which="both", alpha=0.35)
        fig.tight_layout()
        plt.savefig("plot_" + str(id(plt.gcf())) + ".png", dpi=150, bbox_inches="tight"); plt.close()

    def plot_landscape_2d(self, x_opt: np.ndarray | None = None) -> None:
        """
        Plot the 2D Rosenbrock landscape with the optimisation path.

        Only meaningful when n == 2.

        Parameters
        ----------
        x_opt : numpy.ndarray or None
            If provided (shape (2,)), marks the converged solution on the plot.
        """
        if self.n != 2:
            print("Landscape plot only meaningful for n=2; skipping.")
            return

        x1 = np.linspace(-2.0, 1.6, 500)
        x2 = np.linspace(-0.5, 2.5, 500)
        X1, X2 = np.meshgrid(x1, x2)
        Z = 100.0 * (X2 - X1 ** 2) ** 2 + (1.0 - X1) ** 2

        fig, ax = plt.subplots(figsize=(7, 6))
        cs = ax.contourf(X1, X2, np.log1p(Z), levels=50, cmap="viridis")
        fig.colorbar(cs, ax=ax, label="log(1 + f(x₁, x₂))")

        ax.plot(1.0, 1.0, "r*", markersize=14, label="Global min (1, 1)",
                zorder=5)
        ax.plot(self.x0[0], self.x0[1], "w^", markersize=9,
                label=f"Start ({self.x0[0]:.1f}, {self.x0[1]:.1f})",
                zorder=5)
        if x_opt is not None:
            ax.plot(x_opt[0], x_opt[1], "wx", markersize=11, markeredgewidth=2,
                    label=f"Result ({x_opt[0]:.4f}, {x_opt[1]:.4f})",
                    zorder=5)

        ax.set_xlabel("x₁", fontsize=12)
        ax.set_ylabel("x₂", fontsize=12)
        ax.set_title("Rosenbrock Function — 2D Landscape", fontsize=13)
        ax.legend(framealpha=0.85)
        fig.tight_layout()
        plt.savefig("plot_" + str(id(plt.gcf())) + ".png", dpi=150, bbox_inches="tight"); plt.close()


# Entry point

if __name__ == "__main__":
    print("=" * 55)
    print("  Rosenbrock Minimisation — PETSc TAO (n=2)")
    print("=" * 55)
    solver = RosenbrockSolver(n=2)
    x_opt, fval = solver.solve()
    print(f"\n  Optimal x  : {x_opt}")
    print(f"  Optimal f  : {fval:.4e}")

    solver.plot_landscape_2d(x_opt=x_opt)
    solver.plot_convergence()

    # Higher-dimensional test
    print("\n" + "=" * 55)
    print("  Rosenbrock Minimisation — PETSc TAO (n=10)")
    print("=" * 55)
    solver10 = RosenbrockSolver(n=10)
    x_opt10, fval10 = solver10.solve()
    print(f"\n  ||x* - 1||₂ = {np.linalg.norm(x_opt10 - 1.0):.4e}")
