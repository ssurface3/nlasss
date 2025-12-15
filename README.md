# nlasss
run_project.py  (Start Here)
    │
    ├── imports --> ssgetpy (Download)
    │
    └── imports --> benchmark_runner.py
                        │
                        ├── imports --> nlass.py
                        │                   │
                        │                   └── imports --> ric2cg.py
                        │
                        └── saves to -> benchmark_results.csv

plot_results.py (Run Last)
    │
    ├── reads ----> benchmark_results.csv
    │
    └── imports --> ric2cg.py (To re-run trace for history plot)


### Project Overview
**Goal:** To implement the **RIC2S** (Robust Incomplete Cholesky 2nd Order Stabilized) preconditioner from the Kaporin (1998) paper and benchmark it against standard methods (CG, Jacobi, ILU) on sparse matrices.

---

### The File Structure

#### 1. `ric2cg.py` (The Algorithm)
*   **Role:** The Mathematical Core.
*   **Function:**
    *   Implements the custom **RIC2** and **RIC2S** factorization algorithms.
    *   Handles the **Scaling** ($DAD$) required by the paper.
    *   Performs the sparse matrix decomposition ($U, R$) using efficient Python structures (`lil_array` for build, `csr` for solve).
    *   Contains the `solve()` method that runs Conjugate Gradients using the calculated $U$ and $R$.

#### 2. `nlass.py` (The Test Bench)
*   **Role:** The Standardized Testing Class.
*   **Function:**
    *   Wraps every solver (SciPy's CG, ILU, SPLU, and your RIC2CG) into a uniform interface.
    *   Ensures every test returns the exact same metrics: `Time`, `Iterations`, `Residual`, and `NNZ` (Memory usage).
    *   Implements **Symmetric Gauss-Seidel (SGS)** manually since SciPy doesn't have it.
    *   Catches errors (singular matrices, divergence) so one bad matrix doesn't crash the whole project.

#### 3. `benchmark_runner.py` (The Manager)
*   **Role:** The Campaign Manager.
*   **Function:**
    *   Defines specific testing "campaigns":
        *   `run_baselines`: Runs CG, Jacobi, SGS, Exact LU.
        *   `run_ilu_sensitivity`: Runs ILU with tolerances 1e-3, 1e-4, 1e-5.
        *   `run_ric2s_sensitivity`: Runs your RIC2S with tau 0.1, 0.05.
    *   Manages the **CSV Database** (`benchmark_results.csv`).
    *   Saves progress row-by-row. If you stop the script and restart, it continues where it left off.

#### 4. `run_project.py` (The Controller)
*   **Role:** The Master Script (Entry Point).
*   **Function:**
    *   **Downloads data:** Checks if you have matrices; if not, uses `ssgetpy` to download them from the SuiteSparse collection.
    *   **Generates Registry:** Creates `matrix_registry.csv` to track file paths.
    *   **Orchestrates Benchmarks:** Calls the functions in `benchmark_runner.py` in the correct order.

#### 5. `plot_results.py` (The Visualizer)
*   **Role:** The Data Analyst.
*   **Function:**
    *   Reads `benchmark_results.csv`.
    *   Calculates **Fill-in Factor** (how much memory the preconditioner used relative to the original matrix).
    *   Generates the 3 key images in the `graphs/` folder:
        1.  **Pareto Frontier:** Scatter plot of Memory vs. Speed.
        2.  **Convergence History:** Re-runs a specific "hard" matrix to plot error vs. iterations.
        3.  **Robustness:** Bar chart of success rates.
-------------------------------------------------------------------------------------------------------------------------
### Execution Order
1.  Run `python run_project.py` (Downloads matrices and runs hours of benchmarks).
2.  Run `python plot_results.py` (Generates images for report).