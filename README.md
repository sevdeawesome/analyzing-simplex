# Simplex Algorithm Assignment
This READMe explains the files used

## Files
**`mvp.ipynb`** - solving the initial problem from the lecture to understand the library!!

**`generate_data.py`**: data generation script with two modes:
- `--mode example` (default): Builds on lecture problem, extends to larger n and m
- `--mode random`: Generates fully random LP instances

Example usage:
python generate_data.py --mode example


```bash
pip install numpy pandas scipy
```

Uses `scipy.optimize.linprog` with `method='highs'`

## Output
Results saved as CSV with columns:
- `n`, `m`: Problem dimensions
- `runtime`: Solve time in seconds
- `nit`: Number of iterations
- `optimal_value`: Objective function value
- `success`: Boolean feasibility flag
