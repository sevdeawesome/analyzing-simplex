import pandas as pd
import numpy as np
from scipy.optimize import linprog
import time
import argparse


def solve_example_problem():
    print("Solving example problem (n=4, m=2)...")

    c = [7, 4, 6, 1]
    A_eq = [[1, 2, -1, -1], [-1, -5, 2, 3]]
    b_eq = [1, 1]
    bounds = [(0, None)] * 4

    start = time.time()
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    runtime = time.time() - start

    if res.success:
        print(f"  Optimal: {res.fun:.3f}, Solution: {res.x}, Iterations: {res.nit}, Time: {runtime:.4f}s\n")
    else:
        print(f"  Failed: {res.message}\n")

    return res, runtime


def generate_random_lp_dataset(n_values, m_values, seed=1):
    np.random.seed(seed)
    results = []

    total = len(n_values) * len(m_values)
    count = 0

    print(f"Generating {total} random LP problems...")
    print(f"n = {n_values}, m = {m_values}\n")

    for n in n_values:
        for m in m_values:
            count += 1
            print(f"[{count}/{total}] n={n}, m={m}... ", end="")

            A_eq = np.random.randint(-5, 6, size=(m, n))
            b_eq = np.random.randint(1, 11, size=m)
            c = np.random.randint(1, 11, size=n)
            bounds = [(0, None)] * n

            start = time.time()
            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            runtime = time.time() - start

            results.append({
                'n': n, 'm': m, 'runtime': runtime,
                'nit': res.nit if res.success else None,
                'optimal_value': res.fun if res.success else None,
                'success': res.success, 'status': res.status,
                'message': res.message
            })

            status = "OK" if res.success else "FAIL"
            print(f"{status} ({runtime:.4f}s)")

    return pd.DataFrame(results)


def generate_example_based_lp_dataset(n_values, m_values, seed=1):
    np.random.seed(seed)
    results = []

    example_A = np.array([[1, 2, -1, -1], [-1, -5, 2, 3]])
    example_b = np.array([1, 1])
    example_c = np.array([7, 4, 6, 1])

    total = len(n_values) * len(m_values)
    count = 0

    print(f"Generating {total} example-based LP problems...")
    print(f"n = {n_values}, m = {m_values}\n")

    for n in n_values:
        for m in m_values:
            count += 1
            print(f"[{count}/{total}] n={n}, m={m}... ", end="")

            if m >= 2 and n >= 4:
                A_eq = np.zeros((m, n))
                A_eq[0:2, 0:4] = example_A

                if n > 4:
                    A_eq[0:2, 4:n] = np.random.randint(-5, 6, size=(2, n-4))

                if m > 2:
                    A_eq[2:m, :] = np.random.randint(-5, 6, size=(m-2, n))

                b_eq = np.zeros(m)
                b_eq[0:2] = example_b
                if m > 2:
                    b_eq[2:m] = np.random.randint(1, 11, size=m-2)

                c = np.zeros(n)
                c[0:4] = example_c
                if n > 4:
                    c[4:n] = np.random.randint(1, 11, size=n-4)

            elif n >= 4 and m == 1:
                A_eq = np.zeros((1, n))
                A_eq[0, 0:4] = example_A[0]
                if n > 4:
                    A_eq[0, 4:n] = np.random.randint(-5, 6, size=n-4)
                b_eq = np.array([example_b[0]])
                c = np.zeros(n)
                c[0:4] = example_c
                if n > 4:
                    c[4:n] = np.random.randint(1, 11, size=n-4)

            else:
                A_eq = np.random.randint(-5, 6, size=(m, n))
                b_eq = np.random.randint(1, 11, size=m)
                c = np.random.randint(1, 11, size=n)

            bounds = [(0, None)] * n

            start = time.time()
            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            runtime = time.time() - start

            results.append({
                'n': n, 'm': m, 'runtime': runtime,
                'nit': res.nit if res.success else None,
                'optimal_value': res.fun if res.success else None,
                'success': res.success, 'status': res.status,
                'message': res.message
            })

            status = "OK" if res.success else "FAIL"
            print(f"{status} ({runtime:.4f}s)")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['random', 'example'], default='example')
    args = parser.parse_args()

    print("=" * 60)
    print("Simplex Data Generation")
    print("=" * 60 + "\n")

    solve_example_problem()

    n_values = [4, 10, 20, 30, 40, 50, 60, 70, 80]
    m_values = [2, 6, 10, 14, 18, 22]

    if args.mode == 'random':
        print("MODE: Random\n")
        df = generate_random_lp_dataset(n_values, m_values)
        output = 'data_random/simplex_results.csv'
    else:
        print("MODE: Example-based\n")
        df = generate_example_based_lp_dataset(n_values, m_values)
        output = 'data/simplex_results.csv'

    df.to_csv(output, index=False)

    print(f"\n{'=' * 60}")
    print(f"Saved: {output}")
    print(f"Total: {len(df)}, Success: {df['success'].sum()}, Fail: {(~df['success']).sum()}")
    print("=" * 60 + "\n")

    print(df.describe())


if __name__ == '__main__':
    main()
