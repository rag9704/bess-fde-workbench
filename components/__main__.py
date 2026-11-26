"""Solve a site described in YAML.

python -m components sites/tokyo_office_bess.yaml --steps 24
python -m components sites/factory_inventory.yaml --show line.output warehouse.level
"""

import argparse
import sys

import numpy as np

from core_model.solver_runner import Infeasible

from . import ConfigError, load_site, solve_site


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("site", help="Site YAML file")
    ap.add_argument("--steps", type=int, default=None, help="Number of steps to plan (default: all data)")
    ap.add_argument("--start", type=int, default=0, help="First row of the data")
    ap.add_argument("--solver", choices=("highs", "scip"), default="highs")
    ap.add_argument("--show", nargs="*", default=[], help="Print these series, e.g. battery.level grid.buy")
    args = ap.parse_args(argv)

    try:
        site = load_site(args.site)
        df = site.load_data()
        end = None if args.steps is None else args.start + args.steps
        window = df.iloc[args.start : end].reset_index(drop=True)
        result = solve_site(site, window, solver=args.solver)
    except ConfigError as exc:
        print(f"Invalid site: {exc}", file=sys.stderr)
        return 2
    except Infeasible:
        print("Infeasible: the site's limits cannot all be met over this window.", file=sys.stderr)
        return 1

    print(site.description)
    print(f"{len(window)} steps from {window['timestamp'].iloc[0] if 'timestamp' in window else 0}")
    print(result.summary())
    for key in args.show:
        vals = result.get(key)
        print(f"\n{key}:")
        print(np.array2string(np.round(vals, 1) + 0.0, max_line_width=110))  # +0.0: no "-0."
    return 0


if __name__ == "__main__":
    sys.exit(main())
