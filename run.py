"""
A dice game with the following rules:

1. Roll 6d6
2. Reroll all non-unique-valued dice
3. Stop when you have 1, 2, 3, 4, 5, 6
"""

import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from collections import Counter
from itertools import product
from matplotlib.ticker import PercentFormatter
from random import randint


def game(d_what):
    tries = 0
    dice = {}
    while len(dice) < d_what:
        tries += 1
        dice = {n: v for (n, v) in dice.items() if v == 1}
        for _ in range(0, d_what - len(dice)):
            roll = randint(1, d_what)
            dice[roll] = dice.get(roll, 0) + 1
    return tries


def monte(trials, d_what):
    return [game(d_what) for _ in range(trials)]


"""
d_what  ev
     2  2.0 (2/1)
     3  4.5 (18/4)
     4  8.88 (80/9)
     5  16.74, 16.76 (268/16?)
     6  31,
"""


def ev(trials, d_what):
    total = 0
    for trial in range(trials):
        total += game(d_what)
    return total / trials


def ev_outcomes(d_what, fixed):
    prefix = list(range(1, fixed + 1))
    return [
        prefix + list(c) for c in product(range(1, d_what + 1), repeat=d_what - fixed)
    ]


def ev_coeffs(d_what, fixed):
    coeffs = dict.fromkeys(range(d_what + 1), 0)
    coeffs.update(
        Counter(
            [
                len({n: v for (n, v) in Counter(x).items() if v == 1})
                for x in ev_outcomes(d_what=d_what, fixed=fixed)
            ]
        )
    )
    return list(coeffs.values())


def calc_ev(d_what):
    print(f"Running for {d_what}d{d_what}.")
    print(f"Solving {d_what - 1} linear equation{'s' if d_what > 2 else ''}:")
    all_coeffs = []
    consts = []
    for fixed in range(0, d_what - 1):
        coeffs = ev_coeffs(d_what=d_what, fixed=max(1, fixed))
        total = sum(coeffs)
        # print(f"{coeffs=}")
        consts.append(-1 * (coeffs[0] + coeffs[d_what] + (total if fixed > 0 else 0)))
        coeffs[fixed] -= total
        all_coeffs.append(coeffs[:-2])
        strs = [
            f"{str(v).rjust(d_what + 1)}*X_{n}" for (n, v) in enumerate(all_coeffs[-1])
        ]
        print(str(consts[-1]).rjust(d_what + 1), f"=", " + ".join(strs))
        # print(f"{fixed=}, {total=}, {const=}, {coeffs=}")
    x = np.linalg.solve(np.array(all_coeffs), np.array(consts))
    ev = x[0]
    print(f"EV = {ev}")
    return ev


def monte_histo(trials, d_what):
    histo = {}
    for _ in range(trials):
        tries = game(d_what)
        histo[tries] = histo.get(tries, 0) + 1
    return dict(sorted(histo.items()))


def parse_args():
    parser = ArgumentParser(description="Run Monte Carlo sim of a little dice game")
    parser.add_argument(
        "-t",
        type=int,
        default=1000000,
        help="Num of trials to run (default %(default)s)",
    )
    parser.add_argument(
        "-d",
        metavar="D_WHAT",
        type=int,
        default=6,
        help="Sides of dice in game (default %(default)s)",
    )
    parser.add_argument(
        "--ev",
        action="store_true",
        help="Calculate expected value using numpy",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trials = args.t
    d_what = args.d
    if args.ev:
        calc_ev(d_what=d_what)
        exit(0)

    data = monte(trials=trials, d_what=d_what)

    n, bins, patches = plt.hist(data, max(data), density=True, color="green")

    plt.xlabel("# Tries")
    plt.ylabel("% Results")
    plt.title(f"Distribution with {d_what}D{d_what} ({trials:,} trials)")
    plt.xlim(1)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    plt.show()
    # for (tries, v) in histo.items():
    #     print(f"{tries}\t{v}")
