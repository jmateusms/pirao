"""From pirao posterior draws to farofa and faultree.

Two chains, both on teaching data (the `relay-binomial` and `pumps-relevance`
examples, whose numbers are made up):

1. relay K2: posterior draws of the per-demand failure probability go straight
   into faultree's pressure-tank tree as samples of K2.
2. pumps: each posterior draw of the failure rate goes through farofa (three
   pumps, three repair teams) to an unavailability U, and faultree turns U into
   the probability that the station lacks flow (2 of 3 pumps down, or no power).
   The three identical pumps share one draw per scenario: the uncertainty is
   about a rate they have in common, not three unrelated rates.

Get the draws from the GUI (Results -> Draws (CSV)) or from the shell:

    pirao example relay-binomial --out relay
    pirao run relay/spec.json --data relay/data.csv --sampler relay/sampler.json --out relay/run
    pirao example pumps-relevance --out pumps
    pirao run pumps/spec.json --data pumps/data.csv --sampler pumps/sampler.json --out pumps/run

then, in an environment with farofa, faultree and numpy:

    python farofa_faultree_chain.py relay/run/draws.csv pumps/run/draws.csv \
        --tank path/to/faultree/examples/pressure_tank.json
"""

from __future__ import annotations

import argparse
import csv
import json

import farofa
import numpy as np
from faultree import analyze

REPAIR_MEDIAN_H, REPAIR_SIGMA = 30.0, 0.8  # pumps of the farofa/faultree deck
MISSION_H = 8760.0
P_POWER = 1e-3


def column(path: str, name: str) -> np.ndarray:
    with open(path, newline="") as fh:
        return np.array([float(row[name]) for row in csv.DictReader(fh)])


def pump_unavailability(rate: float, reps: int, seed: int) -> float:
    fleet = farofa.Fleet(n_devices=3, n_teams=3)
    fleet.set_failure_dist("exponential", rate)
    fleet.set_repair_dist("lognormal", np.log(REPAIR_MEDIAN_H), REPAIR_SIGMA)
    fleet.set_mission_time(MISSION_H)
    return 1.0 - float(fleet.simulate(reps=reps, seed=seed).fleet_availability)


def station_tree() -> dict:
    def basic(i: str) -> dict:
        return {"id": i, "event_type": "basic", "gate": None, "children": []}

    pumps = [basic("B1"), basic("B2"), basic("B3")]
    return {
        "id": "TOP", "event_type": "top", "gate": "OR",
        "children": [
            basic("POWER"),
            {"id": "PUMPS", "event_type": "intermediate", "gate": "K_OF_N", "k": 2,
             "children": pumps},
        ],
    }


def pct(values: np.ndarray) -> str:
    lo, mid, hi = np.percentile(values, [5, 50, 95])
    return f"median {mid:.3g}, 90% interval {lo:.3g} to {hi:.3g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("relay_draws", help="draws.csv of the relay-binomial run")
    parser.add_argument("pump_draws", help="draws.csv of the pumps-relevance run")
    parser.add_argument("--tank", required=True, help="faultree pressure_tank.json")
    parser.add_argument("--scenarios", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    tank = json.load(open(args.tank))
    k2 = column(args.relay_draws, "prob")
    q_point = analyze(tank)["Q"]
    q_tank = np.asarray(analyze(tank, {"K2": k2.tolist()})["Q"])
    print(f"Tank, K2 at its point value: Q = {q_point:.3g}")
    print(f"Tank, K2 from the posterior: Q {pct(q_tank)}, mean {q_tank.mean():.3g}")

    rates = rng.choice(column(args.pump_draws, "rate"), args.scenarios, replace=False)
    u = np.array([pump_unavailability(r, 60, args.seed) for r in rates])
    tree = station_tree()
    power = [P_POWER] * len(u)
    shared = np.asarray(
        analyze(tree, {"POWER": power, "B1": u.tolist(), "B2": u.tolist(),
                       "B3": u.tolist()})["Q"]
    )
    shuffled = [rng.permutation(u).tolist() for _ in range(3)]
    independent = np.asarray(
        analyze(tree, {"POWER": power, "B1": shuffled[0], "B2": shuffled[1],
                       "B3": shuffled[2]})["Q"]
    )
    print(f"Pump unavailability U: {pct(u)}")
    print(f"Station lacks flow, one shared draw per scenario: {pct(shared)}")
    print(f"Same, pumps drawn independently (understates the tail): {pct(independent)}")


if __name__ == "__main__":
    main()
