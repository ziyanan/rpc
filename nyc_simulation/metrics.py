from typing import Dict, List, Optional, Set
import numpy as np


def compute_step_metrics(traci, controlled_edges: Optional[Set[str]] = None) -> Dict[str, float]:
    vehicle_ids = traci.vehicle.getIDList()
    if vehicle_ids:
        step_delays = [
            traci.vehicle.getWaitingTime(vid)
            for vid in vehicle_ids
        ]
        avg_step_delay = float(np.mean(step_delays))
        max_step_delay = float(np.max(step_delays))
    else:
        avg_step_delay = 0.0
        max_step_delay = 0.0

    all_edge_ids = traci.edge.getIDList()
    if all_edge_ids:
        halting = [traci.edge.getLastStepHaltingNumber(eid) for eid in all_edge_ids]
        occupancies = [traci.edge.getLastStepOccupancy(eid) for eid in all_edge_ids]

        avg_queue = float(np.mean(halting))
        max_queue = float(np.max(halting))
        avg_occupancy = float(np.mean(occupancies))
    else:
        avg_queue = 0.0
        max_queue = 0.0
        avg_occupancy = 0.0

    throughput = float(traci.simulation.getArrivedNumber())

    return {
        "avg_step_delay": avg_step_delay,
        "max_step_delay": max_step_delay,

        "avg_queue": avg_queue,
        "max_queue": max_queue,

        "avg_occupancy": avg_occupancy,

        "throughput": throughput,
    }


def summarize_metrics(rows: List[Dict[str, float]], epsilon_cert: float) -> Dict[str, float]:
    certified_flags = [
        1.0 if r["certified_radius"] >= epsilon_cert else 0.0
        for r in rows
    ]

    return {
        "avg_step_delay": float(np.mean([r["avg_step_delay"] for r in rows])),
        "p95_step_delay": float(np.percentile(
            [r["avg_step_delay"] for r in rows], 95
        )),
        "max_step_delay": float(np.max([r["max_step_delay"] for r in rows])),

        "avg_queue": float(np.mean([r["avg_queue"] for r in rows])),
        "max_queue": float(np.max([r["max_queue"] for r in rows])),

        "avg_occupancy": float(np.mean([r["avg_occupancy"] for r in rows])),

        "throughput": float(np.mean([r["throughput"] for r in rows])),

        "fallback_rate": float(np.mean([r["fallback"] for r in rows])),
        "certified_coverage": float(np.mean(certified_flags)),
        "certified_radius_mean": float(np.mean([r["certified_radius"] for r in rows])),
        "certified_radius_median": float(np.median([r["certified_radius"] for r in rows])),
    }
