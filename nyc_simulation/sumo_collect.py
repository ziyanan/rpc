import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _load_edge_lanes(net_path: str) -> Dict[str, List[Tuple[str, float]]]:
    tree = ET.parse(net_path)
    root = tree.getroot()
    edge_lanes: Dict[str, List[Tuple[str, float]]] = {}
    for edge in root.iter("edge"):
        edge_id = edge.attrib.get("id", "")
        if edge_id.startswith(":"):
            continue
        if edge_id.startswith("junction"):
            continue
        edge_lanes.setdefault(edge_id, [])
        for lane in edge.iter("lane"):
            lane_id = lane.attrib.get("id", "")
            if lane_id.startswith(":"):
                continue
            length_str = lane.attrib.get("length", "0")
            try:
                length = float(length_str)
            except ValueError:
                length = 0.0
            if length > 0:
                edge_lanes[edge_id].append((lane_id, length))
    return edge_lanes


def _write_detectors_file(detector_ids: List[Tuple[str, float]], out_path: str) -> None:
    root = ET.Element("additional")
    for idx, (lane_id, lane_length) in enumerate(detector_ids):
        det = ET.SubElement(root, "laneAreaDetector")
        det.set("id", f"det_{idx}")
        det.set("lane", lane_id)
        desired_pos = 5.0
        pos = min(desired_pos, 0.9 * lane_length)
        pos = max(0.0, pos)
        det.set("pos", f"{pos:.3f}")
        det.set("length", "50")
        det.set("freq", "60")
        det.set("file", "detector_output.xml")
    tree = ET.ElementTree(root)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)


def _write_actuated_net(net_path: str, out_path: str) -> str:
    tree = ET.parse(net_path)
    root = tree.getroot()
    for tl in root.iter("tlLogic"):
        tl_type = tl.attrib.get("type", "")
        if tl_type in {"static", "fixed"}:
            tl.set("type", "actuated")
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def _write_augmented_sumocfg(
    sumocfg_path: str,
    additional_path: str,
    out_path: str,
    end_time: Optional[int],
    net_override_path: Optional[str] = None,
) -> Tuple[str, str]:
    tree = ET.parse(sumocfg_path)
    root = tree.getroot()
    input_node = root.find("input")
    if input_node is None:
        input_node = ET.SubElement(root, "input")
    sumo_dir = os.path.dirname(os.path.abspath(sumocfg_path))

    net_node = input_node.find("net-file")
    if net_node is not None:
        if net_override_path:
            net_node.set("value", os.path.abspath(net_override_path))
        else:
            net_value = net_node.attrib.get("value", "")
            if net_value:
                net_path = net_value if os.path.isabs(net_value) else os.path.join(sumo_dir, net_value)
                net_node.set("value", os.path.abspath(net_path))

    route_node = input_node.find("route-files")
    if route_node is not None:
        route_value = route_node.attrib.get("value", "")
        if route_value:
            routes = []
            for p in route_value.split(","):
                route_path = p if os.path.isabs(p) else os.path.join(sumo_dir, p)
                routes.append(os.path.abspath(route_path))
            route_node.set("value", ",".join(routes))
    add_node = input_node.find("additional-files")
    if add_node is None:
        add_node = ET.SubElement(input_node, "additional-files")
        add_node.set("value", os.path.abspath(additional_path))
    else:
        existing = add_node.attrib.get("value", "")
        if existing:
            add_node.set("value", f"{existing},{os.path.abspath(additional_path)}")
        else:
            add_node.set("value", os.path.abspath(additional_path))
    if end_time is not None:
        time_node = root.find("time")
        if time_node is None:
            time_node = ET.SubElement(root, "time")
        end_node = time_node.find("end")
        if end_node is None:
            end_node = ET.SubElement(time_node, "end")
        end_node.set("value", str(end_time))

    tree.write(out_path, encoding="utf-8", xml_declaration=True)

    net_file = input_node.find("net-file").attrib.get("value")
    return net_file, out_path


def _get_net_path(sumocfg_path: str) -> str:
    tree = ET.parse(sumocfg_path)
    root = tree.getroot()
    input_node = root.find("input")
    if input_node is None:
        raise RuntimeError("SUMO config missing <input> section")
    net_node = input_node.find("net-file")
    if net_node is None:
        raise RuntimeError("SUMO config missing <net-file>")
    net_value = net_node.attrib.get("value", "")
    if not net_value:
        raise RuntimeError("SUMO config net-file value is empty")
    sumo_dir = os.path.dirname(os.path.abspath(sumocfg_path))
    net_path = net_value if os.path.isabs(net_value) else os.path.join(sumo_dir, net_value)
    return os.path.abspath(net_path)


def _resolve_sumo_binary(sumo_home: Optional[str]) -> str:
    try:
        import sumolib
    except Exception as exc:
        raise RuntimeError("SUMO is required. Ensure SUMO is installed and SUMO_HOME is set.") from exc

    if sumo_home:
        if sumo_home.endswith("/sumo") and os.path.isfile(sumo_home):
            return sumo_home
        candidate = os.path.join(sumo_home, "bin", "sumo")
        if os.path.isfile(candidate):
            return candidate
    return sumolib.checkBinary("sumo")


def _init_traci(sumocfg_path: str, seed: int, sumo_home: Optional[str]):
    try:
        import traci
    except Exception as exc:
        raise RuntimeError("SUMO TraCI is required. Ensure SUMO is installed and SUMO_HOME is set.") from exc

    sumo_binary = _resolve_sumo_binary(sumo_home)
    if sumo_home and not sumo_home.endswith("/sumo"):
        os.environ["SUMO_HOME"] = sumo_home
    traci.start([sumo_binary, "-c", sumocfg_path, "--seed", str(seed)])
    return traci


def _scan_active_edges(sumocfg_path: str, seed: int, sumo_home: Optional[str], scan_steps: int = 50, max_edges: int = 10) -> List[str]:
    traci = _init_traci(sumocfg_path, seed, sumo_home)
    edge_activity: Dict[str, int] = {}
    try:
        for step in range(scan_steps):
            traci.simulationStep()
            if step % 3 != 0:
                continue
            for edge_id in traci.edge.getIDList():
                if edge_id not in edge_activity:
                    edge_activity[edge_id] = 0
                edge_activity[edge_id] += traci.edge.getLastStepVehicleNumber(edge_id)
    finally:
        traci.close()

    active_edges = []
    for edge_id, activity in sorted(edge_activity.items(), key=lambda x: x[1], reverse=True):
        if edge_id.startswith(":") or edge_id.startswith("junction"):
            continue
        if len(active_edges) >= max_edges:
            break
        if activity > 0:
            active_edges.append(edge_id)
    return active_edges


def _select_lanes_from_edges(edge_lanes: Dict[str, List[Tuple[str, float]]], edges: List[str], max_det: int = 4) -> List[Tuple[str, float]]:
    selected: List[Tuple[str, float]] = []
    for edge_id in edges:
        lanes = edge_lanes.get(edge_id, [])
        if not lanes:
            continue
        lanes_sorted = sorted(lanes, key=lambda x: x[1], reverse=True)
        selected.append(lanes_sorted[0])
        if len(selected) >= max_det:
            break
    return selected


def collect_sumo_data(
    sumocfg_path: str,
    out_dir: str,
    detector_ids: Optional[List[str]] = None,
    dt_seconds: int = 60,
    seed: int = 42,
    sumo_home: Optional[str] = None,
    end_time: Optional[int] = None,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    sumo_dir = os.path.dirname(sumocfg_path)
    augmented_cfg = os.path.join(out_dir, "augmented.sumocfg")
    detectors_path = os.path.join(out_dir, "additional_detectors.xml")

    if detector_ids is None:
        net_path = _get_net_path(sumocfg_path)
        edge_lanes = _load_edge_lanes(net_path)
        active_edges = _scan_active_edges(sumocfg_path, seed, sumo_home)
        detector_ids = _select_lanes_from_edges(edge_lanes, active_edges, max_det=4)
        if not detector_ids:
            all_lanes = []
            for lanes in edge_lanes.values():
                all_lanes.extend(lanes)
            detector_ids = all_lanes[:4]
        if not detector_ids:
            raise RuntimeError("No lanes found in network; cannot create detectors.")

    _write_detectors_file(detector_ids, detectors_path)
    _, cfg_path = _write_augmented_sumocfg(sumocfg_path, detectors_path, augmented_cfg, end_time)

    traci = _init_traci(cfg_path, seed, sumo_home)

    timestep = 0
    buffer_flow, buffer_speed, buffer_occ = [], [], []
    rows = []

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            timestep += 1
            if end_time is not None and timestep >= end_time:
                break

            det_ids = traci.lanearea.getIDList()
            if not det_ids:
                raise RuntimeError("No lane-area detectors found in simulation.")

            flows = [traci.lanearea.getLastStepVehicleNumber(d) for d in det_ids]
            speeds = [traci.lanearea.getLastStepMeanSpeed(d) for d in det_ids]
            occs = [traci.lanearea.getLastStepOccupancy(d) for d in det_ids]

            buffer_flow.append(np.mean(flows))
            buffer_speed.append(np.mean(speeds))
            buffer_occ.append(np.mean(occs))

            if timestep % dt_seconds == 0:
                rows.append(
                    {
                        "time": timestep,
                        "total_flow": float(np.mean(buffer_flow)),
                        "avg_speed": float(np.mean(buffer_speed)),
                        "avg_occupancy": float(np.mean(buffer_occ)),
                    }
                )
                buffer_flow, buffer_speed, buffer_occ = [], [], []
    finally:
        traci.close()

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "sumo_detector_timeseries.csv")
    df.to_csv(csv_path, index=False)
    return csv_path
