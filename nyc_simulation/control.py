from typing import Optional, List, Set


class FixedTimeController:
    def __init__(self, green_extension: float = 5.0, congestion_threshold: float = 0.04):
        self.green_extension = green_extension
        self.congestion_threshold = congestion_threshold

    def select_action(self, predicted_occupancy: float, is_certified: bool) -> bool:
        if not is_certified:
            return False
        return predicted_occupancy >= self.congestion_threshold


def apply_traffic_light_actions(traci, tls_ids: List[str], extend_green: bool, delta: float) -> str:
    if not tls_ids:
        return "no_tls"
    
    if not extend_green:
        return "default_timing"
    
    extend_count = 0
    force_count = 0
    
    for tls_id in tls_ids:
        state = traci.trafficlight.getRedYellowGreenState(tls_id)
        
        if "G" in state:
            remaining = traci.trafficlight.getNextSwitch(tls_id) - traci.simulation.getTime()
            if remaining > 0:
                traci.trafficlight.setPhaseDuration(tls_id, remaining + 20)
                extend_count += 1
        else:
            traci.trafficlight.setPhase(tls_id, 0)
            traci.trafficlight.setPhaseDuration(tls_id, 20)
            force_count += 1
    
    if extend_count > 0 or force_count > 0:
        return f"extended_{extend_count}_forced_{force_count}"
    return "default_timing"


def apply_traffic_light_action(traci, tls_id: str, extend_green: bool, delta: float) -> str:
    if not tls_id:
        return "no_tls"

    if not extend_green:
        return "default_timing"

    state = traci.trafficlight.getRedYellowGreenState(tls_id)

    if "G" in state:
        remaining = traci.trafficlight.getNextSwitch(tls_id) - traci.simulation.getTime()
        if remaining > 0:
            traci.trafficlight.setPhaseDuration(tls_id, remaining + 20)
            remaining_after = traci.trafficlight.getNextSwitch(tls_id) - traci.simulation.getTime()
            print(
                f"Extend green: remaining_before={remaining:.1f}, "
                f"remaining_after={remaining_after:.1f}, delta=20.0"
            )
            return "extend_green"
        return "default_timing"
    else:
        traci.trafficlight.setPhase(tls_id, 0)
        traci.trafficlight.setPhaseDuration(tls_id, 20)
        print(f"Force green: set to phase 0 for 20s")
        return "force_green"



def select_top_tls_ids(traci, top_n: int = 10) -> List[str]:
    tls_ids = sorted(traci.trafficlight.getIDList())
    tls_activity = {}
    
    for tls_id in tls_ids:
        controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
        vehicle_count = 0
        for lane_id in controlled_lanes:
            try:
                vehicle_count += traci.lane.getLastStepVehicleNumber(lane_id)
            except:
                pass
        tls_activity[tls_id] = vehicle_count
    
    sorted_tls = sorted(tls_activity.items(), key=lambda x: x[1], reverse=True)
    top_tls = [tls_id for tls_id, count in sorted_tls[:top_n]]
    return top_tls


def get_controlled_edges(traci, tls_ids: List[str]) -> Set[str]:
    controlled_edges = set()
    for tls_id in tls_ids:
        controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
        for lane_id in controlled_lanes:
            edge_id = lane_id.rsplit("_", 1)[0] if "_" in lane_id else lane_id
            controlled_edges.add(edge_id)
    return controlled_edges


def select_tls_id(traci) -> Optional[str]:
    tls_ids = sorted(traci.trafficlight.getIDList())
    return tls_ids[0] if tls_ids else None
