import itertools
import json
import math
import sys
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, List, Tuple

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.


def debug(message: any):
    """ """
    print(
        message,
        file=sys.stderr,
        flush=False,
    )


# Function to calculate the Euclidean distance between two points
def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


# Function to check if point A is on the segment between B and C
def pointOnSegment(A, B, C):
    epsilon = 0.0000001
    return -epsilon < distance(B, A) + distance(A, C) - distance(B, C) < epsilon


def sign(x):
    """
    Returns:
    -1 if x is negative,
     0 if x is zero,
     1 if x is positive.
    """
    if x < 0:
        return -1
    elif x > 0:
        return 1
    else:
        return 0


def orientation(p1, p2, p3):
    """
    Determines the orientation of the triplet (p1, p2, p3).

    Args:
    p1, p2, p3: Points represented as dictionaries with 'x' and 'y' keys.

    Returns:
    -1 if the points are oriented counterclockwise,
     1 if the points are oriented clockwise,
     0 if the points are collinear.
    """
    # Compute the determinant of the matrix
    prod = (p3["y"] - p1["y"]) * (p2["x"] - p1["x"]) - (p2["y"] - p1["y"]) * (
        p3["x"] - p1["x"]
    )
    return sign(prod)


def segmentsIntersect(A, B, C, D):
    """
    Determines whether two line segments AB and CD intersect.

    Args:
    A, B, C, D: Points representing the endpoints of the two segments. Each point is a dictionary with 'x' and 'y' keys.

    Returns:
    True if the segments AB and CD intersect, False otherwise.
    """
    # Check the orientations of points to see if segments AB and CD intersect
    return (
        orientation(A, B, C) * orientation(A, B, D) < 0
        and orientation(C, D, A) * orientation(C, D, B) < 0
    )


@dataclass
class TransportLine:
    building_1: "Building"
    building_2: "Building"
    capacity: int = 1

    def __str__(self):
        return self.id

    def dump(self):
        debug(
            json.dumps(
                str(
                    {
                        "type": "TransportLine",
                        "id": self.id,
                        "building_1": self.building_1.id,
                        "building_2": self.building_2.id,
                        "capacity": self.capacity,
                        "distance": self.calc_distance_between_buildings(),
                        "build_cost": self.calc_build_cost(),
                        "upgrade_cost": self.calc_upgrade_cost(),
                    }
                ),
                default=str,
                indent=2,
            )
        )

    @property
    def id(self):
        return f"{self.building_1.id}-{self.building_2.id}"

    def calc_distance_between_buildings(self) -> float:
        return self.building_1.calc_distance_to_building(self.building_2)

    def calc_build_cost(self) -> float:
        # The cost is 1 resource for each 0.1km of tube installed, rounded down.
        return self.calc_distance_between_buildings() * 10  # TODO confirm??

    def calc_upgrade_cost(self):
        # The cost is 1 resource for each 0.1km of tube installed, rounded down.
        original_cost = self.calc_distance_between_buildings() * 10
        return original_cost * (self.capacity + 1)

    @classmethod
    def get_transport_line_by_buildings(
        cls,
        building_1: "Building",
        building_2: "Building",
    ) -> "TransportLine":

        for transport_line in TRANSPORT_LINES.values():
            building_ids = [transport_line.building_1.id, transport_line.building_2.id]
            if building_1.id in building_ids and building_2.id in building_ids:
                return transport_line
        return None

    @classmethod
    def get_transport_lines_unserved_by_pods(cls):
        all_buildings = set()
        all_buildings_with_pods = set()
        buildings_without_pods = []
        for transport_line in TRANSPORT_LINES.values():
            all_buildings.add(transport_line.building_1.id)
            all_buildings.add(transport_line.building_2.id)
        for pod in PODS.values():
            for building_id in pod.path:
                all_buildings_with_pods.add(building_id)
        return all_buildings - all_buildings_with_pods


@dataclass
class Pod:
    id: int
    num_stops: int
    path: List[int]

    def calc_cost(self):
        return 1000


@dataclass
class Building:
    id: int
    type: int
    coordinates: List[int]

    def calc_distance_to_building(self, other_building: "Building") -> float:
        return distance(
            self.coordinates,
            other_building.coordinates,
        )

    def filter_transport_lines(
        self, existing_transport_lines: List[TransportLine]
    ) -> TransportLine:
        this_transport_lines: List[TransportLine] = []
        for existing_transport_line in existing_transport_lines:
            if (
                existing_transport_line.building_1.id == self.id
                or existing_transport_line.building_2.id == self.id
            ):
                this_transport_lines.append(existing_transport_line)
        return existing_transport_line

    @classmethod
    def get_building_from_params(
        cls, building_id: int, buildings: Dict[int, "Building"]
    ):
        return buildings.get(building_id)

    @classmethod
    def get_building_pairs_by_priority(
        cls,
        # buildings: Dict[int, "Building"]
    ):
        # Get all unique combinations of 2 buildings
        building_pairs = list(itertools.combinations(BUILDINGS.values(), 2))

        supplemented_building_pairs = [
            dict(
                building_1=b1,
                building_2=b2,
                is_pad=(
                    isinstance(b1, BuildingLandingPad)
                    or isinstance(b2, BuildingLandingPad)
                ),
                distance_between=b1.calc_distance_to_building(b2),
                existing_transport_line=TransportLine.get_transport_line_by_buildings(
                    building_1=b1,
                    building_2=b2,
                ),
                can_connect_without_intersect=True,  # TODO implement this
            )
            for b1, b2 in building_pairs
        ]

        sorted_building_pairs = sorted(
            supplemented_building_pairs,
            key=lambda x: (
                not x[
                    "existing_transport_line"
                ],  # Sort with True as lower priority (deprioritized)
                x["is_pad"],  # Sort with True as higher priority (prioritized)
                x["distance_between"],  # Sort by distance as the final criterion
            ),
        )

        return sorted_building_pairs


@dataclass
class BuildingLandingPad(Building):
    num_astronauts: int
    astronaut_types: List[int]


@dataclass
class ProgramInput:
    num_resources: int
    num_travel_routes: int
    transport_lines: List[TransportLine]
    num_pods: int
    pods: List[Pod]
    num_new_buildings: int
    new_buildings: List[Building]

    def print_state(self):
        debug(message=json.dumps(asdict(self), default=str, indent=1))


class ActionType(str, Enum):
    TUBE = "TUBE"
    UPGRADE = "UPGRADE"
    TELEPORT = "TELEPORT"
    POD = "POD"
    DESTROY = "DESTROY"
    WAIT = "WAIT"


@dataclass
class Action:
    pass

    def __str__(self):
        # Start with the action_type, then append other attributes
        props = [f"{self.action_type.value}"]
        # Get all attributes except action_type, starting from the second one
        for field, value in vars(self).items():
            if field == "action_type":
                continue
            if isinstance(value, List):
                props.extend(str(v) for v in value)
            else:
                props.append(str(value))
        return " ".join(props)

    def calc_cost(self):
        raise NotImplemented


@dataclass
class ActionTube(Action):
    transport_line: TransportLine = None
    action_type: ActionType = ActionType.TUBE

    def __str__(self):
        props = [f"{self.action_type.value}"]
        props.append(str(self.transport_line.building_1.id))
        props.append(str(self.transport_line.building_2.id))
        return " ".join(props)

    def calc_cost(self):
        return self.transport_line.calc_build_cost()


@dataclass
class ActionUpgrade(Action):
    transport_line: TransportLine = None
    action_type: ActionType = ActionType.UPGRADE

    def __str__(self):
        props = [f"{self.action_type.value}"]
        props.append(str(self.transport_line.building_1.id))
        props.append(str(self.transport_line.building_2.id))
        return " ".join(props)

    def calc_cost(self):
        return self.transport_line.calc_upgrade_cost()


@dataclass
class ActionTeleport(Action):
    building_1: "Building"
    building_2: "Building"
    action_type: ActionType = ActionType.TELEPORT

    def calc_cost(self):
        return 5000


@dataclass
class ActionPod(Action):
    pod: Pod
    action_type: ActionType = ActionType.POD

    def __str__(self):
        props = [f"{self.action_type.value}"]
        props.append(str(self.pod.id))
        props.extend(str(v) for v in self.pod.path)
        return " ".join(props)

    def calc_cost(self):
        return 1000


@dataclass
class ActionDestroy(Action):
    pod: Pod
    action_type: ActionType = ActionType.DESTROY

    def calc_cost(self):
        return -750


@dataclass
class ActionWait(Action):
    action_type: ActionType = ActionType.WAIT

    def calc_cost(self):
        return 0


def output_actions(actions: List[Action]):
    action_str = ";".join([str(a) for a in actions])
    debug("action_str: " + str(action_str))
    print(action_str)


TRANSPORT_LINES: Dict[str, TransportLine] = {}
PODS: Dict[int, Pod] = {}
BUILDINGS: Dict[int, Building] = {}


# game loop
while True:
    resources = int(input())
    num_travel_routes = int(input())

    transport_line_data: List = []
    transport_lines: List[TransportLine] = []
    pods: List[Pod] = []
    new_buildings: List[Building] = []

    for i in range(num_travel_routes):
        building_id_1, building_id_2, capacity = [int(j) for j in input().split()]
        transport_line_data.append((building_id_1, building_id_2, capacity))
    num_pods = int(input())
    for i in range(num_pods):
        pod_properties = [int(j) for j in input().split()]
        pod = Pod(
            id=int(pod_properties[0]),
            num_stops=int(pod_properties[1]),
            path=[int(v) for v in pod_properties[2:]],
        )
        pods.append(pod)
        if not pod.id in PODS.keys():
            PODS[pod.id] = pod

    num_new_buildings = int(input())
    for i in range(num_new_buildings):
        building_properties = [int(j) for j in input().split()]
        if len(building_properties) > 4:
            new_buildings.append(
                BuildingLandingPad(
                    id=int(building_properties[0]),
                    type=int(building_properties[1]),
                    coordinates=[int(v) for v in building_properties[2:4]],
                    num_astronauts=int(building_properties[5]),
                    astronaut_types=building_properties[6:],
                )
            )
        else:
            new_buildings.append(
                Building(
                    id=int(building_properties[0]),
                    type=int(building_properties[1]),
                    coordinates=[int(v) for v in building_properties[2:4]],
                )
            )

    for new_building in new_buildings:
        BUILDINGS[new_building.id] = new_building

    #######
    # examine transport lines....
    #######

    if len(transport_line_data):
        for transport_line_d in transport_line_data:
            building_1 = Building.get_building_from_params(
                building_id=transport_line_d[0], buildings=BUILDINGS
            )
            building_2 = Building.get_building_from_params(
                building_id=transport_line_d[1], buildings=BUILDINGS
            )
            if building_1 and building_2:
                transport_line = TransportLine(
                    building_1=building_1,
                    building_2=building_2,
                    capacity=transport_line_d[2],
                )
                transport_lines.append(transport_line)
            if not transport_line.id in TRANSPORT_LINES.keys():
                TRANSPORT_LINES[transport_line.id] = transport_line
        debug(TRANSPORT_LINES)

    program_inputs = ProgramInput(
        num_resources=resources,
        num_travel_routes=num_travel_routes,
        transport_lines=transport_lines,
        num_pods=num_pods,
        pods=pods,
        num_new_buildings=num_new_buildings,
        new_buildings=new_buildings,
    )

    # program_inputs.print_state()

    remaining_resources = program_inputs.num_resources
    debug(f"Remaining resources: {remaining_resources}")
    actions: List[Action] = []

    ############
    # Create pods from tubes
    ############

    debug(BUILDINGS)
    building_pairs = Building.get_building_pairs_by_priority()
    for supplemented_building_pairs in building_pairs:
        debug(f"Remaining resources: {remaining_resources}")
        debug(
            f"Building pair: {json.dumps(supplemented_building_pairs, default=str, indent=2)}"
        )
        existing_transport_line = supplemented_building_pairs.get(
            "existing_transport_line"
        )
        if existing_transport_line:
            debug(f"TransportLine already exists: {str(existing_transport_line)}")
            upgrade_cost = existing_transport_line.calc_upgrade_cost()
            debug(upgrade_cost)
            # if upgrade_cost <= remaining_resources:
            #     actions.append(ActionUpgrade(transport_line=transport_line_to_build))
            #     remaining_resources -= upgrade_cost

            # else:
            #     debug("too expensive to upgrade transport line")
            continue

        debug("TransportLine doesn't exist...")

        transport_line_to_build = TransportLine(
            building_1=supplemented_building_pairs.get("building_1"),
            building_2=supplemented_building_pairs.get("building_2"),
        )
        debug(f"TransportLine doesn't exist, might create?")

        transport_line_to_build.dump()

        build_cost = transport_line_to_build.calc_build_cost()
        if build_cost <= remaining_resources:
            debug(transport_line_to_build)
            debug(transport_line_to_build.calc_build_cost())
            actions.append(ActionTube(transport_line=transport_line_to_build))
            remaining_resources -= build_cost
        else:
            debug("too expensive to build")

    buildings_without_pods = TransportLine.get_transport_lines_unserved_by_pods()
    debug(json.dumps(buildings_without_pods, default=str, indent=2))
    if len(BUILDINGS):
        unoptimized_paths = sorted(list(BUILDINGS.keys()))
        unoptimized_paths.extend([unoptimized_paths[0]])
        debug("unoptimized_paths")
        debug(unoptimized_paths)
        pod_with_all_paths = Pod(
            id=len(PODS) + 1,
            num_stops=len(unoptimized_paths),
            path=unoptimized_paths,
        )
        debug(json.dumps(str(pod_with_all_paths), default=str, indent=2))

        if pod_with_all_paths.calc_cost() <= remaining_resources:
            debug("Can create pod...r")
            actions.append(ActionPod(pod=pod_with_all_paths))
            PODS[pod_with_all_paths.id] = pod_with_all_paths
        else:
            debug("Too expensive to create pod...r")

    if not actions:
        actions.append(ActionWait())
    output_actions(actions=actions)
