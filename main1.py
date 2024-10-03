import json
import math
import sys
from dataclasses import asdict, dataclass
from enum import Enum
from typing import List

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.


# Function to calculate the Euclidean distance between two points
def distance(p1, p2):
    return math.sqrt((p2["x"] - p1["x"]) ** 2 + (p2["y"] - p1["y"]) ** 2)


# Function to check if point A is on the segment between B and C
def pointOnSegment(A, B, C):
    epsilon = 0.0000001
    return -epsilon < distance(B, A) + distance(A, C) - distance(B, C) < epsilon


import math


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


def debug(message: str):
    """ """
    print(
        message,
        file=sys.stderr,
        flush=False,
    )


@dataclass
class TransportLine:
    building_id_1: int
    building_id_2: int
    capacity: int


@dataclass
class Pod:
    id: int
    num_stops: int
    path: List[int]


@dataclass
class Building:
    id: int
    type: int
    coordinates: List[int]


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
    buildings: List[Building]

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
            if field != "action_type":
                props.append(str(value))
        return " ".join(props)


@dataclass
class ActionTube(Action):
    building_id_1: int
    building_id_2: int
    action_type: ActionType = ActionType.TUBE

    def calc_distance_between_buildings(self):
        return distance(self.building_id_1, self.building_id_2)

    def calc_cost(self):
        # The cost is 1 resource for each 0.1km of tube installed, rounded down.
        return self.calc_distance_between_buildings() / 10  # TODO confirm??


@dataclass
class ActionUpgrade(Action):
    building_id_1: int
    building_id_2: int
    action_type: ActionType = ActionType.UPGRADE


def output_actions(actions: List[Action]):
    debug("output_actions: " + str(actions))
    action_str = ";".join([str(a) for a in actions])
    debug("action_str: " + str(action_str))
    print(action_str)


# game loop
while True:
    resources = int(input())
    num_travel_routes = int(input())

    transport_lines: List[TransportLine] = []
    pods: List[Pod] = []
    buildings: List[Building] = []

    for i in range(num_travel_routes):
        building_id_1, building_id_2, capacity = [int(j) for j in input().split()]
        transport_lines.append(
            TransportLine(
                building_id_1=building_id_1,
                building_id_2=building_id_1,
                capacity=capacity,
            )
        )
    num_pods = int(input())
    for i in range(num_pods):
        pod_properties = [int(j) for j in input().split()]
        pods.append(
            Pod(
                id=int(pod_properties[0]),
                num_stops=int(pod_properties[1]),
                path=[int(v) for v in pod_properties[2:]],
            )
        )

    num_new_buildings = int(input())
    for i in range(num_new_buildings):
        building_properties = [int(j) for j in input().split()]
        if len(building_properties) > 4:
            buildings.append(
                BuildingLandingPad(
                    id=int(building_properties[0]),
                    type=int(building_properties[1]),
                    coordinates=[int(v) for v in building_properties[2:3]],
                    num_astronauts=int(building_properties[4]),
                    astronaut_types=[int(v) for v in building_properties[5:]],
                )
            )
        else:
            buildings.append(
                Building(
                    id=int(building_properties[0]),
                    type=int(building_properties[1]),
                    coordinates=[int(v) for v in building_properties[2:3]],
                )
            )
    program_inputs = ProgramInput(
        num_resources=resources,
        num_travel_routes=num_travel_routes,
        transport_lines=transport_lines,
        num_pods=num_pods,
        pods=pods,
        num_new_buildings=num_new_buildings,
        buildings=buildings,
    )

    program_inputs.print_state()

    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr, flush=True)

    # TUBE | UPGRADE | TELEPORT | POD | DESTROY | WAIT
    actions: List[Action] = []
    actions.append(ActionTube(building_id_1=0, building_id_2=1))
    output_actions(actions=actions)
    # print("TUBE 0 1;TUBE 0 2;POD 42 0 1 0 2 0 1 0 2")
