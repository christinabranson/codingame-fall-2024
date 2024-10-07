import itertools
import json
import math
import random
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

POD_COST = 1000
MAX_TRANSPORT_LINE_COUNT = 4


def debug(message: any):
    """ """
    print(
        message,
        file=sys.stderr,
        flush=False,
    )


# Function to calculate the Euclidean distance between two points
def distance(p1: Tuple[int, int], p2: Tuple[int, int]):
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
    prod = (p3[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
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


def detect_anomalous_keys(counts, z_threshold=4):
    """
    Detect keys with anomalously high values in a dictionary based on z-score.

    :param counts: defaultdict or dict with keys and their respective counts.
    :param z_threshold: Threshold for z-score to consider a value anomalous (default is 2).
    :return: List of keys that have anomalously high values.
    """
    # Extract the values from the dictionary
    values = list(counts.values())
    if not len(values):
        return []

    try:

        # Step 1: Calculate the mean
        mean = sum(values) / len(values)

        # Step 2: Calculate the standard deviation manually
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = variance**0.5

        # Step 3: Identify keys with counts that are anomalously high
        anomalous_keys = []
        for key, value in counts.items():
            z_score = (value - mean) / std_dev
            if z_score > z_threshold:
                anomalous_keys.append(key)

        return anomalous_keys
    except Exception as exc:
        debug(exc)
        return []


def subtract_arrays(array1, array2):
    """
    Subtracts elements of array2 from array1 element-wise.

    :param array1: The first list of numbers.
    :param array2: The second list of numbers.
    :return: A list where each element is the result of subtracting corresponding elements of array2 from array1.
    """
    return [item for item in array1 if item not in array2]


@dataclass(frozen=True)
class TransportLine:
    building_1: "Building"
    building_2: "Building"
    capacity: int = 1

    @property
    def id(self):
        building_ids = [
            self.building_1.id,
            self.building_2.id,
        ]
        building_ids.sort()
        return ";".join([str(bid) for bid in building_ids])

    def __str__(self):
        return f"Building 1: {self.building_1.id} | Building 2: {self.building_2.id}"

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
    def build_cost(self):
        return self.calc_build_cost()

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

        for transport_line in TRANSPORT_LINES.values():
            # debug(transport_line)
            all_buildings.add(transport_line.building_1.id)
            all_buildings.add(transport_line.building_2.id)

        for pod in PODS.values():
            for building_id in pod.path:
                all_buildings_with_pods.add(building_id)

        # debug("all_buildings")
        # debug(all_buildings)
        # debug("all_buildings_with_pods")
        # debug(all_buildings_with_pods)

        return all_buildings - all_buildings_with_pods

    @classmethod
    def get_connected_buildings(
        cls, adjacency_list: Dict["Building", List["Building"]], building_id
    ) -> List["Building"]:
        # Use adjacency list to get connected buildings in O(1)
        return adjacency_list.get(building_id, [])

    @classmethod
    def build_adjacency_list(cls) -> Dict["Building", List["Building"]]:
        # Build an adjacency list to avoid recomputing neighbors
        adjacency_list = {}
        # debug(type(TRANSPORT_LINES))
        # debug(type(TRANSPORT_LINES.values()))
        for transport_line in list(TRANSPORT_LINES.values()):
            # debug(transport_line)
            # debug(transport_line.id)
            if not isinstance(transport_line, TransportLine):
                debug(transport_line)
                assert False
            if transport_line.building_1.id not in adjacency_list:
                adjacency_list[transport_line.building_1.id] = []
            if transport_line.building_2.id not in adjacency_list:
                adjacency_list[transport_line.building_2.id] = []

            # Add bidirectional connections
            adjacency_list[transport_line.building_1.id].append(
                transport_line.building_2
            )
            adjacency_list[transport_line.building_2.id].append(
                transport_line.building_1
            )

        return adjacency_list

    @classmethod
    def generate_paths_from_building(
        cls,
        start_building: "Building",
        adjacency_list,
        min_unique_nodes: int = 2,
        max_depth: int = 4,
    ) -> List[List["Building"]]:
        debug("start_building")
        debug(start_building)
        # adjacency_list = cls.build_adjacency_list()  # Precompute neighbors
        stack = [
            (start_building.id, [start_building.id], set([start_building.id]))
        ]  # Track the path and visited set
        # current_building = start_building.id
        # current_path = [start_building.id]
        # visited = set()
        # visited.add(start_building.id)

        paths = []

        while stack:
            current_building, current_path, visited = stack.pop()

            # If we returned to the start and the path has more than 1 node, add to result
            if current_building == start_building and len(current_path) > 1:
                if len(visited) >= min_unique_nodes:
                    paths.append(current_path[:])

            # Limit maximum depth to avoid long paths (optional)
            if max_depth is not None and len(current_path) > max_depth:
                continue

            # Explore neighbors
            for neighbor in cls.get_connected_buildings(
                adjacency_list, current_building
            ):
                # Allow revisits only if needed
                # new_path = current_path + [neighbor]
                new_visited = visited.copy()
                new_visited.add(neighbor.id)

                # current_building = neighbor.id
                # current_path = current_path + [neighbor.id]
                # visited.add(neighbor.id)

                stack.append((neighbor.id, current_path + [neighbor.id], new_visited))

        return paths

    @classmethod
    def get_transport_lines_prioritized_least_connecions(cls) -> List["TransportLine"]:
        debug("get_transport_lines_prioritized_least_connecions")
        copy_transport_lines = deepcopy(list(TRANSPORT_LINES.values()))
        supp_transport_lines = []
        for tl in copy_transport_lines:

            supp_transport_lines.append(
                (
                    tl,
                    min(
                        tl.building_1.transport_line_count,
                        tl.building_2.transport_line_count,
                    ),
                )
            )

        ordered = sorted(supp_transport_lines, key=lambda x: x[1], reverse=False)
        # debug("ordered")
        # debug(json.dumps(ordered, indent=2, default=str))

        return [o[0] for o in ordered]



class PotentialTransportLine(TransportLine):
    pass

    @staticmethod
    def create_valid_transport_line(
        building_1: "Building",
        building_2: "Building",
        remaining_resources: float,
        limit_types: List[int],
    ) -> Optional["PotentialTransportLine"]:

        if limit_types:
            if building_1.type in limit_types:
                debug("Building 1 is in limit types..")
                return None

            if building_2.type in limit_types:
                debug("Building 2 is in limit types..")
                return None

        if building_1 == building_2:
            return None

        pot_line = PotentialTransportLine(
            building_1=building_1,
            building_2=building_2,
        )
        build_cost = pot_line.build_cost
        if build_cost > remaining_resources:
            debug(
                f"PotentialTransportLine invalid due cost {build_cost}/{remaining_resources} "
            )
            return None

        if not pot_line.is_valid():
            return None
        return pot_line

    def is_valid(self):

        if self._does_it_already_exist():
            debug("PotentialTransportLine already exists!")
            return False

        if self._buildings_have_too_many_connections():
            debug(
                "PotentialTransportLine contains at least 1 building which is already connected"
            )
            return False

        if self._does_it_intersect():
            debug("PotentialTransportLine intersects another line")
            return False
        
        if self._does_it_intersect_a_building():
            debug("PotentialTransportLine intersects another building")
            return False

        return True

    def _does_it_already_exist(self):
        return self.id in TRANSPORT_LINES.keys()

    def _buildings_have_too_many_connections(self):
        if self.building_1.transport_line_count >= MAX_TRANSPORT_LINE_COUNT:
            return True

        if self.building_2.transport_line_count >= MAX_TRANSPORT_LINE_COUNT:
            return True

        return False

    def _does_it_intersect(self):
        for transport_line in TRANSPORT_LINES.values():
            if segmentsIntersect(
                A=self.building_1.coordinates,
                B=self.building_2.coordinates,
                C=transport_line.building_1.coordinates,
                D=transport_line.building_2.coordinates,
            ):
                return True
        return False
    
    def _does_it_intersect_a_building(self):
        return False
        for building in BUILDINGS.get_buildings_as_list():
            if pointOnSegment(
                A=building.coordinates,
                B=self.building_1.coordinates,
                C=self.building_2.coordinates,
            ):
                return True
        return False


@dataclass
class Pod:
    id: int
    num_stops: int
    path: List[int]

    def calc_cost(self):
        return POD_COST


@dataclass
class PotentialPod(Pod):
    pass


@dataclass(frozen=True)
class Building:
    id: int
    type: int
    coordinates: Tuple[int, int]

    def __str__(self):
        return f"ID: {self.id} | Coords: {str(self.coordinates)} | Type: {self.type}"

    def dump(self):
        debug(
            json.dumps(
                str(
                    {
                        "type": "Building",
                        "id": self.id,
                        "type": self.type,
                        "coordinates": self.coordinates,
                    }
                ),
                default=str,
                indent=2,
            )
        )

    @property
    def transport_line_count(self):
        return len(
            [
                tl
                for tl in list(TRANSPORT_LINES.values())
                if tl.building_1 == self or tl.building_2 == self
            ]
        )

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

        # prioritize connecting landing pads to non-landing pads by distance
        supplemented_building_pairs = [
            dict(
                building_1=b1,
                building_2=b2,
                one_is_pad=(
                    isinstance(b1, BuildingLandingPad)
                    ^ isinstance(b2, BuildingLandingPad)
                ),
                distance_between=b1.calc_distance_to_building(b2),
                has_existing_transport_line=TransportLine.get_transport_line_by_buildings(
                    building_1=b1,
                    building_2=b2,
                )
                is not None,
                existing_transport_line=TransportLine.get_transport_line_by_buildings(
                    building_1=b1,
                    building_2=b2,
                ),
            )
            for b1, b2 in building_pairs
        ]

        sorted_building_pairs = sorted(
            supplemented_building_pairs,
            key=lambda x: (
                not x["one_is_pad"],  # Sort with True as higher priority (prioritized)
                x[
                    "has_existing_transport_line"
                ],  # Sort with True as lower priority (deprioritized)
                x["distance_between"],  # Sort by distance as the final criterion
            ),
        )

        debug(json.dumps(sorted_building_pairs[0:10], default=str, indent=5))

        return sorted_building_pairs

    @classmethod
    def find_priority_building_pairs(
        cls,
        ensure_one_is_landing=True,
        ensure_same_type=False,
    ) -> List[Tuple["Building", "Building", float]]:

        if ensure_same_type:
            # Group buildings by type
            buildings_by_type = {}

            buildings = BUILDINGS.values()

            for building in buildings:
                building_type = (
                    building.type if building.type > 0 else building.prominent_type
                )
                if building_type not in buildings_by_type:
                    buildings_by_type[building_type] = []
                buildings_by_type[building_type].append(building)

            # debug(json.dumps(buildings_by_type, default=str, indent=1))
        else:
            buildings_by_type = {"dont_care": []}
            for building in buildings:
                buildings_by_type["dont_care"].append(building)

        result_pairs = []

        # For each type, find pairs and calculate distances
        for building_type, building_list in buildings_by_type.items():
            for i in range(len(building_list)):
                for j in range(i + 1, len(building_list)):

                    building1 = building_list[i]
                    building2 = building_list[j]

                    if ensure_one_is_landing:
                        if not (
                            isinstance(building1, BuildingLandingPad)
                            ^ isinstance(building2, BuildingLandingPad)
                        ):
                            # debug("at least one must be a BuildingLandingPad, skipping")
                            continue

                    dist = distance(building1.coordinates, building2.coordinates)
                    result_pairs.append((building1, building2, dist))

        # Sort pairs by distance (increasing)
        result_pairs.sort(key=lambda x: x[2])

        return result_pairs


@dataclass(frozen=True)
class BuildingLandingPad(Building):
    num_astronauts: int
    astronaut_types: List[int]

    def __str__(self):
        str = super().__str__()
        str += f" | Astr Type: {self.prominent_type}"
        return str

    def dump(self):
        debug(
            json.dumps(
                str(
                    {
                        "type": "BuildingLandingPad",
                        "id": self.id,
                        "type": self.type,
                        "coordinates": self.coordinates,
                        "prominent_type": self.prominent_type,
                        "astronaut_types": self.astronaut_types,
                    }
                ),
                default=str,
                indent=2,
            )
        )

    @property
    def prominent_type(self):
        return Counter(self.astronaut_types).most_common(1)[0][
            0
        ]  # Return the most common integer


import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BuildingDistanceMatrix:
    buildings: Dict[int, Building] = field(default_factory=dict)
    distance_matrix: List[List[float]] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.distance_matrix = self.create_distance_matrix()

    def get_buildings_as_list(self, filter_type: int = None) -> List[Building]:
        buildings_list = list(self.buildings.values())

        if filter_type is not None:
            buildings_list = [
                building for building in buildings_list if building.type == filter_type
            ]

        return buildings_list

    def create_distance_matrix(self) -> List[List[float]]:
        building_list = list(self.buildings.values())
        n = len(building_list)
        matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = distance(building_list[i].coordinates, building_list[j].coordinates)
                matrix[i][j] = matrix[j][i] = d
        return matrix

    def add_building(self, new_building: Building):
        # Add new building to the dictionary
        self.buildings[new_building.id] = new_building
        building_list = list(self.buildings.values())
        n = len(building_list)

        # Update the distance matrix
        for i in range(n - 1):
            d = distance(building_list[i].coordinates, new_building.coordinates)
            self.distance_matrix[i].append(d)
        # Add a new row for the new building
        new_row = [0] * n
        self.distance_matrix.append(new_row)
        for i in range(n - 1):
            self.distance_matrix[-1][i] = self.distance_matrix[i][-1]

    def get_neighbors(
        self,
        building_id: int,
        N: int,
        building_types: Optional[List[int]] = None,
        exclude_list: Optional[List[Building]] = None,
    ) -> List[Building]:
        if building_id not in self.buildings:
            return []

        building_idx = list(self.buildings).index(building_id)
        building_list = list(self.buildings.values())

        # if exclude_list:
        #     building_list = [b for b in building_list if b not in exclude_list]

        # if building_types and len(building_types) > 0:
        #     building_list = [b for b in building_list if b.type in building_types]

        # debug("building_list")
        # debug(building_list)

        # Get distances from the building to others and sort by distance
        distances = [
            (i, dist)
            for i, dist in enumerate(self.distance_matrix[building_idx])
            if i != building_idx
        ]
        distances.sort(key=lambda x: x[1])

        # debug("distances")
        # debug(distances)

        # Filter buildings by type if specified
        if building_types is not None:
            # debug(building_types)
            distances = [
                (i, dist)
                for i, dist in distances
                if building_list[i].type in building_types
            ]
            # building_list = [b for b in building_list if b.type in building_types]

        if exclude_list is not None:
            # debug(exclude_list)
            distances = [
                (i, dist)
                for i, dist in distances
                if building_list[i] not in exclude_list
            ]
            # building_list = [b for b in building_list if b.type in building_types]

        try:
            # Return the N closest buildings that match the type
            closest_buildings = [building_list[i] for i, _ in distances[:N]]
            # closest_buildings = []
            # for i, _ in distances[:N]:
            #     debug(i)
            #     debug(_)
            #     debug(building_list[i])
            return closest_buildings
        except Exception as exc:
            debug(exc)
            debug(json.dumps(building_list, default=str, indent=2))
            debug(json.dumps(distances, default=str, indent=2))
            raise


@dataclass
class Path:
    nodes: List[Building]


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


def build_bidirectional_path_from_transport_line(transport_lines: List[TransportLine]):
    pass


def find_paths(
    transport_lines: List[TransportLine],
    adjacency_list: Dict[int, List[Building]],
    mirror: bool = False,
    max_depth: int = 10,
) -> List[List[Building]]:
    """Find circular or mirrored paths from transport lines with repeated visits and max depth."""
    if not adjacency_list.keys():
        debug("no adjacency_list provided")
        return []

    paths = []

    def dfs(current: Building, path: List[Building], start: Building, depth: int):
        if depth > max_depth:
            return  # Stop if max depth is exceeded

        if len(path) > 1 and current == start:
            # We found a circular path, add it to the list of paths
            paths.append(path[:])
            return

        for neighbor in adjacency_list[current.id]:
            if depth < max_depth:  # Continue if the depth limit has not been reached
                path.append(neighbor)
                dfs(neighbor, path, start, depth + 1)
                path.pop()

    # For each building, try to find circular or mirrored paths
    for line in transport_lines:
        # debug(line)
        b1, b2 = line.building_1, line.building_2

        # Circular path starting from building_1
        dfs(b1, [b1], b1, 1)

    # debug(json.dumps(paths, default=str, indent=1))

    if mirror:
        # Create mirrored paths
        mirrored_paths = []
        for path in paths:
            mirrored_path = (
                path + path[-2::-1]
            )  # Mirror the path by excluding the last point and reversing
            mirrored_paths.append(mirrored_path)
        return mirrored_paths

    return paths


def build_bidirectional_path(buildings):
    # Step 1: Create an adjacency list for bidirectional connections
    from collections import defaultdict

    adjacency_list = defaultdict(list)
    for building1, building2 in buildings:
        adjacency_list[building1].append(building2)
        adjacency_list[building2].append(building1)

    # Step 2: Start from any building, say building 0 or first in the input, and traverse the graph
    start_building = buildings[0][0]
    path = []
    visited = set()

    def traverse(building):
        path.append(building)
        visited.add(building)

        # Visit all connected buildings that have not yet been visited
        for neighbor in adjacency_list[building]:
            if neighbor not in visited:
                traverse(neighbor)
                path.append(
                    building
                )  # Backtrack to the current building after visiting a neighbor

    traverse(start_building)

    # debug(path)

    return path


TRANSPORT_LINES: Dict[str, TransportLine] = {}
PODS: Dict[int, Pod] = {}
BUILDINGS = BuildingDistanceMatrix()


# game loop
while True:

    # Reset transport lines and pods
    TRANSPORT_LINES = {}
    PODS = {}

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
        # debug(building_properties)
        if building_properties[0] == 0:
            new_buildings.append(
                BuildingLandingPad(
                    id=int(building_properties[1]),
                    type=int(building_properties[0]),
                    coordinates=[int(v) for v in building_properties[2:4]],
                    num_astronauts=int(building_properties[5]),
                    astronaut_types=building_properties[6:],
                )
            )
        else:
            new_buildings.append(
                Building(
                    id=int(building_properties[1]),
                    type=int(building_properties[0]),
                    coordinates=[int(v) for v in building_properties[2:4]],
                )
            )

    for new_building in new_buildings:
        BUILDINGS.add_building(new_building=new_building)

    # debug(json.dumps(BUILDINGS.buildings, default=str, indent=2))
    # raise

    #######
    # examine transport lines....
    #######

    if len(transport_line_data):
        for transport_line_d in transport_line_data:
            building_1 = BUILDINGS.buildings.get(transport_line_d[0])
            building_2 = BUILDINGS.buildings.get(transport_line_d[1])
            if building_1 and building_2:
                transport_line = TransportLine(
                    building_1=building_1,
                    building_2=building_2,
                    capacity=transport_line_d[2],
                )
                transport_lines.append(transport_line)
            if not transport_line.id in TRANSPORT_LINES.keys():
                TRANSPORT_LINES[transport_line.id] = transport_line
        # debug(TRANSPORT_LINES)

    limit_types = []
    building_type_count = defaultdict(int)
    for b in BUILDINGS.get_buildings_as_list():
        if b.type != 0:
            building_type_count[b.type] += 1
    anomalous_building_occ = detect_anomalous_keys(building_type_count)
    limit_types.extend(anomalous_building_occ)
    connected_types = defaultdict(int)
    for tl in TRANSPORT_LINES.values():
        connected_types[tl.building_1.type] += 1
        connected_types[tl.building_2.type] += 1
    limited_types_due_to_connections = detect_anomalous_keys(connected_types)
    limit_types.extend(limited_types_due_to_connections)
    limit_types = list(set(limit_types))

    if len(limit_types):
        debug(f"limiting the following types: {str(limit_types)}")

    program_inputs = ProgramInput(
        num_resources=resources,
        num_travel_routes=num_travel_routes,
        transport_lines=transport_lines,
        num_pods=num_pods,
        pods=pods,
        num_new_buildings=num_new_buildings,
        new_buildings=new_buildings,
    )

    # debug("INPUT STATE.....")
    # program_inputs.print_state()

    remaining_resources = program_inputs.num_resources
    debug(f"Remaining resources: {remaining_resources}")
    actions: List[Action] = []
    building_landing_pads = BUILDINGS.get_buildings_as_list(filter_type=0)
    # debug(json.dumps(TRANSPORT_LINES, default=str, indent=1))

    building_adjacency_list = TransportLine.build_adjacency_list()

    ############
    # POD MANAGEMENT
    ############

    debug("CREATING PODS....")
    debug(f"Current # of pods: {len(PODS)}")

    max_pods_created_per_turn = 2
    created_pods = 0
    if True and POD_COST <= remaining_resources and len(TRANSPORT_LINES.values()):
        debug("In creating pods section...")

        paths = find_paths(
            transport_lines=list(TRANSPORT_LINES.values()),
            adjacency_list=building_adjacency_list,
            mirror=True,
        )

        def sort_by_unique_items(list_of_lists):
            """
            Sorts a list of lists by the number of unique items in each sublist.

            :param list_of_lists: List of lists to be sorted.
            :return: A sorted list of lists based on the number of unique items in each sublist.
            """
            return sorted(
                list_of_lists,
                key=lambda sublist: len(set([b.id for b in sublist])),
                reverse=True,
            )

        paths = sort_by_unique_items(paths)

        # debug("paths....")
        # debug(json.dumps(paths, default=str, indent=1))
        # raise

        desired_path_length = 2
        for path in paths:
            path_ids = [b.id for b in path]
            # debug(f"path length: {len(set(path_ids)) }")
            if len(set(path_ids)) > desired_path_length:
                desired_path_length = len(set(path_ids))
                break

        debug(f"desired_path_length: {desired_path_length}")

        for path in paths:
            path_ids = [b.id for b in path]
            if len(set(path_ids)) < desired_path_length:
                # debug("path too short, continuing....")
                break
            debug(f"evaluating path {str(path_ids)}")
            if POD_COST <= remaining_resources:
                pod = Pod(
                    id=len(PODS) + 1,
                    num_stops=len(path_ids),
                    path=path_ids,
                )
                debug(f"Creating pod: {pod}")
                actions.append(ActionPod(pod=pod))
                PODS[pod.id] = pod
                remaining_resources = remaining_resources - POD_COST
                created_pods += 1
            else:
                debug("Too expensive to create pod...")
                break

            if remaining_resources <= POD_COST:
                debug("remaining resources <= POD_COST, breaking...")
                break

            if created_pods > max_pods_created_per_turn:
                debug("too many pods created this turn, breaking...")
                break

    debug("/CREATING PODS....")

    ############
    # TUBE MANAGEMENT
    ############

    debug("CREATING TUBES BETWEEN BUILDINGS....")
    debug(f"Current # of transport lines: {len(TRANSPORT_LINES.values())}")
    building_type_filters = [e for e in list(range(0, 19)) if e > 0]
    building_type_filters = subtract_arrays(building_type_filters, limit_types)
    if new_buildings:
        is_first_round = len(TRANSPORT_LINES.values()) == 0
        max_lines_created_per_turn = 10
        created_lines = 0

        building_landing_pads = [
            b
            for b in building_landing_pads
            if b.transport_line_count < MAX_TRANSPORT_LINE_COUNT
        ]
        # debug("building_landing_pads/!")
        # debug(json.dumps(building_landing_pads,default=str,indent=1))

        for building_landing_pad in random.sample(
            building_landing_pads, min(len(building_landing_pads), 20)
        ):
            # debug("list(set(building_landing_pad.astronaut_types))")
            # debug(list(set(building_landing_pad.astronaut_types)))

            if is_first_round:
                building_type_filters = list(set(building_landing_pad.astronaut_types))

            nearest_buildings = BUILDINGS.get_neighbors(
                building_id=building_landing_pad.id,
                N=5,
                building_types=building_type_filters,
            )

            for nearest_building in nearest_buildings:
                # debug(json.dumps(nearest_buildings, default=str, indent=2))
                pot_line = PotentialTransportLine.create_valid_transport_line(
                    building_1=building_landing_pad,
                    building_2=nearest_building,
                    remaining_resources=remaining_resources,
                    limit_types=limit_types,
                )
                if pot_line:
                    actions.append(ActionTube(transport_line=pot_line))
                    remaining_resources = remaining_resources - pot_line.build_cost
                    TRANSPORT_LINES[pot_line.id] = pot_line
                    created_lines += 1

                if remaining_resources <= POD_COST:
                    debug("remaining resources <= POD_COST, breaking...")
                    break

                if created_lines >= max_lines_created_per_turn:
                    debug("too many lines created")
                    break

    debug("/CREATING TUBES BETWEEN BUILDINGS....")
    debug("EXTENDING TUBES BETWEEN BUILDINGS....")

    debug(f"Current # of transport lines: {len(TRANSPORT_LINES.values())}")
    debug(f"remaining_resources: {remaining_resources}")

    max_lines_extended_per_turn = 10
    extended_lines = 0

    building_adjacency_list = TransportLine.build_adjacency_list()

    # if len(TRANSPORT_LINES.values()) and remaining_resources <= POD_COST:
    if len(TRANSPORT_LINES.values()):
        # copy_transport_lines = deepcopy(list(TRANSPORT_LINES.values()))

        # if limit_types:
        #     copy_transport_lines = [
        #         tl
        #         for tl in copy_transport_lines
        #         if tl.building_1.type not in limit_types
        #         and tl.building_2.type not in limit_types
        #     ]

        # for transport_line in random.sample(
        #     copy_transport_lines, min(len(copy_transport_lines), 100)
        # ):
        for (
            transport_line
        ) in TransportLine.get_transport_lines_prioritized_least_connecions():
            building_1 = transport_line.building_1
            building_2 = transport_line.building_2

            if building_1.type == 0 and building_2.type == 0:
                continue

            if (
                building_1.transport_line_count > MAX_TRANSPORT_LINE_COUNT
                or building_2.transport_line_count > MAX_TRANSPORT_LINE_COUNT
            ):
                continue

            debug(f"Trying to extend line {str(transport_line)}")

            debug("looking to extend lines for building_2")

            building_2_types_to_filter = None
            if len(limit_types):
                building_2_types_to_filter = subtract_arrays(
                    connected_types.keys(), limit_types
                )

            if building_2.type == 0:
                building_2_types_to_filter = [0]

            debug("building_2_types_to_filter")
            debug(building_2_types_to_filter)

            building_2_neighbors = BUILDINGS.get_neighbors(
                building_id=building_2.id,
                N=1 if building_2.type == 0 else 2,
                building_types=building_2_types_to_filter,
                exclude_list=TransportLine.get_connected_buildings(
                    building_id=building_2.id,
                    adjacency_list=building_adjacency_list,
                ),
            )

            for building_2_neighbor in building_2_neighbors:
                debug(f"building 2 has a neighbor: {building_2_neighbor.id}")

                pot_line = PotentialTransportLine.create_valid_transport_line(
                    building_1=building_2,
                    building_2=building_2_neighbor,
                    remaining_resources=remaining_resources,
                    limit_types=limit_types,
                )
                if pot_line:
                    debug(pot_line)
                    actions.append(ActionTube(transport_line=pot_line))
                    remaining_resources = remaining_resources - pot_line.build_cost
                    TRANSPORT_LINES[pot_line.id] = pot_line
                    extended_lines += 1
                    building_adjacency_list = TransportLine.build_adjacency_list()

            # if remaining_resources <= POD_COST:
            #     debug("remaining resources <= POD_COST, breaking...")
            #     break

            debug("looking to extend lines for building_1")
            building_1_types_to_filter = None
            if len(limit_types):
                building_1_types_to_filter = subtract_arrays(
                    connected_types.keys(), limit_types
                )

            if building_1.type == 0:
                building_1_types_to_filter = [0]

            debug("building_1_types_to_filter")
            debug(building_1_types_to_filter)

            building_1_neighbors = BUILDINGS.get_neighbors(
                building_id=building_1.id,
                N=1 if building_1.type == 0 else 2,
                building_types=building_1_types_to_filter,
                exclude_list=TransportLine.get_connected_buildings(
                    building_id=building_1.id,
                    adjacency_list=building_adjacency_list,
                ),
            )

            for building_1_neighbor in building_1_neighbors:
                debug(f"building 1 has a neighbor: {building_1_neighbor.id}")
                pot_line = PotentialTransportLine.create_valid_transport_line(
                    building_1=building_1,
                    building_2=building_1_neighbor,
                    remaining_resources=remaining_resources,
                    limit_types=limit_types,
                )
                if pot_line:
                    actions.append(ActionTube(transport_line=pot_line))
                    remaining_resources = remaining_resources - pot_line.build_cost
                    TRANSPORT_LINES[pot_line.id] = pot_line
                    extended_lines += 1
                    building_adjacency_list = TransportLine.build_adjacency_list()

            # if remaining_resources <= POD_COST:
            #     debug("remaining resources <= POD_COST, breaking...")
            #     break

            if extended_lines >= max_lines_extended_per_turn:
                debug("too many lines extended")
                break

        debug("/EXTENDING TUBES BETWEEN BUILDINGS....")

    if not actions:
        actions.append(ActionWait())
    output_actions(actions=actions)
