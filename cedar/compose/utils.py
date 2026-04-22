# import networkx as nx
# import matplotlib.pyplot as plt
import copy
from collections import deque

from typing import Dict, List, Set, Tuple, Union

from cedar.pipes import Pipe


def traverse_feature_graph(
    pipe: Pipe,
) -> Tuple[Dict[int, Pipe], Dict[int, Set[int]]]:
    """
    Traverse the feature, returning a Dictionary mapping each id(Pipe)
    to the Pipe for all pipes in the feature.

    Args:
        pipe: Pipe representing the output node of the feature

    Returns:
        Tuple(
            Dict of all Pipes in the feature, mapping the Python id() of
                each Pipe to the pipe itself,
            A dict representing the adjacency list of the
                *directed* edges of each Pipe
        )
    """
    if not pipe:
        raise RuntimeError("Feature does not have an applied source.")

    _assign_pipe_ids(pipe, 0)

    res: Dict[int, Pipe] = {}
    adj: Dict[int, Set[int]] = {}
    adj[pipe.id] = set()  # no outgoing edges for output node
    _dfs_helper(res, adj, pipe)

    return (res, adj)


def _assign_pipe_ids(p: Pipe, next_id: int):
    if p.id is None:
        p.id = next_id
        next_id += 1

    for in_p in p.input_pipes:
        if in_p.id is None:
            next_id = _assign_pipe_ids(in_p, next_id)

    return next_id


def _dfs_helper(d: Dict[int, Pipe], a: Dict[int, Set[int]], p: Pipe):
    d[p.id] = p

    for in_p in p.input_pipes:
        if in_p.id not in a:
            a[in_p.id] = set()
        # track only outgoing edges
        a[in_p.id].add(p.id)
        if in_p.id not in d:
            _dfs_helper(d, a, in_p)


def get_sources(pipes: Dict[int, Pipe]) -> List[int]:
    """
    Given a dict of pipe IDs to the pipe object, returns a list of all
    source pipe IDs (i.e., ones with no inputs).
    """
    return [k for k, v in pipes.items() if v.is_source()]


def viz_graph(
    pipes: Dict[int, Pipe],
    graph: Dict[int, Set[int]],
    path: str,
    logical: bool = True,
):
    """
    Visualize graph to an image.

    Args:
        pipes: Dict mapping id of each pipe to pipe itself
        graph: Dict mapping id of each pipe to its directed adjacency list
        path: Path to save image
        logical: True to viz the logical graph, false for physical
    """
    raise NotImplementedError
    # G = nx.DiGraph()

    # for p_id, pipe in pipes.items():
    #     edges = graph[p_id]
    #     for e_id in edges:
    #         p_name = (
    #             pipe.get_logical_uname()
    #             if logical
    #             else pipe.get_physical_uname()
    #         )
    #         e_name = (
    #             pipes[e_id].get_logical_uname()
    #             if logical
    #             else pipes[e_id].get_physical_uname()
    #         )
    #         G.add_edge(p_name, e_name)

    # # pos = nx.spring_layout(G)
    # pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    # plt.figure(figsize=(10, 10))
    # nx.draw_networkx(G, pos, with_labels=True, arrows=True)

    # plt.axis("off")  # Turn off the axes
    # plt.savefig(path)
    # plt.close()


def bfs_order(pipes: Dict[int, Pipe], graph: Dict[int, Set[int]]):
    """
    Returns a list of Pipe IDs obtained from traversing the
    graph in BFS order.
    """
    sources = get_sources(pipes)
    if len(sources) == 0:
        raise RuntimeError("Could not extract a source pipe.")

    if len(sources) > 1:
        raise NotImplementedError("Multiple sources")

    visited = set()
    queue = deque([sources[0]])

    order = []

    while queue:
        v = queue.popleft()
        if v not in visited:
            visited.add(v)
            order.append(v)

            for neighbor in graph[v]:
                if neighbor not in visited:
                    queue.append(neighbor)
                else:
                    raise NotImplementedError("Incast nodes")

    return order


def topological_sort(graph: Dict[int, Set[int]]) -> List[int]:
    """
    Returns a list of Pipe IDs representing a topological sort of the graph.

    Args:
        graph: Dict representing the adjacency list of the graph.
            Edges specified as outgoing, directed edges.
            For example, {"A": ("B", "C")} represents an
            outgoing edge to A->B and A->C
    """
    visited = set()
    marked = set()

    sorted_l = []

    def _visit(n: int):
        if n in visited:
            return
        if n in marked:
            raise RuntimeError("Detected cycle!")

        marked.add(n)

        for m in graph[n]:
            _visit(m)

        marked.remove(n)
        visited.add(n)
        sorted_l.append(n)

    for pipe in graph:
        _visit(pipe)

    sorted_l.reverse()
    return sorted_l


def flip_adj_list(input_adj_list: Dict[int, Union[List[int], Set[int]]]):
    """
    Returns an adjacency list representing outgoing/incoming edges,
    given an adjacency list of incoming/outgoing edges
    """
    output_adj_list = {}
    for n in input_adj_list:
        output_adj_list[n] = set()

    for n, neighbors in input_adj_list.items():
        for m in neighbors:
            if m in output_adj_list:
                output_adj_list[m].add(n)
            else:
                raise RuntimeError(f"Found node {m} not specified in graph")

    return output_adj_list


def derive_constraint_graph(pipes: Dict[int, Pipe]) -> Dict[int, Set[int]]:
    """
    Given all logical pipes, returns an adjacency list represnting the
    specified constraint graph over the pipes.

    Args:
        pipes: Dict mapping pipe ID to pipe

    Returns:
        An adjacency list detailing a (potentially unconnected) graph
        of dependency constraints

    NOTE: Does not specify "fixed" pipes, only depends_on relationships
    """
    d = {}
    tag_to_p_id = {}

    # Extract tags from all pipes
    for p_id, pipe in pipes.items():
        tag = pipe.tag
        # Assign default tag
        if tag is None:
            tag = pipe.get_logical_uname()
            pipe.tag = tag

        d[p_id] = set()

        if tag in tag_to_p_id:
            raise NotImplementedError("Tags must be unique: {}".format(tag))
        tag_to_p_id[tag] = p_id

    for p_id, pipe in pipes.items():
        depends_on = pipe._depends_on_tags

        if depends_on is not None:
            for tag in depends_on:
                # Create an edge from tag -> this
                d[tag_to_p_id[tag]].add(p_id)

    if has_cycle(d):
        raise ValueError("Detected cycle in constraint graph!")

    return d


def has_cycle(graph: Dict[int, Set[int]]):
    """
    Returns true if there is a cycle in the graph
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}

    def dfs(node):
        color[node] = GRAY

        for neighbor in graph[node]:
            if color[neighbor] == GRAY:
                return True
            if color[neighbor] == WHITE and dfs(neighbor):
                return True

        color[node] = BLACK
        return False

    for node in graph:
        if color[node] == WHITE and dfs(node):
            return True
    return False


def is_reachable(graph: Dict[int, Set[int]], u: int, v: int) -> bool:
    """
    Returns if v is reachable from u in the graph
    """
    visited = set()

    def dfs(graph, start, end):
        if start == end:
            return True
        visited.add(start)
        for neighbor in graph[start]:
            if neighbor not in visited and dfs(graph, neighbor, end):
                return True

        return False

    return dfs(graph, u, v)


def calculate_reachability_matrix(
    pipes: Dict[int, Pipe]
) -> Dict[Tuple[int, int], bool]:
    """
    Given all pipes, return a dict that specifies if two pipes are reorderable.
    """
    matrix = {}  # just use a dict with tuple keys
    constraint_graph = derive_constraint_graph(pipes)
    p_ids = list(pipes.keys())

    # For each
    for u in p_ids:
        for v in p_ids:
            key = (u, v)
            if u == v:
                matrix[key] = True
            else:
                matrix[key] = is_reachable(constraint_graph, u, v)

    return matrix


def get_fixed_pipes(pipes: Dict[int, Pipe]) -> Set[int]:
    """
    Returns a set of all pipe IDs that are fixed
    """
    s = set()
    for p_id, pipe in pipes.items():
        if pipe._fix_order:
            s.add(p_id)
    return s


def is_reorderable(
    reachability_matrix: Dict[Tuple[int, int], bool],
    fixed_pipes: Set[int],
    u: int,
    v: int,
) -> bool:
    # Ignore directionality on the edge
    if (
        reachability_matrix[(u, v)]
        or reachability_matrix[(v, u)]
        or u in fixed_pipes
        or v in fixed_pipes
    ):
        return False
    return True


def _add_root(graph: Dict[int, Set[int]], root: int):
    """
    Adds root r to graph.
    """
    curr_root = get_output_pipe(graph)
    if root in graph:
        raise RuntimeError("Found root sub-graph")
    graph[curr_root].add(root)
    graph[root] = set()


def _replace_root(
    graph: Dict[int, Set[int]], root: int
) -> Tuple[int, bool]:
    """
    Replaces the current root of graph with root.
    Returns (old_root, was_append): was_append is True when the graph had
    only one node and we only appended root (no real swap), so the caller
    should not recurse on the result to avoid infinite recursion.
    """
    curr_root = get_output_pipe(graph)
    if root in graph:
        raise RuntimeError("Error swapping roots")

    input_adj_list = flip_adj_list(graph)
    num_predecessors = len(input_adj_list.get(curr_root, set()))

    if num_predecessors == 0:
        # Single-node graph: append root as new output (curr_root -> root)
        graph[curr_root] = set([root])
        graph[root] = set()
        return (curr_root, True)

    if num_predecessors != 1:
        raise RuntimeError("Error swapping roots")

    input_node = list(input_adj_list[curr_root])[0]

    if len(graph[input_node]) != 1:
        raise RuntimeError("Error swapping roots")

    graph[input_node] = set([root])
    graph[root] = set()
    del graph[curr_root]

    return (curr_root, False)


def _remove_root(graph: Dict[int, set[int]]) -> int:
    """
    Remove the root from this graph and return it
    """
    root = get_output_pipe(graph)
    input_adj_list = flip_adj_list(graph)

    if len(input_adj_list[root]) != 1:
        raise RuntimeError("Error removing root. Multiple inputs")

    input_node = list(input_adj_list[root])[0]
    graph[input_node] = set()
    del graph[root]
    return root


def calculate_reorderings(
    pipes: Dict[int, Pipe], graph: Dict[int, Set[int]]
) -> List[Dict[int, Set[int]]]:
    """
    Calculates all permissable reorderings of the dataflow specified by graph,
    under the constraint graph contained in the pipe definitions.

    Precondition: The graph should be fully connected

    NOTE: non-linear graphs are not supported

    Args:
        pipes: Dict of all pipe IDs to pipes in the graph. Pipes should
            contain attributes specifying their ordering/dependencies
        graph: The original dataflow

    Returns:
        A list of graphs specifying all permissible reorderings of the
        input graph.
    """
    if len(bfs_order(pipes, graph)) != len(graph) or len(pipes) != len(graph):
        raise RuntimeError("Invalid graph. Graph should be fully connected")

    fixed_pipes = get_fixed_pipes(pipes)
    reachability_matrix = calculate_reachability_matrix(pipes)
    memo = {}

    # If the root is a source pipe, there are no more reorderings
    def _reordering_helper(subgraph: Dict[int, Set[int]]) -> List[Dict]:
        subgraph_reorderings = []
        dict_key = _get_dict_key(subgraph)

        if dict_key in memo:
            return memo[dict_key]

        root = get_output_pipe(subgraph)

        if pipes[root].is_source():
            # Base case, just return the root itself
            subgraph_reorderings = [{root: set()}]
        else:
            # Remove the root
            shrunk_graph = copy.deepcopy(subgraph)
            removed_root = _remove_root(shrunk_graph)
            if removed_root != root:
                raise RuntimeError("Error removing root")

            # Calculate reorderings for the shrunk graph
            shrunk_reorderings = _reordering_helper(shrunk_graph)
            cand = set()

            # For each (shrunk) reordering, extend the graph with root.
            # Also see if we can reorder the
            # root of the reordering with the current root. If so,
            # create a new reordering that reorders the two nodes.
            for shrunk_reordering in shrunk_reorderings:
                shrunk_root = get_output_pipe(shrunk_reordering)
                # Add the current root as the output
                g = copy.deepcopy(shrunk_reordering)
                _add_root(g, root)
                subgraph_reorderings.append(g)

                if shrunk_root not in cand and is_reorderable(
                    reachability_matrix, fixed_pipes, shrunk_root, root
                ):
                    cand.add(shrunk_root)

                    swapped_graph = copy.deepcopy(shrunk_reordering)
                    _, was_append = _replace_root(swapped_graph, root)

                    # When graph had one node we only appended; same as _add_root
                    # above — do not recurse to avoid infinite recursion.
                    if not was_append:
                        swapped_reorderings = _reordering_helper(swapped_graph)
                        for swapped_reordering in swapped_reorderings:
                            g = copy.deepcopy(swapped_reordering)
                            _add_root(g, shrunk_root)
                            subgraph_reorderings.append(g)

        memo[dict_key] = subgraph_reorderings
        return subgraph_reorderings

    final_reorderings = _reordering_helper(graph)
    return final_reorderings


def _get_dict_key(d: Dict[int, Set[int]]):
    return tuple((k, tuple(sorted(v))) for k, v in sorted(d.items()))


def get_output_pipe(graph: Dict[int, Set[int]]):
    """
    Returns the ID of the output pipe of the input graph.

    Raises:
        AssertionError on invalid input.
    """
    leaf_node_ids = [k for k, v in graph.items() if len(v) == 0]
    if len(leaf_node_ids) < 1:
        raise AssertionError("Could not find output node.")
    elif len(leaf_node_ids) > 1:
        raise AssertionError("Ember does not support multiple output pipes.")

    return leaf_node_ids[0]


def power_set(s: List[int]) -> List[List[int]]:
    """
    Returns the power set of s
    """
    if len(s) == 0:
        return [[]]
    subsets = power_set(s[:-1])
    next_elem = s[-1]
    return subsets + [item + [next_elem] for item in subsets]


def all_slices(s: List[int]) -> List[List[int]]:
    """
    Returns list of all slices of s
    """
    n = len(s)
    slices = []
    for i in range(n):
        for j in range(i + 1, n + 1):
            slices.append(s[i:j])
    return slices


def find_all_paths(g: Dict[int, Set[int]], start: int, end: int, path=[]):
    path = path + [start]

    if start == end:
        return [path]

    if start not in g:
        return []

    paths = []
    for node in g[start]:
        if node not in path:
            newpaths = find_all_paths(g, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)

    return paths
