# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:37:59 2026

@author: 10857
"""

import itertools
import re
from typing import List, Tuple
import networkx as nx
import numpy as np
import pandas as pd


def _subset_mapping(
    input_data: List[str], mapping: List[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    return [m for m in mapping if m[0] in input_data]


def _subset_pathways_on_idx(
    pathways: List[Tuple[str, str]], mapping: List[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    """
    Recursively find all source nodes that are connected to the mapping
    targets. Only pathways whose 'source' is in the resulting set are kept.
    """

    def add_pathways(idx_list, target):
        if not target:
            return idx_list
        updated_idx_list = idx_list + target
        subsetted = [p for p in pathways if p[0] in target]
        new_target = list({p[1] for p in subsetted})
        return add_pathways(updated_idx_list, new_target)

    original_target = list({m[1] for m in mapping})
    idx_list = add_pathways([], original_target)
    return [p for p in pathways if p[0] in idx_list]


def _get_mapping_to_all_layers(
    pathways: List[Tuple[str, str]], mapping: List[Tuple[str, str]]
) -> pd.DataFrame:
    """
    Build a DataFrame of (input, connections) by traversing the directed graph
    from each mapped translation node.
    """
    graph = nx.DiGraph()
    graph.add_edges_from(pathways)

    components = {"input": [], "connections": []}
    unique_inputs = {m[0] for m in mapping}

    for inp in unique_inputs:
        # Collect all translation nodes for this input
        translation_nodes = [m[1] for m in mapping if m[0] == inp]

        for node_id in translation_nodes:
            if graph.has_node(node_id):
                reachable_nodes = nx.single_source_shortest_path(graph, node_id).keys()
                # Record the connections
                for connection in reachable_nodes:
                    components["input"].append(inp)
                    components["connections"].append(connection)

    df_components = pd.DataFrame(components).drop_duplicates()
    return df_components


def _get_map_from_layer(layer_dict: dict) -> pd.DataFrame:
    """
    Convert a dictionary like {pathway: [inputs]} into a binary matrix
    (rows = pathways, columns = inputs).
    """
    pathways = list(layer_dict.keys())
    all_inputs = list(itertools.chain.from_iterable(layer_dict.values()))
    unique_inputs = list(np.unique(all_inputs))

    df = pd.DataFrame(index=pathways, columns=unique_inputs)
    for pathway, inputs in layer_dict.items():
        df.loc[pathway, inputs] = 1
    df = df.infer_objects(copy=False).fillna(0)

    return df.T


def _add_edges(G: nx.DiGraph, node: str, n_layers: int) -> nx.DiGraph:
    """
    Create 'n_layers' copies of a node (node_copy1, node_copy2, ...),
    connecting each copy to the next, and add them to the graph.
    """
    edges = []
    source = node
    for level in range(n_layers):
        target = f"{node}_copy{level + 1}"
        edges.append((source, target))
        source = target

    G.add_edges_from(edges)
    return G


def _complete_network(G: nx.DiGraph, n_layers: int = 4) -> nx.DiGraph:
    """
    Extend the network so that every terminal node has up to 'n_layers' copies,
    ensuring a uniform depth for all terminal nodes.
    """
    sub_graph = nx.ego_graph(G, "output_node", radius=n_layers)
    terminal_nodes = [n for n, d in sub_graph.out_degree() if d == 0]

    for node in terminal_nodes:
        distance = len(nx.shortest_path(sub_graph, source="output_node", target=node))
        if distance <= n_layers:
            diff = n_layers - distance + 1
            _add_edges(sub_graph, node, diff)

    return sub_graph


def _get_nodes_at_level(net: nx.DiGraph, distance: int) -> List[str]:
    """
    Get the set of nodes that lie exactly 'distance' steps away from 'output_node'.
    """
    nodes_at_distance = set(nx.ego_graph(net, "output_node", radius=distance))
    if distance >= 1:
        nodes_one_less = set(nx.ego_graph(net, "output_node", radius=distance - 1))
        nodes_at_distance -= nodes_one_less
    return list(nodes_at_distance)


def _get_layers_from_net(net: nx.DiGraph, n_layers: int) -> List[dict]:
    """
    Extract layer information from the graph, from level 0 up to n_layers-1.
    """
    layers = []
    for dist in range(n_layers):
        nodes = _get_nodes_at_level(net, dist)
        layer_map = {}
        for n in nodes:
            base_name = re.sub(r"_copy.*", "", n)
            successors = list(net.successors(n))
            successors_base = [re.sub(r"_copy.*", "", s) for s in successors]
            layer_map[base_name] = successors_base
        layers.append(layer_map)
    return layers


class PathwayNetwork:
    """
    Represents a network of pathways built from an input set, a list of edges,
    and a mapping that links each input to one or more pathways.
    """

    def __init__(
        self,
        input_data: List[str],
        pathways: List[Tuple[str, str]] = None,
        mapping: List[Tuple[str, str]] = None,
        routing_mode: str = "strict",
        residual_width: int = 1,
    ):
        """
        :param input_data: e.g., [protein1, protein2, protein3]
        :param pathways: e.g., [(path1, path2), (path3, path4)]
        :param mapping: e.g., [(protein1, path1), (protein2, path3)]
        :param residual_width: number of Residual nodes per layer; 1 keeps the
            original single "Residual" node.
        """
        self.input_data = input_data
        self.pathways = pathways
        self.mapping = mapping
        if routing_mode not in {"original", "simple_dual", "high_exclusive", "strict"}:
            raise ValueError(
                "routing_mode must be 'original', 'simple_dual', "
                "'high_exclusive', or 'strict', "
                f"got {routing_mode!r}"
            )
        self.routing_mode = routing_mode
        if int(residual_width) < 1:
            raise ValueError(f"residual_width must be at least 1, got {residual_width!r}")
        digits = len(str(int(residual_width)))
        self.residual_nodes = (
            ["Residual"] if int(residual_width) == 1
            else [f"Residual_{k:0{digits}d}" for k in range(1, int(residual_width) + 1)]
        )

        # Keep the full input list before subsetting.  Residual membership is
        # decided later from reachability across *all* requested H layers, not
        # merely from the input -> H1 matrix.
        self.all_inputs = sorted(input_data)

        # Subset the mapping to keep only entries related to input_data
        self.mapping = _subset_mapping(self.input_data, self.mapping)

        # Subset the pathways by the indices derived from the mapping
        self.pathways = _subset_pathways_on_idx(self.pathways, self.mapping)

        # Expand the mapping to all reachable layers
        self.mapping = _get_mapping_to_all_layers(self.pathways, self.mapping)

        # Maintain a list of unique inputs from the expanded mapping
        self.inputs = sorted(set(self.mapping["input"].to_list()))

        # Build the base network graph
        self.pathway_graph = self.build_network()

    def build_network(self) -> nx.DiGraph:
        """
        Build a directed graph from pathways. Also adds an artificial node,
        'output_node', which connects to all nodes that have no incoming edges.
        """
        pathway_graph = nx.DiGraph()
        pathway_graph.add_edges_from(self.pathways)
        pathway_graph = pathway_graph.reverse()

        # Create a special output node for all sources with no incoming edges
        output_node = "output_node"
        sink_nodes = [n for n, d in pathway_graph.in_degree() if d == 0]
        edges_to_output = [(output_node, node) for node in sink_nodes]
        pathway_graph.add_edges_from(edges_to_output)

        return pathway_graph

    def get_layers(self, n_layers: int) -> List[dict]:
        """
        Get layered structure of the network up to n_layers, plus a final dict
        mapping terminal pathways -> which inputs map to them.

        :param n_layers: Number of layers to extract
        :return: A list of length n_layers + 1:
                 - n_layers layers describing pathway relationships
                 - A final dictionary mapping terminal pathways to input(s).
        """
        completed_graph = _complete_network(self.pathway_graph, n_layers=n_layers)
        layers = _get_layers_from_net(completed_graph, n_layers)

        # Identify terminal nodes in the extended graph
        terminal_nodes = [n for n, d in completed_graph.out_degree() if d == 0]

        # Create a final mapping of terminal pathways -> input(s)
        mapping_df = self.mapping
        terminal_mapping = {}
        missing_pathways = []
        for term_node in terminal_nodes:
            pathway_name = re.sub(r"_copy.*", "", term_node)
            inputs_for_pathway = (
                mapping_df.loc[mapping_df["connections"] == pathway_name, "input"]
                .unique()
                .tolist()
            )

            if not inputs_for_pathway:
                missing_pathways.append(pathway_name)
            terminal_mapping[pathway_name] = inputs_for_pathway

        layers.append(terminal_mapping)
        return layers

    def get_connectivity_matrices(self, n_layers: int) -> List[pd.DataFrame]:
        """
        Produce one connectivity matrix per layer (bottom-up). Each matrix
        displays which inputs are connected to which pathways in that layer.

        :param n_layers: Number of layers
        :return: A list of pandas DataFrames, one per layer
        """
        matrices = []
        layers = self.get_layers(n_layers)

        # H1 ... Hn are the column-node sets of the first n matrices.  An input
        # is Residual only if it reaches none of these sets.  If its first
        # reachable set is Hk (k > 1), it enters the model only through a direct
        # input -> Hk connection and is not duplicated in H1 Residual.
        h_layer_nodes = [set(layers[n_layers - i].keys()) for i in range(n_layers)]
        connections_by_input = (
            self.mapping.groupby("input")["connections"].agg(set).to_dict()
        )
        self.input_connected_layers = {}
        self.input_entry_layer = {}
        self.input_entry_nodes = {}
        self.uses_h1_residual = {}
        self.uses_direct_high_layer = {}
        for inp in self.all_inputs:
            connections = connections_by_input.get(inp, set())
            connected_layers = tuple(
                i + 1 for i, nodes in enumerate(h_layer_nodes)
                if connections.intersection(nodes)
            )
            self.input_connected_layers[inp] = connected_layers
            # Original MAPS only admits raw inputs at H1. Inputs with no H1
            # connection are routed through H1 Residual even if Reactome can
            # connect them to H2-Hn. The newer modes expose those higher-layer
            # connections directly.
            if self.routing_mode == "original" and 1 not in connected_layers:
                self.input_entry_layer[inp] = "Residual"
                self.input_entry_nodes[inp] = ["Residual"]
            elif connected_layers:
                entry_layer = connected_layers[0]
                self.input_entry_layer[inp] = entry_layer
                self.input_entry_nodes[inp] = sorted(
                    connections.intersection(h_layer_nodes[entry_layer - 1])
                )
            else:
                self.input_entry_layer[inp] = "Residual"
                self.input_entry_nodes[inp] = ["Residual"]
            # Strict/high-exclusive: H1 Residual contains only inputs unable to
            # reach H1-Hn.  The two modes differ only in pathway-node skips:
            # strict adds nearest-reachable skips; high-exclusive adds none.
            # Simple dual-path: an input without an H1 mapping keeps the H1
            # Residual route even when it also enters its earliest H2-Hn node.
            # Original: an input without an H1 mapping uses Residual only.
            self.uses_h1_residual[inp] = (
                not connected_layers
                if self.routing_mode in {"high_exclusive", "strict"}
                else 1 not in connected_layers
            )
            self.uses_direct_high_layer[inp] = bool(
                self.routing_mode != "original"
                and connected_layers
                and connected_layers[0] > 1
            )

        # One optional raw-input skip mask per hidden layer.  H1 uses the normal
        # first connectivity matrix, so index 0 remains None.  H2-Hn masks are
        # populated only for inputs whose earliest reachable layer is that H.
        self.direct_input_matrices = [None] * n_layers

        # Process layers in reverse order so the bottom-most is first.
        current_inputs = self.inputs  # will be updated as we go
        for i, layer_dict in enumerate(layers[::-1]):
            layer_matrix = _get_map_from_layer(layer_dict)

            # The bottom matrix must keep every raw feature.  Some rows are
            # intentionally all-zero here because those inputs enter at H2-Hn.
            if i == 0:
                current_inputs = self.all_inputs
                self.inputs = current_inputs

            # LEFT merge: keep every node in current_inputs.
            # Nodes absent from layer_matrix get NaN → filled to 0 below.
            # (Original was how="inner", which silently dropped unmatched nodes.)
            placeholder_df = pd.DataFrame(index=current_inputs)
            merged_df = placeholder_df.merge(
                layer_matrix, right_index=True, left_index=True, how="left"
            )
            merged_df = merged_df.fillna(0)

            # At the raw-input layer, the H1 Residual node(s) receive only inputs
            # that are unreachable from every requested H layer.  At higher layers
            # only Residual nodes feed Residual nodes (all-to-all when wider than
            # one node); real pathway nodes are never folded into this
            # non-biological channel.
            if i == 0:
                unmatched_mask = pd.Series(
                    [self.uses_h1_residual[x] for x in merged_df.index],
                    index=merged_df.index,
                )
                if unmatched_mask.any():
                    for node in self.residual_nodes:
                        merged_df[node] = 0
                        merged_df.loc[unmatched_mask, node] = 1
            else:
                residual_rows = [node for node in self.residual_nodes if node in merged_df.index]
                if residual_rows:
                    for node in self.residual_nodes:
                        if node not in merged_df.columns:
                            merged_df[node] = 0
                    merged_df.loc[residual_rows, self.residual_nodes] = 1

            # Sort rows and columns for consistency
            merged_df = merged_df.reindex(sorted(merged_df.columns), axis=1)
            merged_df = merged_df.sort_index()

            # Raw-input skip mask for inputs that first enter at this H layer.
            # This preserves their actual Reactome connection(s); they are not
            # mixed into Residual and are not duplicated in lower H layers.
            if 0 < i < n_layers:
                direct_df = pd.DataFrame(
                    0, index=self.all_inputs, columns=merged_df.columns,
                    dtype=np.uint8,
                )
                eligible = [
                    inp for inp in self.all_inputs
                    if self.uses_direct_high_layer[inp]
                    and self.input_entry_layer[inp] == i + 1
                ]
                for inp in eligible:
                    targets = [
                        node for node in self.input_entry_nodes[inp]
                        if node in direct_df.columns
                    ]
                    if not targets:
                        raise RuntimeError(
                            f"Input {inp!r} is assigned to H{i + 1} but has no "
                            "matching node in that layer"
                        )
                    direct_df.loc[inp, targets] = 1
                self.direct_input_matrices[i] = direct_df

            # Residual is the only special node propagated through all layers.
            current_inputs = sorted(merged_df.columns)
            matrices.append(merged_df)

        # Rescue biologically meaningful routes hidden by the uniform-depth
        # padding used above.  A same-name Hi -> H(i+1) edge is an artificial
        # copy edge, not a Reactome parent relation.  When a real pathway node
        # has no different-node target in the adjacent layer, connect it to all
        # Reactome-reachable nodes in the nearest higher layer.  Existing copy
        # edges are retained as continuity paths; the new masked projections
        # are initialized at zero by the model.
        self.pathway_skip_matrices = {}
        self.pathway_skip_edges = []
        pathway_up_graph = nx.DiGraph()
        pathway_up_graph.add_edges_from(self.pathways)  # child -> parent
        residual = set(self.residual_nodes)
        h_nodes = [set(matrix.columns) - residual for matrix in matrices[:n_layers]]

        for source_idx in range(n_layers - 1):
            if self.routing_mode != "strict":
                break
            adjacent = matrices[source_idx + 1]
            source_nodes = sorted(h_nodes[source_idx])
            for source_node in source_nodes:
                if source_node not in adjacent.index:
                    continue
                real_adjacent_targets = [
                    target_node
                    for target_node in adjacent.columns
                    if target_node != source_node and target_node not in residual
                    and adjacent.loc[source_node, target_node] != 0
                ]
                if real_adjacent_targets:
                    continue

                reachable = (
                    nx.descendants(pathway_up_graph, source_node)
                    if pathway_up_graph.has_node(source_node)
                    else set()
                )
                for target_idx in range(source_idx + 2, n_layers):
                    targets = sorted(reachable.intersection(h_nodes[target_idx]))
                    if not targets:
                        continue
                    key = (source_idx, target_idx)
                    if key not in self.pathway_skip_matrices:
                        self.pathway_skip_matrices[key] = pd.DataFrame(
                            0,
                            index=sorted(h_nodes[source_idx] | residual),
                            columns=sorted(h_nodes[target_idx] | residual),
                            dtype=np.uint8,
                        )
                    skip_matrix = self.pathway_skip_matrices[key]
                    for target_node in targets:
                        skip_matrix.loc[source_node, target_node] = 1
                        self.pathway_skip_edges.append({
                            "source_layer": f"H{source_idx + 1}",
                            "target_layer": f"H{target_idx + 1}",
                            "source_node": source_node,
                            "target_node": target_node,
                            "reason": "no_real_adjacent_edge_nearest_reachable_layer",
                        })
                    break

        return matrices


def dataframes_to_pathway_network(
    data_matrix: pd.DataFrame,
    pathway_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    entity_col: str = "Protein",
    source_col: str = "source",
    target_col: str = "target",
    input_col: str = "input",
    translation_col: str = "translation",
    routing_mode: str = "strict",
    residual_width: int = 1,
) -> PathwayNetwork:
    """
    Construct a PathwayNetwork from three DataFrames:
      1) A 'data_matrix' containing the input entities (e.g. proteins).
      2) A 'pathway_df' containing pathway edges (source -> target).
      3) A 'mapping_df' mapping each entity in data_matrix to pathways.

    This method should be as flexible as possible.

    Args:
        data_matrix (pd.DataFrame): A DataFrame that contains the column 'entity_col'.
          For example, if `entity_col='Protein'`, this DataFrame might have a
          "Protein" column listing protein identifiers.
        pathway_df (pd.DataFrame): A DataFrame with at least two columns describing
          directed edges in a pathway (defaults: 'source', 'target').
        mapping_df (pd.DataFrame): A DataFrame with columns describing how
          each entity maps to pathways (defaults: 'input', 'translation').
        entity_col (str): Name of the column in `data_matrix` containing
          the entities of interest (defaults to 'Protein').
        source_col (str): Name of the column in `pathway_df` representing
          the source node in an edge (defaults to 'source').
        target_col (str): Name of the column in `pathway_df` representing
          the target node in an edge (defaults to 'target').
        input_col (str): Name of the column in `mapping_df` representing
          the input entity (defaults to 'input').
        translation_col (str): Name of the column in `mapping_df` representing
          the mapped pathway node (defaults to 'translation').

    Returns:
        PathwayNetwork: A newly constructed PathwayNetwork instance, ready for
        further analysis (e.g., layer extraction, connectivity matrices).
    """

    input_data = data_matrix[entity_col].dropna().unique().tolist()
    pathways = list(pathway_df[[source_col, target_col]].itertuples(index=False, name=None))
    mapping = list(mapping_df[[input_col, translation_col]].itertuples(index=False, name=None))

    pn = PathwayNetwork(
        input_data=input_data,
        pathways=pathways,
        mapping=mapping,
        routing_mode=routing_mode,
        residual_width=residual_width,
    )
    return pn
