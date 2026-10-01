from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "BINN.main"))
from binn.model.pathway_network import dataframes_to_pathway_network  # noqa: E402


def toy_matrices(residual_width: int):
    """T1 > M1 > L1 and T2 > M2 > L2; p3 maps only to M2 and p4 is unmapped."""
    pathways = pd.DataFrame({"source": ["M1", "M2", "L1", "L2"], "target": ["T1", "T2", "M1", "M2"]})
    mapping = pd.DataFrame({
        "input": ["p1", "p1", "p1", "p2", "p2", "p2", "p3", "p3"],
        "translation": ["L1", "M1", "T1", "L2", "M2", "T2", "M2", "T2"],
    })
    data = pd.DataFrame({"Protein": ["p1", "p2", "p3", "p4"]})
    network = dataframes_to_pathway_network(
        data, pathways, mapping, entity_col="Protein", routing_mode="simple_dual",
        residual_width=residual_width,
    )
    return network, network.get_connectivity_matrices(n_layers=3)


class WideResidualTests(unittest.TestCase):
    def test_width_one_keeps_the_single_residual_node(self):
        network, matrices = toy_matrices(1)
        self.assertEqual(network.residual_nodes, ["Residual"])
        for matrix in matrices[:3]:
            self.assertIn("Residual", matrix.columns)

    def test_wide_residual_is_isolated_and_fully_connected(self):
        network, matrices = toy_matrices(3)
        residual = network.residual_nodes
        self.assertEqual(residual, ["Residual_1", "Residual_2", "Residual_3"])
        first = matrices[0]
        self.assertTrue((first.loc[["p3", "p4"], residual] == 1).all().all())
        self.assertTrue((first.loc[["p1", "p2"], residual] == 0).all().all())
        for matrix in matrices[1:3]:
            pathway_rows = [row for row in matrix.index if row not in residual]
            pathway_cols = [col for col in matrix.columns if col not in residual]
            self.assertTrue((matrix.loc[residual, residual] == 1).all().all())
            self.assertTrue((matrix.loc[residual, pathway_cols] == 0).all().all())
            self.assertTrue((matrix.loc[pathway_rows, residual] == 0).all().all())

    def test_direct_entry_does_not_touch_residual_nodes(self):
        network, _ = toy_matrices(3)
        direct = network.direct_input_matrices[1]
        self.assertEqual(int(direct.loc["p3", "M2"]), 1)
        self.assertTrue((direct[network.residual_nodes] == 0).all().all())


if __name__ == "__main__":
    unittest.main()
