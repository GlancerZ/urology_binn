# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 01:30:06 2026

@author: 10857
"""

import os
import pandas as pd


def load_reactome_db(input_source="uniprot"):
    """
    Load default Reactome mapping and pathways files.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if input_source == "uniprot":
        mapping_path = os.path.join(
            base_dir, "..", "data", "downloads", "UniProt2Reactome_All_Levels.txt"
        )
    elif input_source == "mirbase":
        mapping_path = os.path.join(
            base_dir, "..", "data", "downloads", "mirbase_2_reactome_2025_01_16.txt"
        )
    elif input_source == "uniprot&chebi":
        mapping_path = os.path.join(
            base_dir, "..", "data", "downloads", "UniProt&ChEBI&DIY2Reactome_Human.txt"
        )

    pathways_path = os.path.join(
        base_dir, "..", "data", "downloads", "ReactomePathwaysRelation.txt"
    )

    try:
        mapping = pd.read_csv(
            mapping_path,
            sep="\t",
            header=None,
            names=["input", "translation", "url", "name", "x", "species"],
        )
        pathways = pd.read_csv(
            pathways_path, sep="\t", header=None, names=["target", "source"]
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Unable to find file: {e.filename}")

    return {"mapping": mapping, "pathways": pathways}
