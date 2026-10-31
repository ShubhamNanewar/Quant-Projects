"""Add a second oracle observation and dependency inventory to an existing capture."""
import argparse
import json

from .analyze import ROOT, verify
from .collect import Capture, MORPHO


def enrich(run):
    directory = ROOT / "data/raw" / run
    manifest = verify(directory)
    capture = Capture.__new__(Capture)
    capture.directory, capture.records = directory, manifest["requests"]
    if (directory / "morpho_oracles_end.json").exists():
        raise ValueError("Enrichment already exists; do not overwrite observed data")
    try:
        capture.fetch("morpho_oracles_end", MORPHO,
            body={"query": (ROOT / "queries/oracles.graphql").read_text()})
        revision = manifest["settings"]["contract_revision"]
        capture.fetch("contract_LICENSE", f"https://raw.githubusercontent.com/morpho-org/morpho-blue/{revision}/LICENSE", text=True)
    finally:
        # The first collection boundary remains available after the second observation.
        settings = manifest["settings"]
        settings["initial_collection_completed_at"] = manifest["completed_at"]
        capture.save_manifest(settings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=(ROOT / "data/LATEST").read_text().strip())
    enrich(parser.parse_args().run)
