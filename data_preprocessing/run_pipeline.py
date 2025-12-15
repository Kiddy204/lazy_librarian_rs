"""Main entry point for data preprocessing pipeline."""

import argparse
import json
from pathlib import Path

from build_item_features import build_item_features
from config import OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(description="Run data preprocessing pipeline")
    parser.add_argument(
        "--local-embeddings",
        action="store_true",
        help="Use local sentence-transformers instead of OpenAI API",
    )
    parser.add_argument(
        "--output-stats", type=str, default=None, help="Path to save statistics JSON"
    )
    args = parser.parse_args()

    print("Starting data preprocessing pipeline...")
    print(f"Using {'local' if args.local_embeddings else 'OpenAI'} embeddings")

    stats = build_item_features(use_local_embeddings=args.local_embeddings)

    # Save stats
    stats_path = args.output_stats or OUTPUT_DIR / "pipeline_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"\nStatistics saved to: {stats_path}")


if __name__ == "__main__":
    main()
