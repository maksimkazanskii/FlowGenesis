from pathlib import Path
import argparse

import torch
import pandas as pd


# ================================================================
# Arguments
# ================================================================

parser = argparse.ArgumentParser(
    description="Summarize neurogenesis experiments."
)

parser.add_argument(
    "folder",
    type=str,
    help="Folder containing neurons_*/seed_* experiment directories.",
)

args = parser.parse_args()

ROOT = Path(args.folder)

if not ROOT.exists():
    raise FileNotFoundError(
        f"Folder does not exist: {ROOT}"
    )


# ================================================================
# Collect runs
# ================================================================

rows = []

for stats_file in sorted(
        ROOT.glob("neurons_*/seed_*/statistics.pt")
):

    run_dir = stats_file.parent

    initial_n = int(
        run_dir.parent.name.split("_")[1]
    )

    seed = int(
        run_dir.name.split("_")[1]
    )

    # ------------------------------------------------------------
    # Developmental statistics
    # ------------------------------------------------------------

    stats = torch.load(
        stats_file,
        map_location="cpu",
        weights_only=False,
    )

    births = stats.get("birth_history", [])
    deaths = stats.get("death_history", [])

    total_born = len(births)
    total_pruned = len(deaths)

    # ------------------------------------------------------------
    # Final model
    # ------------------------------------------------------------

    final_model_file = run_dir / "final_model.pt"

    if final_model_file.exists():

        final_model = torch.load(
            final_model_file,
            map_location="cpu",
            weights_only=False,
        )

        final_n = int(
            final_model["hidden_neurons"]
        )

    else:

        final_n = (
                initial_n
                + total_born
                - total_pruned
        )

    # ------------------------------------------------------------
    # Best model
    # ------------------------------------------------------------

    best_model_file = run_dir / "best_model.pt"

    if best_model_file.exists():

        best_model = torch.load(
            best_model_file,
            map_location="cpu",
            weights_only=False,
        )

        best_n = int(
            best_model["hidden_neurons"]
        )

        best_acc = float(
            best_model["test_acc"]
        )

        best_epoch = int(
            best_model["epoch"]
        )

    else:

        best_n = None
        best_acc = None
        best_epoch = None

    # ------------------------------------------------------------
    # Store
    # ------------------------------------------------------------

    rows.append({
        "initial_n": initial_n,
        "seed": seed,
        "best_n": best_n,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "final_n": final_n,
        "total_born": total_born,
        "total_pruned": total_pruned,
        "net_change": final_n - initial_n,
    })


# ================================================================
# Dataframe
# ================================================================

if not rows:
    raise RuntimeError(
        f"No statistics.pt files found under: {ROOT}"
    )

df = pd.DataFrame(rows)

df = df.sort_values(
    ["initial_n", "seed"]
)

print()
print(df.to_string(index=False))


# ================================================================
# Save
# ================================================================

out = ROOT / "neurogenesis_summary.csv"

df.to_csv(
    out,
    index=False,
)

print(f"\nSaved: {out}")