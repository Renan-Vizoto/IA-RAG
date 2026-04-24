import json
import os
import traceback
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
os.environ.setdefault("MPLBACKEND", "Agg")


ROOT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT_DIR / "notebooks" / "dutch_energy_regressao.ipynb"
RESULTS_DIR = ROOT_DIR / "notebooks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_notebook_code_cells(path: Path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [
        (idx, "".join(cell.get("source", [])))
        for idx, cell in enumerate(notebook.get("cells", []))
        if cell.get("cell_type") == "code"
    ]


def write_execution_summary(globals_dict):
    summary_path = RESULTS_DIR / "execution_summary.md"

    dataset_audit = globals_dict.get("dataset_audit")
    split_summary = globals_dict.get("split_summary")
    res_df = globals_dict.get("res_df")

    lines = [
        "# Dutch Energy Regression Summary",
        "",
    ]

    if dataset_audit is not None:
        lines.extend(
            [
                "## Dataset Audit",
                "",
                "```text",
                dataset_audit.to_string(index=False),
                "```",
                "",
            ]
        )

    if split_summary is not None:
        lines.extend(
            [
                "## Split Summary",
                "",
                "```text",
                split_summary.to_string(index=False),
                "```",
                "",
            ]
        )

    if res_df is not None:
        lines.extend(
            [
                "## Model Comparison",
                "",
                "```text",
                res_df.to_string(),
                "```",
                "",
            ]
        )

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Resumo salvo em: {summary_path}")


def main():
    os.chdir(ROOT_DIR)
    cells = load_notebook_code_cells(NOTEBOOK_PATH)
    shared_globals = {"__name__": "__main__", "__file__": str(NOTEBOOK_PATH)}

    for idx, source in cells:
        if not source.strip():
            continue
        print(f"\n>>> Executando celula {idx}")
        try:
            exec(compile(source, f"{NOTEBOOK_PATH}::cell_{idx}", "exec"), shared_globals)
        except Exception:
            print(f"\nFalha na celula {idx}\n")
            print(source)
            print("\nTraceback:\n")
            print(traceback.format_exc())
            raise

    write_execution_summary(shared_globals)


if __name__ == "__main__":
    main()
