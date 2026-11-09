import matplotlib.pyplot as plt
import numpy as np

from cvd70k.evaluation.report import markdown_table
from cvd70k.visualization.figures import annotated_confusion, save, setup


def test_markdown_report_actual_values():
    table = markdown_table(["Model", "AP"], [["Synthetic fixture", 0.75]])
    assert "| Synthetic fixture | 0.75 |" in table
    assert "| --- | --- |" in table


def test_figure_generation(tmp_path):
    setup()
    fig, ax = plt.subplots()
    annotated_confusion(ax, np.array([5, 2, 1, 4]), "Invented test fixture")
    save(fig, tmp_path, "fixture")
    for extension in ["png", "pdf"]:
        path = tmp_path / f"fixture.{extension}"
        assert path.exists() and path.stat().st_size > 1000
