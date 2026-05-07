"""
conftest de integração.
Usa LocalStorage (filesystem real) para que cada etapa escreva arquivos reais.
"""
import os
import pandas as pd
import pytest

from app.pipeline.storage import LocalStorage


@pytest.fixture
def local_storage(tmp_path):
    """Storage real baseado em filesystem."""
    return LocalStorage(str(tmp_path))


@pytest.fixture
def data_dir(tmp_path):
    """Diretório com CSVs sintéticos maiores para integração."""
    d = tmp_path / "data"
    d.mkdir()

    import numpy as np
    rng = np.random.default_rng(0)
    n = 200  # linhas suficientes para split e treinamento real

    df = pd.DataFrame({
        "net_manager":                  (["Manager A"] * 100 + ["Manager B"] * 100),
        "purchase_area":                (["Area A"] * 100 + ["Area B"] * 100),
        "city":                         [f"City_{i}" for i in range(n)],
        "num_connections":              (rng.integers(10, 50, n)).astype(float),
        "delivery_perc":                rng.uniform(80, 100, n),
        "perc_of_active_connections":   rng.uniform(85, 100, n),
        "type_of_connection":           (["3x25"] * 100 + ["1x35"] * 100),
        "type_conn_perc":               rng.uniform(70, 100, n),
        "annual_consume":               rng.uniform(2000, 8000, n),
        "annual_consume_lowtarif_perc": rng.uniform(40, 70, n),
        "smartmeter_perc":              rng.uniform(5, 30, n),
    })
    df.to_csv(d / "electricity_2018.csv", index=False)
    return str(d)
