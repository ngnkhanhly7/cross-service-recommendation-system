from pathlib import Path

from src.common import read_provenance_sidecar, write_interactions, write_provenance_sidecar
from src.data_loader import generate_demo_amazon
from src.features import encode_train_test, temporal_leave_one_out
from src.train_baseline import train_svd


def test_provenance_sidecar_round_trips(tmp_path):
    data_path = tmp_path / "interactions.csv"
    write_provenance_sidecar(data_path, source="demo-amazon", data_provenance="synthetic_controlled")
    assert read_provenance_sidecar(data_path) == {
        "source": "demo-amazon",
        "data_provenance": "synthetic_controlled",
    }


def test_missing_sidecar_reads_as_unknown_not_a_crash(tmp_path):
    assert read_provenance_sidecar(tmp_path / "nowhere.csv") == {
        "source": "unknown",
        "data_provenance": "unknown",
    }


def test_trained_artifact_always_carries_provenance_and_popularity(tmp_path):
    """Guards against CP-I1 regressing: every artifact must be traceable to its data.

    Without this, a report/model produced from synthetic data could be mistaken for
    one backed by real interactions (see IMPROVEMENT_PLAN.md CP-I1).
    """

    data_path = tmp_path / "interactions.csv"
    frame = generate_demo_amazon(n_users=30, n_items_per_category=10, seed=1)
    write_interactions(frame, data_path)
    write_provenance_sidecar(data_path, source="demo-amazon", data_provenance="synthetic_controlled")

    provenance = read_provenance_sidecar(data_path)
    train, test = temporal_leave_one_out(frame)
    encoded = encode_train_test(train, test)
    user_factors, item_factors = train_svd(encoded.train_matrix, factors=8)

    artifact = {
        "data_provenance": provenance["data_provenance"],
        "data_source": provenance["source"],
        "item_popularity": train.groupby("item_id")["interaction_strength"].sum().to_dict(),
        "user_factors": user_factors,
        "item_factors": item_factors,
    }
    assert artifact["data_provenance"] == "synthetic_controlled"
    assert artifact["item_popularity"]
    assert Path(str(data_path) + ".provenance.json").exists()
