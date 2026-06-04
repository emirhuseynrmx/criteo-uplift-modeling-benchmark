from pathlib import Path

from criteo_uplift_benchmark.downloader import find_criteo_csv, has_kaggle_credentials


def test_has_kaggle_credentials_with_file(tmp_path: Path) -> None:
    kaggle_dir = tmp_path / ".kaggle"
    kaggle_dir.mkdir()
    (kaggle_dir / "kaggle.json").write_text("{}", encoding="utf-8")

    assert has_kaggle_credentials(home=tmp_path)


def test_find_criteo_csv(tmp_path: Path) -> None:
    csv = tmp_path / "criteo-uplift-v2.1.csv"
    csv.write_text("f0,treatment,visit\n0,1,0\n", encoding="utf-8")

    assert find_criteo_csv(tmp_path) == csv

