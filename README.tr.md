# Criteo Uplift Modeling Benchmark

[English README](README.md)

Criteo Uplift v2.1 dataseti üzerinde uplift modeling benchmark projesi.

Bu repo notebook-first değildir. Kodlar profesyonel bir Python proje yapısına bölünmüştür: Pydantic v2 config modelleri, CLI, Kaggle downloader, testler, linting ve görsel üretim scriptleri vardır.

Ana soru:

> "Kim dönüşüm yapar?" değil, "Kim kampanya sayesinde dönüşüm yapar?"

## Öne Çıkanlar

- Notebook yerine modüler `.py` dosyaları
- Pydantic v2 ile config ve result schema yapısı
- S-Learner, T-Learner, X-Learner, DR-Learner, Causal Forest
- Naive response ranker baseline
- AUUC, Qini curve, top-decile incremental uplift
- Visit ve conversion outcome desteği
- Runtime ve sampled process RSS takibi
- Kaggle downloader
- `start.py` üzerinden CLI
- Ruff lint, pytest testleri ve GitHub Actions CI

## Sonuç

7M-row run içinde validation winner S-Learner oldu.

Ben bunu "S-Learner açık ara en iyi model" diye yazmazdım. S-Learner ile DR-Learner arasındaki paired bootstrap interval `[-217.63, 317.13]`, yani fark çok net desteklenmiyor.

Daha doğru yorum:

> S-Learner bu run'da en iyi production trade-off'u verdi.

## Çalıştırma

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python start.py assets
python start.py download --data-dir data --dataset arashnic/uplift-modeling
python start.py run --sample-size 500000
ruff check .
pytest
```

## Pydantic v2 Kullanımı

```python
from criteo_uplift_benchmark.config import BenchmarkConfig, RuntimeConfig
from criteo_uplift_benchmark.pipeline import run_benchmark

config = BenchmarkConfig(
    runtime=RuntimeConfig(
        sample_size=500_000,
        run_dr_learner=False,
        run_causal_forest=False,
    )
)

result = run_benchmark(config)
print(result.winner)
```

## Kaggle

Kaggle indirme için:

```bash
python start.py download --data-dir data --dataset arashnic/uplift-modeling
```

Kaggle credential için `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variable'ları veya `~/.kaggle/kaggle.json` gerekir.
