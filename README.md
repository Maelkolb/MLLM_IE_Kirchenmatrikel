# gendb-mllm-ie

MLLM-basierte Informationsextraktion aus historischen Kirchenbüchern (Matrikeln):
Evaluation von **Gemini 3.5 Flash** gegen die Verkartungsdatenbank
**GenDB** (Bistumsarchiv Passau) mit Scans von **Matricula Online**.

## Pipeline

```
scraper/    GenDB-Abfrage + Matricula-Scan-Download
            -> corpus/corpus_manifest.json (Bild -> Gold-Einträge), gold.csv, scans/
extraction/ Gemini 3.5 Flash, Structured Output (JSON-Schema = GenDB-Felder)
            -> predictions/predictions_<config>.json
evaluation/ Entry-Alignment (ungarische Methode) + feldweise Metriken
            -> eval/report_<config>.json, comparisons_<config>.csv
paper/      LaTeX-Quelle (Overleaf)
```

## Quickstart

```bash
pip install -r requirements.txt

# 1) Korpus bauen (Seiten-Specs in scraper/gendb_scraper.py anpassen)
python scraper/gendb_scraper.py

# 2) Extraktion (beide Settings, Ablationen via Flags)
export GEMINI_API_KEY=...
python extraction/gemini_extract.py --manifest corpus/corpus_manifest.json --mode page
python extraction/gemini_extract.py --manifest corpus/corpus_manifest.json --mode entry
python extraction/gemini_extract.py --manifest ... --mode page --thinking-level high --media-resolution high

# 3) Evaluation
python evaluation/evaluate.py --manifest corpus/corpus_manifest.json \
    --predictions predictions/predictions_page_gemini-3.5-flash_tl-default_mr-default.json
```

Der Scraper unterstützt weiterhin den ursprünglichen Suchmodus
(`query_gendb(...)`, CSV/HTML-Output) zusätzlich zum Page-Mode
(`build_corpus(...)`).

## Methodische Hinweise

- GenDB ist eine **Verkartung** (Index), keine Transkription: bewertet werden
  nur Felder, die im Gold Standard belegt sind.
- Datumsfelder werden zusätzlich getrennt für nicht-geschätzte Gold-Daten
  ausgewiesen (`accuracy_exact_dates_only`).
- Scans werden aus Lizenzgründen **nicht** im Repo versioniert
  (`corpus/`, `predictions/` in `.gitignore`); das Manifest macht den Korpus
  reproduzierbar.
