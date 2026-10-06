# Signal: Customer Feedback Intelligence

An analysis workspace for the original Apple-product review sentiment classifier. The new frontend makes its actual model outputs easier to inspect, compare, and export without changing the trained weights.

## Included

- Single-review inference and CSV batch analysis (up to 100 reviews).
- All five original classes: very negative, negative, neutral, positive, very positive.
- Per-class model scores, vocabulary coverage, token count, truncation warnings, and artifact version.
- Session review queue, sentiment/review filters, and spreadsheet-safe CSV export.
- Responsive interface, accessible forms, loading/error states, and a model card describing limitations.
- Original Keras model, tokenizer, and label encoder. No keyword-based or fabricated fallback predictions.

## Run Locally

Tested on Python 3.13. Use Python 3.10-3.13 with TensorFlow 2.20 support for your operating system.

```sh
git lfs install
git lfs pull
python -m venv .venv
# Activate .venv, then:
pip install -r requirements-web.txt
python app.py
```

Open http://127.0.0.1:5051. Initial inference loads the model and may take longer. A production-style local server can be started with:

```sh
waitress-serve --listen=127.0.0.1:5051 app:app
```

The original `requirements.txt` records the older environment. The isolated `requirements-web.txt` supplies a current Python-compatible serving runtime. Legacy Keras loading and the original tokenizer pickle module are handled explicitly in `inference.py`. scikit-learn stays pinned to 1.6.0, the saved label encoder's version.

## CSV and API

A UTF-8 CSV must contain a `review` column. Quotes, commas, and multiline quoted reviews are parsed with Python's standard CSV parser. Maximum upload: 1 MB; maximum review length: 5,000 characters. Empty rows are rejected with their row number. A sample is available in `static/sample-reviews.csv`.

```http
POST /api/analyze
Content-Type: application/json

{"reviews":["The battery lasts all day and the camera is excellent."]}
```

`POST /api/analyze` also accepts multipart `file`. `GET /api/model` returns serving metadata. A missing or incompatible model returns HTTP 503 and an actionable UI error, not fake scores.

## Interpretation and Limits

The model processes the original saved tokenization and post-pads/truncates to 350 tokens. Class names come from the saved encoder. The displayed artifact version is the first 12 characters of its SHA-256 hash.

Review flags are transparent interface heuristics: a top score below 70%, vocabulary coverage below 50%, an unrecognized input, or truncation. They are **not calibrated uncertainty estimates or validated operational thresholds**. Positive/negative summary shares include their respective "very" classes.

No held-out evaluation dataset or metrics artifact is present, so the interface makes no new accuracy, lift, ROI, or production-performance claim. The original domain is Apple product reviews. This illustrates unstructured-text preparation, classification, inspection, and operational handoff; it is not a collections, credit-risk, or propensity-to-pay model. A new use case needs labeled domain data and validation first.

## Privacy and Deployment

Inference runs on this application's server, not an external AI service. Review text is held in memory for the request. Session history is kept only in the current browser tab, is not saved to browser storage, and disappears on reload. Uploaded files are not saved. No submitted text is logged by application code.

This local portfolio app has no user authentication or shared review database. Before public/multi-user hosting, add access control, request/compute rate limiting, HTTPS, appropriate retention rules, and resource isolation. Model inference is serialized to avoid simultaneous access to the legacy model. Large public workloads need a queue rather than unbounded HTTP requests. Never load pickled model artifacts from untrusted sources.

## Verify

```sh
python -m pytest tests -q
# To include real model inference:
# PowerShell: $env:RUN_MODEL_TESTS='1'
# macOS/Linux: export RUN_MODEL_TESTS=1
python -m pytest tests -q
```

The API tests cover CSV parsing, blank/invalid input, limits, and model failure. The opt-in integration test runs the real artifact and checks all five scores, version, out-of-vocabulary input, and truncation flags.

## Credits

Original model and project: Frederick Ian Aranico. Frontend and serving improvements developed with OpenAI Codex.

Icons: Lucide 0.468.0 (ISC; vendored license header retained). Laptop photograph: [Unsplash image source](https://images.unsplash.com/photo-1517336714731-489689fd1ca8), served locally as an illustrative product image. No Apple affiliation is implied.
