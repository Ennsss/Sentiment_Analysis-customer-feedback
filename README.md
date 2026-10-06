# Signal: Customer Feedback Intelligence

An analysis workspace for the original Apple-product review sentiment classifier. The new frontend makes its actual model outputs easier to inspect, compare, and export without changing the trained weights.

Live application: https://signal-sentiment-frederick.vercel.app

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

The original `requirements.txt` records the older environment. The isolated `requirements-web.txt` supplies a Python-compatible serving runtime. Legacy Keras loading and the original tokenizer pickle module are handled explicitly in `inference.py`. scikit-learn 1.6.1 loads the original 1.6.0 label encoder with a version warning; the real-model test verifies all five original classes and finite scores. Python 3.13 also produces upstream TensorFlow/gast deprecation warnings. Neither warning is an evaluation of model accuracy.

## CSV and API

A UTF-8 CSV must contain a `review` column. Quotes, commas, and multiline quoted reviews are parsed with Python's standard CSV parser. Maximum upload: 1 MB; maximum review length: 5,000 characters. Empty rows are rejected with their row number. A sample is available in `static/sample-reviews.csv`.

```http
POST /api/analyze
Content-Type: application/json
X-Signal-Request: 1

{"reviews":["The battery lasts all day and the camera is excellent."]}
```

`POST /api/analyze` also accepts multipart `file`. `GET /api/model` returns serving metadata. A missing or incompatible model returns HTTP 503 and an actionable UI error, not fake scores.

## Interpretation and Limits

The model processes the original saved tokenization and post-pads/truncates to 350 tokens. Class names come from the saved encoder. The displayed artifact version is the first 12 characters of its SHA-256 hash.

Review flags are transparent interface heuristics: a top score below 70%, vocabulary coverage below 50%, an unrecognized input, or truncation. They are **not calibrated uncertainty estimates or validated operational thresholds**. Positive/negative summary shares include their respective "very" classes.

No held-out evaluation dataset or metrics artifact is present, so the interface makes no new accuracy, lift, ROI, or production-performance claim. The original domain is Apple product reviews. This illustrates unstructured-text preparation, classification, inspection, and operational handoff; it is not a collections, credit-risk, or propensity-to-pay model. A new use case needs labeled domain data and validation first.

## Privacy and Deployment

Inference runs on this application's server, not an external AI service. Review text is held in memory for the request. Session history is kept only in the current browser tab, is not saved to browser storage, and disappears on reload. Uploaded files are not saved. No submitted text is logged by application code.

This public portfolio app has no user authentication or shared review database. Use non-sensitive sample text, not confidential customer records. On Vercel, inference requests require a custom header and reject cross-site origins. Per-worker, per-IP limits allow six analyses per minute and 60 per hour. This in-memory throttle resets when a worker restarts and is not a distributed quota; add a shared limiter, access control, and a bounded job queue before a larger rollout. Inference is serialized per worker to avoid simultaneous access to the legacy model. Never load pickled model artifacts from untrusted sources.

## Vercel

The deployment uses Python 3.12 and the dependencies in `pyproject.toml`, not the legacy training environment. `build_web.py` retrieves the original public Git LFS model from a pinned repository commit and verifies its full SHA-256 digest before deployment. Weights are never retrained or replaced. Only browser assets are published under `public/static`; model artifacts stay in the server bundle.

The TensorFlow bundle requires Vercel's Large Functions beta, enabled with `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` in the production environment. No paid plan upgrade is selected. Free-tier quotas still apply; first inference on a cold worker can be slow. Run `vercel deploy --prod` from this folder. Git-based preview deployments need the same beta setting to build the model runtime.

## Verify

```sh
python -m pytest tests -q
# To include real model inference:
# PowerShell: $env:RUN_MODEL_TESTS='1'
# macOS/Linux: export RUN_MODEL_TESTS=1
python -m pytest tests -q
```

The API tests cover CSV parsing, blank/invalid input, size limits, model failure, cross-origin rejection, the production header, and per-worker throttling. The opt-in integration test runs the real artifact and checks all five scores, version, out-of-vocabulary input, and truncation flags.

## Credits

Original model and project: Frederick Ian Aranico. Frontend and serving improvements developed with OpenAI Codex.

Icons: Lucide 0.468.0 (ISC; vendored license header retained). Laptop photograph: [Unsplash image source](https://images.unsplash.com/photo-1517336714731-489689fd1ca8), served locally as an illustrative product image. No Apple affiliation is implied.
