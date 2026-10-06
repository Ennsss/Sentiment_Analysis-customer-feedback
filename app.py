"""HTTP interface for the original Apple-review sentiment model."""

import csv
import io
from flask import Flask, jsonify, render_template, request
from inference import SentimentEngine


def create_app(engine=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024
    model = engine or SentimentEngine()

    @app.after_request
    def headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        )
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/model")
    def metadata():
        return jsonify(model.metadata())

    @app.post("/api/analyze")
    def analyze():
        if "file" in request.files:
            try:
                content = request.files["file"].read().decode("utf-8-sig")
                reader = csv.DictReader(io.StringIO(content))
                if not reader.fieldnames or "review" not in reader.fieldnames:
                    return jsonify(error="The CSV needs a column named review."), 400
                reviews = [row.get("review") for row in reader]
            except (UnicodeError, csv.Error):
                return jsonify(error="Upload a valid UTF-8 CSV file."), 400
        else:
            payload = request.get_json(silent=True)
            reviews = payload.get("reviews") if isinstance(payload, dict) else None
        if not isinstance(reviews, list) or not 1 <= len(reviews) <= 100:
            return jsonify(error="Provide between 1 and 100 reviews."), 400
        for i, review in enumerate(reviews):
            if not isinstance(review, str) or not review.strip():
                return jsonify(
                    error=f"Review {i + 1} is empty. Remove the empty row and try again."
                ), 400
            if len(review) > 5000:
                return jsonify(
                    error=f"Review {i + 1} exceeds the 5,000-character limit."
                ), 400
        try:
            return jsonify(results=model.analyze([s.strip() for s in reviews]))
        except Exception:
            app.logger.exception("Sentiment model inference failed")
            return jsonify(
                error="The trained model is unavailable. Check the server runtime and model files, then retry. No substitute scores were generated."
            ), 503

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify(
            error="The upload is too large. Use a CSV smaller than 1 MB."
        ), 413

    return app


app = create_app()
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5051)
