from __future__ import annotations

from io import BytesIO

from flask import Flask, jsonify, render_template, request, send_file

from api_service import ValidationError, build_prediction_response, get_schema
from reporting import generate_prediction_report


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template("index.html", schema=list(get_schema()))

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/api/predict")
    def predict():
        payload = request.get_json(silent=True) or {}
        explanation_model = payload.pop("explanation_model", None)
        try:
            result = build_prediction_response(payload, explanation_model=explanation_model)
        except ValidationError as exc:
            return jsonify({"message": "Validation failed.", "errors": exc.errors}), 400
        except Exception as exc:
            return jsonify({"message": str(exc)}), 500
        return jsonify(result)

    @app.post("/download-report")
    def download_report():
        payload = request.get_json(silent=True) or {}
        explanation_model = payload.pop("explanation_model", None)
        try:
            result = build_prediction_response(payload, explanation_model=explanation_model)
        except ValidationError as exc:
            return jsonify({"message": "Validation failed.", "errors": exc.errors}), 400
        except Exception as exc:
            return jsonify({"message": str(exc)}), 500

        pdf_bytes = generate_prediction_report(result)
        return send_file(
            BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name="heart_disease_prediction_report.pdf",
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
