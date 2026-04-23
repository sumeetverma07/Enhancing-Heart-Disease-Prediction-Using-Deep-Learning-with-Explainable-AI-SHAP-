from __future__ import annotations

from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _build_table(rows, column_widths=None):
    table = Table(rows, colWidths=column_widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#102542")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D8E1EB")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def generate_prediction_report(prediction_response: dict) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=16 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    title_style.textColor = colors.HexColor("#102542")
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#1F4E79"),
        spaceAfter=8,
        spaceBefore=8,
    )
    body_style = styles["BodyText"]
    body_style.leading = 14

    prediction = prediction_response["prediction"]
    input_rows = [["Feature", "Value"]]
    for key, value in prediction_response["input_data"].items():
        input_rows.append([key.replace("_", " ").title(), str(value)])

    metrics_rows = [
        ["Field", "Value"],
        ["Generated", prediction_response["timestamp"]],
        ["Primary model", prediction["primary_model"]],
        ["Risk label", prediction["label"]],
        ["Probability", f"{prediction['probability'] * 100:.2f}%"],
        ["Confidence", f"{prediction['confidence'] * 100:.2f}%"],
        ["Explanation model", prediction_response["explanation_model"]],
    ]

    shap_rows = [["Feature", "Scaled value", "SHAP", "Direction"]]
    for item in prediction_response["feature_importance"]:
        shap_rows.append(
            [
                item["feature"],
                str(item["value"]),
                f"{item['shap_value']:.5f}",
                item["direction"],
            ]
        )

    summary_html = (
        f"<b>Prediction summary:</b> {prediction['primary_model']} estimates a "
        f"<b>{prediction['probability'] * 100:.1f}%</b> heart disease risk "
        f"with <b>{prediction['confidence'] * 100:.1f}%</b> confidence. "
        f"The current assessment is <b>{prediction['label']}</b>."
    )

    story = [
        Paragraph("Heart Disease Prediction Report", title_style),
        Paragraph("Patient-ready machine learning summary with SHAP explanation", body_style),
        Spacer(1, 8),
        Paragraph(summary_html, body_style),
        Spacer(1, 12),
        Paragraph("Patient Input Data", subtitle_style),
        _build_table(input_rows, column_widths=[62 * mm, 96 * mm]),
        Spacer(1, 12),
        Paragraph("Prediction Overview", subtitle_style),
        _build_table(metrics_rows, column_widths=[58 * mm, 100 * mm]),
        Spacer(1, 12),
        Paragraph("Top SHAP Feature Impacts", subtitle_style),
        _build_table(shap_rows, column_widths=[46 * mm, 32 * mm, 28 * mm, 52 * mm]),
        Spacer(1, 12),
        Paragraph(
            "Interpretation note: positive SHAP values push the prediction toward higher heart disease risk, "
            "while negative SHAP values push it toward lower risk.",
            body_style,
        ),
    ]

    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf
