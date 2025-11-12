# app.py
import os
import io
import json
import tempfile
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from pymongo import MongoClient
from analysis import parse_and_clean, generate_insights, header_automap
from dotenv import load_dotenv

# ML model (optional)
try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# PDF & charts
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

import matplotlib
matplotlib.use("Agg")   # headless backend
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import certifi

# -------------------- CONFIGURATION -------------------- #
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret")

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("Set MONGO_URI in .env")

# Use certifi to avoid SSL handshake problems with Atlas
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client.get_default_database()
uploads_col = db["uploads"]
reports_col = db["analysis_reports"]

ALLOWED_EXTENSIONS = {"csv", "xls", "xlsx", "txt"}
STATIC_DIR = os.path.join(app.root_path, "static")
os.makedirs(STATIC_DIR, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# -------------------- HELPERS -------------------- #

def safe_read_csv_bytes(content_bytes):
    """
    Try reading CSV bytes with utf-8, fallback to latin1 and ISO-8859-1.
    Returns a pandas.DataFrame or raises the last exception.
    """
    last_exc = None
    for enc in ("utf-8", "latin1", "ISO-8859-1"):
        try:
            return pd.read_csv(io.BytesIO(content_bytes), encoding=enc, engine="python")
        except Exception as e:
            last_exc = e
    raise last_exc


def save_price_vs_volume_image(df_clean, upload_id):
    """
    Generate and save a Price vs Sales Volume PNG to static directory and return its path.
    Requires df_clean to have Price_Set and Sales_Volume columns.
    """
    img_name = f"price_vs_volume_{upload_id}.png"
    img_path = os.path.join(STATIC_DIR, img_name)
    try:
        X = df_clean["Price_Set"].values.reshape(-1, 1)
        y = df_clean["Sales_Volume"].values
        if SKLEARN_AVAILABLE and len(df_clean) > 2:
            model = LinearRegression()
            model.fit(X, y)
            sort_idx = np.argsort(df_clean["Price_Set"].values)
            xs_sorted = df_clean["Price_Set"].values[sort_idx]
            ys_pred = model.predict(xs_sorted.reshape(-1, 1))
        else:
            model = None
            ys_pred = None

        plt.figure(figsize=(6, 4))
        plt.scatter(df_clean["Price_Set"], df_clean["Sales_Volume"], alpha=0.6)
        if ys_pred is not None:
            plt.plot(xs_sorted, ys_pred, color="red")
        plt.xlabel("Price")
        plt.ylabel("Sales Volume")
        plt.title("Price vs Sales Volume")
        plt.tight_layout()
        plt.savefig(img_path, dpi=120)
        plt.close()
        return img_path
    except Exception:
        # ensure no matplotlib leftover
        plt.close()
        return None


def create_charts_images(report):
    """
    Create matplotlib chart images from report dict.
    Returns list of tuples: (title, io.BytesIO)
    """
    images = []

    # Revenue over time
    ts = report.get("time_series", [])
    if ts:
        df_ts = pd.DataFrame(ts)
        if "Date" in df_ts.columns and "Revenue" in df_ts.columns:
            try:
                df_ts["Date"] = pd.to_datetime(df_ts["Date"])
                df_ts = df_ts.sort_values("Date")
                plt.figure(figsize=(8, 3.5))
                plt.plot(df_ts["Date"].dt.strftime("%Y-%m-%d"), df_ts["Revenue"], marker='o', linestyle='-')
                plt.xticks(rotation=45, ha='right')
                plt.title("Revenue Over Time")
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=150)
                buf.seek(0)
                images.append(("Revenue Over Time", buf))
                plt.close()
            except Exception:
                plt.close()

    # By category
    bc = report.get("by_category", [])
    if bc:
        df_bc = pd.DataFrame(bc)
        if "Product_Category" in df_bc.columns and "Revenue" in df_bc.columns:
            try:
                plt.figure(figsize=(8, 3.5))
                plt.bar(df_bc["Product_Category"].astype(str), df_bc["Revenue"])
                plt.xticks(rotation=45, ha='right')
                plt.title("Revenue by Category")
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=150)
                buf.seek(0)
                images.append(("Revenue by Category", buf))
                plt.close()
            except Exception:
                plt.close()

    # Top products
    tp = report.get("top_products", [])
    if tp:
        df_tp = pd.DataFrame(tp)
        if "Product_ID" in df_tp.columns and "Revenue" in df_tp.columns:
            try:
                df_tp_sorted = df_tp.sort_values("Revenue", ascending=False).head(10)
                plt.figure(figsize=(8, 3.5))
                plt.bar(df_tp_sorted["Product_ID"].astype(str), df_tp_sorted["Revenue"])
                plt.xticks(rotation=45, ha='right')
                plt.title("Top Products (by Revenue)")
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=150)
                buf.seek(0)
                images.append(("Top Products", buf))
                plt.close()
            except Exception:
                plt.close()

    return images


# watermark helper for PDF
def _draw_watermark(canvas_obj: Canvas, doc):
    canvas_obj.saveState()
    canvas_obj.setFont("Helvetica", 60)
    # alpha parameter not supported directly in older reportlab: set fill color lighter
    canvas_obj.setFillColorRGB(0.85, 0.85, 0.85)
    w, h = A4
    canvas_obj.translate(w / 2, h / 2)
    canvas_obj.rotate(45)
    canvas_obj.drawCentredString(0, 0, "CONFIDENTIAL")
    canvas_obj.restoreState()


# -------------------- ROUTES -------------------- #

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        f = request.files.get("file")
        raw_text = request.form.get("raw_text", "").strip()
        start_date = request.form.get("start_date")
        end_date = request.form.get("end_date")

        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            content = f.read()
            ext = filename.rsplit(".", 1)[1].lower()
            try:
                if ext in ["csv", "txt"]:
                    df = safe_read_csv_bytes(content)
                else:
                    # For xls/xlsx try openpyxl engine
                    df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
            except Exception as e:
                flash(f"Error reading file: {e}", "danger")
                return redirect(url_for("upload"))
        elif raw_text:
            try:
                df = pd.read_csv(io.StringIO(raw_text), engine="python")
                filename = "pasted_text.csv"
            except Exception as e:
                flash(f"Error reading pasted CSV: {e}", "danger")
                return redirect(url_for("upload"))
        else:
            flash("Please upload a file or paste CSV text.", "warning")
            return redirect(url_for("upload"))

        # Auto-map headers and parse
        header_map, warnings = header_automap(list(df.columns))
        records, problems = parse_and_clean(df, header_map)
        if not records:
            flash("No valid rows after parsing. Check file format.", "danger")
            return redirect(url_for("upload"))

        df_clean = pd.DataFrame(records)

        # Apply optional date filter
        if "Date" in df_clean.columns and start_date and end_date:
            try:
                df_clean["Date"] = pd.to_datetime(df_clean["Date"], errors="coerce")
                mask = (df_clean["Date"] >= pd.to_datetime(start_date)) & (df_clean["Date"] <= pd.to_datetime(end_date))
                df_clean = df_clean.loc[mask]
            except Exception:
                # if parsing fails, continue with full dataset but warn
                flash("Warning: date filtering failed; continuing with full dataset.", "warning")

        # AI-Powered Insights (price vs volume)
        ai_insight = {}
        price_img_path = None
        if "Price_Set" in df_clean.columns and "Sales_Volume" in df_clean.columns and len(df_clean) >= 3 and SKLEARN_AVAILABLE:
            try:
                X = df_clean[["Price_Set"]].values
                y = df_clean["Sales_Volume"].values
                model = LinearRegression()
                model.fit(X, y)
                slope = float(model.coef_[0])
                intercept = float(model.intercept_)
                corr = float(np.corrcoef(df_clean["Price_Set"], df_clean["Sales_Volume"])[0, 1])
                median_price = float(np.median(df_clean["Price_Set"]))
                # heuristic optimal price (very simple): median minus small adjustment based on slope
                optimal_price = round(median_price - slope * 0.1, 2)

                ai_insight = {
                    "correlation": round(corr, 3),
                    "trend": "Negative (Price ↑, Volume ↓)" if slope < 0 else "Positive (Price ↑, Volume ↑)",
                    "optimal_price": optimal_price
                }

                # save price vs volume image per-upload
                price_img_path = save_price_vs_volume_image(df_clean, "u" + datetime.utcnow().strftime("%Y%m%d%H%M%S"))
            except Exception:
                ai_insight = {"message": "AI insight generation failed for this dataset."}
        else:
            # if sklearn not available or columns missing
            if not SKLEARN_AVAILABLE:
                ai_insight = {"message": "scikit-learn not installed; AI insights disabled."}
            else:
                ai_insight = {"message": "Insufficient columns (Price_Set and Sales_Volume) or too few rows for AI insights."}

        # Save upload metadata and rows
        upload_doc = {
            "filename": filename,
            "upload_time": datetime.utcnow(),
            "original_headers": list(df.columns),
            "header_map": header_map,
            "warnings": warnings + problems,
            "row_count": len(df_clean),
            "ai_insight": ai_insight,
        }
        upload_id = uploads_col.insert_one(upload_doc).inserted_id

        # store cleaned rows under collection rows_<id>
        rows = []
        for r in df_clean.to_dict(orient="records"):
            r["_upload_id"] = str(upload_id)
            # ensure Date serialized as ISO string for storage
            if "Date" in r and pd.notnull(r["Date"]):
                r["Date"] = (pd.to_datetime(r["Date"]).isoformat())
            rows.append(r)
        if rows:
            db[f"rows_{upload_id}"].insert_many(rows)

        # generate standard report
        report = generate_insights(df_clean)

        report_doc = {
            "upload_id": str(upload_id),
            "created_at": datetime.utcnow(),
            "report": report,
            "ai_insight": ai_insight,
        }
        reports_col.insert_one(report_doc)

        # render dashboard
        return render_template("dashboard.html", report=report, ai_insight=ai_insight, upload_id=str(upload_id))

    return render_template("upload.html")


@app.route("/report/<upload_id>")
def view_report(upload_id):
    r = reports_col.find_one({"upload_id": str(upload_id)}, sort=[("created_at", -1)])
    if not r:
        flash("Report not found.", "warning")
        return redirect(url_for("index"))
    return render_template("dashboard.html", report=r["report"], ai_insight=r.get("ai_insight", {}), upload_id=str(upload_id))


@app.route("/api/product/<upload_id>/<product_id>")
def product_detail(upload_id, product_id):
    rows_col = db[f"rows_{upload_id}"]
    data = list(rows_col.find({"Product_ID": product_id}, {"_id": 0}).sort("Date", 1))
    return jsonify(data)


@app.route("/price_vs_sales/<upload_id>")
def price_vs_sales_image(upload_id):
    """
    Serve price vs sales image generated earlier (or generate on-demand from stored rows)
    """
    # try to find static file matching upload id prefix
    # if none, try building from DB rows
    static_candidates = [f for f in os.listdir(STATIC_DIR) if f.startswith(f"price_vs_volume_{upload_id}")]
    if static_candidates:
        return send_file(os.path.join(STATIC_DIR, static_candidates[0]), mimetype="image/png")

    # otherwise try generate from DB rows
    rows_col = db[f"rows_{upload_id}"]
    rows = list(rows_col.find({}, {"_id": 0}))
    if not rows:
        return jsonify({"error": "No data for this upload"}), 404
    df = pd.DataFrame(rows)
    if "Price_Set" not in df.columns or "Sales_Volume" not in df.columns:
        return jsonify({"error": "Required columns not found"}), 400

    path = save_price_vs_volume_image(df, upload_id)
    if path and os.path.exists(path):
        return send_file(path, mimetype="image/png")
    return jsonify({"error": "Could not generate image"}), 500


@app.route("/download/<upload_id>")
def download_report(upload_id):
    r = reports_col.find_one({"upload_id": str(upload_id)})
    if not r:
        flash("Report not found.", "warning")
        return redirect(url_for("index"))

    report = r.get("report", {})
    ai_insight = r.get("ai_insight", {})
    chart_images = create_charts_images(report)

    # Build PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf_path = tmp_pdf.name

    doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    elements = []
    styles = getSampleStyleSheet()

    # top logo if exists
    logo_path = os.path.join(STATIC_DIR, "logo.png")
    if os.path.exists(logo_path):
        try:
            elements.append(Image(logo_path, width=4*cm, height=4*cm))
            elements.append(Spacer(1, 0.2*cm))
        except Exception:
            pass

    elements.append(Paragraph("<b>Dynamic Pricing Analysis Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.2*cm))

    # Summary table
    summary = report.get("summary", {})
    summary_rows = [
        ["Total Revenue", f"{summary.get('total_revenue', 'N/A')}"],
        ["Total Volume", f"{summary.get('total_volume', 'N/A')}"],
        ["Products Analyzed", f"{summary.get('num_products', 'N/A')}"],
    ]
    if "date_range" in summary and isinstance(summary["date_range"], (list, tuple)):
        summary_rows.append(["Date Range", f"{summary['date_range'][0]} to {summary['date_range'][1]}"])
    t = Table(summary_rows, hAlign="LEFT", colWidths=[5*cm, 10*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f4f8")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.5*cm))

    # Charts
    for desc, buf in chart_images:
        try:
            img_reader = ImageReader(buf)
            elements.append(Paragraph(f"<b>{desc}</b>", styles["Heading3"]))
            elements.append(Spacer(1, 0.1*cm))
            elements.append(Image(img_reader, width=16*cm, height=6*cm))
            elements.append(Spacer(1, 0.3*cm))
        except Exception:
            pass

    # Top products table
    elements.append(Paragraph("<b>Top Products</b>", styles["Heading3"]))
    tp = report.get("top_products", [])
    if tp:
        table_data = [["Product ID", "Revenue", "Avg Volume", "Avg Price"]]
        for p in tp:
            table_data.append([
                str(p.get("Product_ID", "")),
                f"{p.get('Revenue', '')}",
                f"{p.get('Sales_Volume', '')}",
                f"{p.get('Price_Set', '')}"
            ])
        prod_table = Table(table_data, hAlign="LEFT", colWidths=[4*cm, 4*cm, 4*cm, 4*cm])
        prod_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0078D7")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ]))
        elements.append(prod_table)
    else:
        elements.append(Paragraph("No top product data available.", styles["Normal"]))

    elements.append(Spacer(1, 0.4*cm))
    elements.append(PageBreak())

    # Recommendations
    elements.append(Paragraph("<b>Recommendations</b>", styles["Heading2"]))
    for rec in report.get("recommendations", []):
        rec_text = f"<b>{rec.get('product', '')}</b>: {rec.get('action','')} — {rec.get('justification','')}"
        elements.append(Paragraph(rec_text, styles["Normal"]))
        elements.append(Spacer(1, 0.1*cm))

    elements.append(PageBreak())

    # AI Insights section
    elements.append(Paragraph("<b>AI-Powered Insights</b>", styles["Heading2"]))
    if ai_insight:
        if "correlation" in ai_insight:
            elements.append(Paragraph(f"Correlation (Price vs Sales Volume): {ai_insight.get('correlation')}", styles["Normal"]))
            elements.append(Paragraph(f"Trend: {ai_insight.get('trend')}", styles["Normal"]))
            elements.append(Paragraph(f"Suggested Optimal Price: {ai_insight.get('optimal_price')}", styles["Normal"]))
            # attempt to include saved image for this upload (first file matching pattern)
            static_candidates = [f for f in os.listdir(STATIC_DIR) if f.startswith("price_vs_volume_")]
            if static_candidates:
                try:
                    elements.append(Spacer(1, 0.2*cm))
                    elements.append(Image(os.path.join(STATIC_DIR, static_candidates[-1]), width=14*cm, height=8*cm))
                except Exception:
                    pass
        else:
            elements.append(Paragraph(str(ai_insight.get("message", ai_insight)), styles["Normal"]))
    else:
        elements.append(Paragraph("No AI insights available for this report.", styles["Normal"]))

    # Build PDF with watermark
    try:
        doc.build(elements, onFirstPage=_draw_watermark, onLaterPages=_draw_watermark)
    except TypeError:
        # older reportlab may not accept onLaterPages param
        doc.build(elements, onFirstPage=_draw_watermark)

    return send_file(pdf_path, as_attachment=True, download_name=f"report_{upload_id}.pdf")


# -------------------- RUN -------------------- #
if __name__ == "__main__":
    print("Starting Dynamic Pricing Dashboard...")
    app.run(debug=True)
