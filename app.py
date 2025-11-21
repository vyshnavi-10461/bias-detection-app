from flask import Flask, render_template, request, redirect, url_for, send_file, flash, session
import os, io, csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import joblib
import hashlib

from utils import (
    load_dataset, basic_preprocess, basic_bias_metrics,
    build_feature_pipeline, positive_rate,
    demographic_parity_difference_from_rates,
    disparate_impact_ratio_from_rates, tpr_by_group,
    compute_reweighing_weights, plot_rates
)

from database import get_db, init_db, close_db

app = Flask(__name__)
app.secret_key = "change_this_in_prod"

UPLOAD_FOLDER = "data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        csv_file = request.files.get("file")
        if not csv_file or csv_file.filename == "":
            flash("No file selected", "danger")
            return redirect(url_for("upload_file"))

        filename = csv_file.filename
        path = os.path.join(UPLOAD_FOLDER, filename)
        csv_file.save(path)
        return redirect(url_for("bias", filename=filename))

    return render_template("upload.html")

@app.route("/bias/<filename>")
def bias(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)

    try:
        df, target, sensitive_cols = load_dataset(path)
        metrics = basic_bias_metrics(df, target, sensitive_cols)
    except Exception as e:
        flash(str(e), "danger")
        return redirect(url_for("index"))

    return render_template(
        "bias.html",
        filename=filename,
        head_html=df.to_html(classes="table table-sm", max_rows=1000),
        target=target,
        sensitive_cols=sensitive_cols,
        metrics=metrics
    )

from utils import plot_rates

from database import get_db, init_db, close_db


@app.route("/metrics/<filename>")
def metrics(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        df, target, sensitive_cols = load_dataset(path)
    except Exception as e:
        flash(str(e), "danger")
        return redirect(url_for("index"))

    sens = sensitive_cols[0]
    rates = {g: df[df[sens] == g][target].mean() for g in df[sens].unique()}
    
    dpd = max(rates.values()) - min(rates.values())
    dir_val = min(rates.values()) / max(rates.values()) if max(rates.values()) > 0 else None

    chart = plot_rates(rates, f"Positive Rates by Group ({sens})")

    metrics = {
        "sensitive_column": sens,
        "rates": rates,
        "dpd": dpd,
        "dir": dir_val,
        "chart": chart
    }

    return render_template("metrics.html", filename=filename, metrics=metrics)



@app.route("/model/<filename>", methods=["GET","POST"])
def model_page(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)
    df, target, sensitive_cols = load_dataset(path)

    # Choose sensitive column (prefer gender, then region)
    sens = "gender" if "gender" in sensitive_cols else ("region" if "region" in sensitive_cols else None)
    if sens is None:
        flash("No sensitive column found for modeling.", "danger")
        return redirect(url_for("bias", filename=filename))

    # Prepare feature matrix (drop target + sensitive)
    X_all = df.drop(columns=[target] + sensitive_cols, errors="ignore")
    X_all = X_all.select_dtypes(include=[np.number])  # keep only numeric
    if X_all.shape[1] == 0:
        flash("No numeric features available for modeling.", "danger")
        return redirect(url_for("bias", filename=filename))

    y = df[target].astype(int)

    # Skip stratification if some classes have only 1 member
    stratify_col = y if y.value_counts().min() > 1 else None

    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X_all, y, df[sens], test_size=0.25, stratify=stratify_col, random_state=42
    )

    # Choose model
    model_choice = request.args.get("model", "lr")  # lr or dt
    mitigation = request.args.get("mitigation", None)  # reweigh or none

    if model_choice == "dt":
        clf = DecisionTreeClassifier(random_state=42)
    else:
        clf = LogisticRegression(max_iter=1000)

    # Fit model
    pipe = Pipeline([("clf", clf)])

    if mitigation == "reweigh":
        # Compute sample weights for training
        weights = compute_reweighing_weights(pd.concat([X_train, y_train, sens_train], axis=1), sens, label_col=target)
        pipe.fit(X_train, y_train, clf__sample_weight=weights)
    else:
        pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_test)
    acc = pipe.score(X_test, y_test)

    # Fairness metrics
    rates_pred = positive_rate(y_pred, sens_test.values, sens)
    dpd_model = demographic_parity_difference_from_rates(rates_pred)
    dir_model = disparate_impact_ratio_from_rates(rates_pred)
    tprs = tpr_by_group(y_test.values, y_pred, sens_test.values)

    # Prepare plots
    plot_before = plot_rates(rates_pred, title="Prediction positive rate (by group)")

    return render_template("model.html",
                           filename=filename,
                           acc=acc,
                           dpd_model=dpd_model,
                           dir_model=dir_model,
                           rates_pred=rates_pred,
                           tprs=tprs,
                           privileged=max(rates_pred, key=lambda k: rates_pred[k]),
                           plot_before=plot_before,
                           model_choice=model_choice,
                           mitigation=mitigation)


@app.route("/export/metrics/<filename>")
def export_metrics(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)
    df, target, sensitive_cols = load_dataset(path)
    sens = sensitive_cols[0]
    rates = {g: df[df[sens] == g][target].mean() for g in df[sens].unique()}
    dpd = max(rates.values()) - min(rates.values())
    dir_val = min(rates.values()) / max(rates.values()) if max(rates.values()) > 0 else None

    # Create CSV in-memory
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["sensitive_group", "positive_rate"])
    for g, r in rates.items():
        writer.writerow([g, r])
    writer.writerow([])
    writer.writerow(["DPD", dpd])
    writer.writerow(["DIR", dir_val])
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()), mimetype="text/csv",
                     as_attachment=True, download_name=f"{filename}_metrics.csv")

@app.route("/export/data/<filename>")
def export_data(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename), as_attachment=True)



@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        password_hash = hashlib.sha256(password.encode()).hexdigest()

        db = get_db()
        try:
            db.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, password_hash)
            )
            db.commit()
            flash("Registration successful!", "success")
            return redirect(url_for("login"))
        except:
            flash("Username already exists!", "danger")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        flash("You are already logged in.", "info")
        return redirect(url_for("index"))

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if not username or not password:
            flash("Please enter both username and password", "danger")
            return render_template("login.html")

        password_hash = hashlib.sha256(password.encode()).hexdigest()
        db = get_db()

        user = db.execute(
            "SELECT * FROM users WHERE username = ? AND password_hash = ?",
            (username, password_hash)
        ).fetchone()

        if user:
            session.permanent = True
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            flash("Login successful!", "success")
            return redirect(url_for("index"))

        flash("Invalid username or password", "danger")

    return render_template("login.html")


def log_action(user_id, filename, action):
    db = get_db()
    db.execute(
        "INSERT INTO performance_logs (user_id, filename, action) VALUES (?, ?, ?)",
        (user_id, filename, action)
    )
    db.commit()

from database import get_db, init_db
import hashlib

@app.teardown_appcontext
def teardown(exception):
    close_db(exception)

if __name__ == "__main__":
    with app.app_context():
        init_db()      # NOW works safely inside context
    app.run(debug=True)



