from flask import Flask
from flask import render_template
from flask import request
from flask import redirect
from flask import url_for

import os
from werkzeug.utils import secure_filename

from utils.fish_measure import measure_fish

os.makedirs(
    "static/uploads",
    exist_ok=True
)

os.makedirs(
    "static/results",
    exist_ok=True
)

# =========================================================
# APP CONFIG
# =========================================================

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

# buat folder otomatis jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# model YOLO
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "bestnew.pt"
)

# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )

# =========================================================
# PREDICT
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # cek file
    if "image" not in request.files:

        return redirect("/")

    file = request.files["image"]

    # tidak ada file dipilih
    if file.filename == "":

        return redirect("/")

    # =====================================================
    # SAVE UPLOAD
    # =====================================================

    filename = secure_filename(
        file.filename
    )

    upload_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(upload_path)

    # =====================================================
    # RUN FISH MEASUREMENT
    # =====================================================

    try:

        result_image_path, fish_data, summary = measure_fish(
            image_path=upload_path,
            model_path=MODEL_PATH
        )

        return render_template(
            "result.html",
            image=result_image_path,
            fish_data=fish_data,
            summary=summary
        )

    except Exception as e:

        return render_template(
            "result.html",
            image=None,
            fish_data=[],
            summary=None,
            error=str(e)
        )

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=7860,
        debug=True
    )
