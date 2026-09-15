import os
from flask import Flask, render_template, jsonify, request, send_from_directory
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from detect_cubes import detect_cubes
from sorter import plan_sort, HOME_POS

app = Flask(__name__)

# Base directory = wherever app.py itself lives, regardless of cwd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "digit_model", "digit_cnn_model.keras")
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "..", "digit_model", "test_images")
DETECTIONS_DIR = os.path.join(BASE_DIR, "detections")

# Which image /api/scan and /api/plan use. Swap this out once the live
# camera feed is wired in (step: ESP32 + camera -> arm coordinate calibration).
DEFAULT_IMAGE = "test1.png"

model = load_model(MODEL_PATH)

# In-memory cache of the last scan, so /api/plan doesn't need to re-run
# detection + the CNN every time the user just changes the sort mode.
_last_scan_cache = {"image_path": None, "cubes": None}


def predict_crop(crop_bgr):
    img = Image.fromarray(crop_bgr).convert("L").resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 64, 64, 1)
    prediction = model.predict(img_array, verbose=0)
    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    return digit, confidence


def run_scan(image_filename):
    """Runs detection + CNN prediction, caches result, returns cube list."""
    image_path = os.path.join(TEST_IMAGES_DIR, image_filename)
    raw_cubes = detect_cubes(image_path, output_dir=DETECTIONS_DIR, debug=True)

    cubes = []
    for c in raw_cubes:
        digit, confidence = predict_crop(c["digit_crop"])
        cubes.append({
            "id": c["id"],
            "bbox": c["bbox"],
            "centroid_px": c["centroid_px"],
            "predicted_digit": digit,
            "confidence": round(confidence * 100, 2),
        })

    _last_scan_cache["image_path"] = image_path
    _last_scan_cache["cubes"] = cubes
    return cubes, image_path


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/scan')
def scan():
    image_filename = request.args.get("image", DEFAULT_IMAGE)
    try:
        cubes, image_path = run_scan(image_filename)
    except Exception as e:
        print(f"Scan error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 400

    result = {
        "status": "ok",
        "cubes_found": len(cubes),
        "cubes": cubes,
        "annotated_image_url": "/detections/debug_annotated.png",
        "original_image_url": f"/test_images/{os.path.basename(image_path)}",
    }
    return jsonify(result)


@app.route('/api/plan')
def plan():
    """
    Query params:
      mode          - ascending | descending | odd_even | group_by_digit | custom
      custom_order  - e.g. "5,3,1,0,2" (only used when mode == custom)
      image         - optional filename override
      rescan        - "1" to force a fresh scan instead of using cached cubes
    """
    mode = request.args.get("mode", "ascending")
    custom_order = request.args.get("custom_order", "")
    image_filename = request.args.get("image", DEFAULT_IMAGE)
    force_rescan = request.args.get("rescan") == "1"

    if force_rescan or _last_scan_cache["cubes"] is None:
        cubes, _ = run_scan(image_filename)
    else:
        cubes = _last_scan_cache["cubes"]

    try:
        result = plan_sort(cubes, mode, custom_order, start_pos=HOME_POS)
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400

    result["status"] = "ok"
    result["cubes_found"] = len(cubes)
    return jsonify(result)


@app.route('/api/execute')
def execute():
    """
    Stub: replays the plan for the currently selected mode as a simulated
    execution. Once the ESP32 + PCA9685 firmware is up, replace the
    `# TODO send to arm` block with the real serial/HTTP call per step.
    """
    mode = request.args.get("mode", "ascending")
    custom_order = request.args.get("custom_order", "")

    if _last_scan_cache["cubes"] is None:
        return jsonify({"status": "error", "message": "No scan available. Call /api/scan first."}), 400

    result = plan_sort(_last_scan_cache["cubes"], mode, custom_order, start_pos=HOME_POS)

    executed_steps = []
    for step in result["steps"]:
        # TODO send to arm: move to step['pick_px'] (converted to arm coords),
        # close gripper, move to step['place_px'], open gripper.
        executed_steps.append({**step, "status": "simulated"})

    return jsonify({
        "status": "simulated",
        "mode": mode,
        "steps": executed_steps,
        "total_distance": result["total_distance"],
        "note": "Arm firmware not connected yet — this is a dry run of the move sequence.",
    })


@app.route('/detections/<path:filename>')
def serve_detection(filename):
    return send_from_directory(DETECTIONS_DIR, filename)


@app.route('/test_images/<path:filename>')
def serve_test_image(filename):
    return send_from_directory(TEST_IMAGES_DIR, filename)

@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"status":"error","message": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error","message": "No selected file"}), 400

    if file:
        save_path = os.path.join(TEST_IMAGES_DIR, "test1.png")
        file.save(save_path)
        return jsonify({"status":"ok","message": "Image uploaded successfully !"})
    


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)