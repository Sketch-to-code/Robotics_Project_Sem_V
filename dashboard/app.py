from flask import Flask, render_template, jsonify
from detect_cubes import detect_cubes   # import the function we already wrote and tested separately

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/scan')
def scan():
    # Hardcoded path for now — later this becomes "whatever photo the phone camera just captured"
    image_path = "test_images/test1.png"

    # Run the real detection pipeline
    cubes = detect_cubes(image_path, output_dir="detections", debug=True)

    # PROBLEM: 'cubes' contains a 'digit_crop' key holding a raw numpy image array.
    # JSON has no concept of a numpy array — jsonify() would crash trying to serialize it.
    # So we build a *new*, lighter list containing only JSON-safe values before sending it out.
    json_safe_cubes = []
    for c in cubes:
        json_safe_cubes.append({
            "id": c["id"],
            "bbox": c["bbox"],                 # a plain tuple of 4 ints — fine for JSON
            "centroid_px": c["centroid_px"]    # a plain tuple of 2 ints — fine for JSON
            # digit_crop deliberately left out — the browser doesn't need raw pixel arrays,
            # it just needs to know how many cubes and where
        })

    result = {
        "status": "ok",
        "cubes_found": len(json_safe_cubes),
        "cubes": json_safe_cubes
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)