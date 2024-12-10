import os
import json
import requests # type: ignore
from flask import Flask, send_from_directory, abort, send_file, jsonify
from flask_cors import CORS

# Configuration
CONFIG_PATH = "config/regions.json"
IMAGE_DIR = "tests/images"
MAP_SVG_PATH = "map.svg"  # Update path to where map.svg is stored
HEATMAP_OUTPUT_PATH = "assets/heatmap.svg" # Update to a valid path
API_ENDPOINT = "http://127.0.0.1:8000/predict/"

# Create assets directory if it doesn't exist
os.makedirs("assets", exist_ok=True)

# Load regions
try:
    with open(CONFIG_PATH, "r") as f:
        regions = json.load(f)["regions"]
    print(f"Loaded regions from {CONFIG_PATH}")
except FileNotFoundError:
    print(f"Error: {CONFIG_PATH} not found.")
    exit(1)
except json.JSONDecodeError:
    print(f"Error: Failed to parse {CONFIG_PATH}. Ensure it is valid JSON.")
    exit(1)

# Initialize crowd counts
crowd_counts = {region["name"]: 0 for region in regions}

def assign_image_to_region(image_path):
    """
    Assigns an image to a region based on its filename.
    Assumes that the filename contains the region name.
    Example: restroom1_image1.png
    """
    filename = os.path.basename(image_path).lower()
    for region in regions:
        if region["name"].replace(" ", "").lower() in filename:
            return region["name"]
    return None

def get_crowd_count(image_path):
    """
    Sends the image to the FastAPI /predict/ endpoint and retrieves the crowd count.
    """
    with open(image_path, "rb") as image_file:
        files = {"file": ("image.jpg", image_file, "image/jpeg")}
        try:
            response = requests.post(API_ENDPOINT, files=files)
            response.raise_for_status()
            data = response.json()
            count = data.get("count", 0)
            print(f"Received count {count} for {image_path}")
            return count
        except requests.exceptions.RequestException as e:
            print(f"Error processing {image_path}: {e}")
            return 0
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON response for {image_path}")
            return 0

def generate_heatmap():
    """
    Generates a heatmap based on the crowd counts and overlays it on the SVG map.
    """
    try:
        with open(MAP_SVG_PATH, "r") as svg_file:
            svg_content = svg_file.read()
        print(f"Loaded SVG map from {MAP_SVG_PATH}")
    except FileNotFoundError:
        print(f"Error: {MAP_SVG_PATH} not found.")
        return

    heatmap_rects = ""
    for region in regions:
        count = crowd_counts.get(region["name"], 0)
        if count > 0:
            # Determine color based on count
            if count <= 7:
                color = "#00FF00"  # Green
                opacity = 0.6
            elif count <= 10:
                color = "#FFA500"  # Orange
                opacity = 0.6
            else:
                color = "#FF0000"  # Red
                opacity = 0.6

            # Create a new rectangle element for the heatmap
            heatmap_rect = (
                f'<rect x="{region["x"]}" y="{region["y"]}" width="{region["width"]}" '
                f'height="{region["height"]}" fill="{color}" fill-opacity="{opacity}" />'
            )
            heatmap_rects += heatmap_rect + "\n"
            print(f"Added heatmap rectangle for {region['name']}: Color={color}, Opacity={opacity}")

    # Insert the heatmap rectangles before the closing </svg> tag
    if "</svg>" in svg_content:
        heatmap_svg = svg_content.replace("</svg>", heatmap_rects + "</svg>")
    else:
        print("Error: </svg> tag not found in the SVG file.")
        return

    try:
        with open(HEATMAP_OUTPUT_PATH, "w") as heatmap_file:
            heatmap_file.write(heatmap_svg)
        print(f"Heatmap generated successfully. Saved as {HEATMAP_OUTPUT_PATH}")
    except Exception as e:
        print(f"Error saving heatmap SVG: {e}")

def main():
    # Ensure the FastAPI server is running
    try:
        response = requests.get("http://127.0.0.1:8000/")
        if response.status_code != 200:
            print("FastAPI server is not running properly. Please check the server.")
            return
        print("Connected to FastAPI server successfully.")
    except requests.exceptions.ConnectionError:
        print("Failed to connect to FastAPI server. Please ensure it's running on http://127.0.0.1:8000/")
        return

    # Process each image
    processed_images = 0
    for image_file in os.listdir(IMAGE_DIR):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(IMAGE_DIR, image_file)
            region_name = assign_image_to_region(image_path)
            if region_name:
                count = get_crowd_count(image_path)
                crowd_counts[region_name] += count
                print(f"Processed {image_file}: Region={region_name}, Count={count}")
                processed_images += 1
            else:
                print(f"Could not assign {image_file} to any region.")

    if processed_images == 0:
        print("No images were processed. Please check the IMAGE_DIR and filenames.")
    else:
        print(f"Processed {processed_images} images successfully.")

    # Generate heatmap
    generate_heatmap()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Serve the heatmap.svg file
@app.route('/heatmap', methods=['GET'])
def serve_heatmap():
    if os.path.exists(HEATMAP_OUTPUT_PATH):
        return send_file(HEATMAP_OUTPUT_PATH, mimetype='image/svg+xml')
    else:
        return jsonify({"detail": "Heatmap SVG file not found"}), 404

if __name__ == "__main__":
    main()
    # Start the Flask server
    app.run(host='0.0.0.0', port=8000)
