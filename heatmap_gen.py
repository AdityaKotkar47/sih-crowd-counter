import os
import json
import requests

# Configuration
CONFIG_PATH = "config/regions.json"
IMAGE_DIR = "tests/images"
MAP_SVG_PATH = "map.svg"
HEATMAP_OUTPUT_PATH = "heatmap.svg"
API_ENDPOINT = "http://127.0.0.1:8000/predict/"

# Load regions
with open(CONFIG_PATH, "r") as f:
    regions = json.load(f)["regions"]

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
            return data.get("count", 0)
        except requests.exceptions.RequestException as e:
            print(f"Error processing {image_path}: {e}")
            return 0

def generate_heatmap():
    """
    Generates a heatmap based on the crowd counts and overlays it on the SVG map.
    """
    max_count = max(crowd_counts.values())

    with open(MAP_SVG_PATH, "r") as svg_file:
        svg_content = svg_file.read()

    heatmap_rects = ""
    for region in regions:
        count = crowd_counts[region["name"]]
        if count > 0:
            # Normalize the count to a value between 0 and 1
            intensity = count / max_count
            opacity = intensity * 0.8  # Adjust the opacity factor as needed
            color = f"rgba(255, 0, 0, {opacity})"  # Red color with varying opacity

            # Create a new rectangle element for the heatmap
            heatmap_rect = f'<rect x="{region["x"]}" y="{region["y"]}" width="{region["width"]}" height="{region["height"]}" fill="{color}" />'
            heatmap_rects += heatmap_rect + "\n"

    # Insert the heatmap rectangles before the closing </svg> tag
    heatmap_svg = svg_content.replace("</svg>", heatmap_rects + "</svg>")

    with open(HEATMAP_OUTPUT_PATH, "w") as heatmap_file:
        heatmap_file.write(heatmap_svg)

    print(f"Heatmap generated successfully. Saved as {HEATMAP_OUTPUT_PATH}")

def main():
    # Ensure the FastAPI server is running
    try:
        response = requests.get("http://127.0.0.1:8000/")
        if response.status_code != 200:
            print("FastAPI server is not running. Please start it before running this script.")
            return
    except requests.exceptions.ConnectionError:
        print("Failed to connect to FastAPI server. Please ensure it's running on http://127.0.0.1:8000/")
        return

    # Process each image
    for image_file in os.listdir(IMAGE_DIR):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(IMAGE_DIR, image_file)
            region_name = assign_image_to_region(image_path)
            if region_name:
                count = get_crowd_count(image_path)
                crowd_counts[region_name] += count
                print(f"Processed {image_file}: Region={region_name}, Count={count}")
            else:
                print(f"Could not assign {image_file} to any region.")

    # Generate heatmap
    generate_heatmap()

if __name__ == "__main__":
    main()