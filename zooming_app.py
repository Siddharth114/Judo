from flask import Flask, render_template, request, jsonify, url_for
import json
import base64
from io import BytesIO
from PIL import Image
import zooming
import numpy as np
import cv2
import os

app = Flask(__name__)


def numpy_array_to_base64(frame):
    success, encoded_bytes = cv2.imencode(".jpg", frame)
    base64_string = base64.b64encode(encoded_bytes).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{base64_string}"
    return data_uri


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    output = request.get_json()
    mouse_x = output["mouseX"]
    mouse_y = output["mouseY"]
    vid_width = output["vidWidth"]
    vid_height = output["vidHeight"]
    current_frame = output["frameNumber"]

    stopped_frame = frames[current_frame]
    stopped_boxes = bounding_boxes[current_frame]

    new_x, new_y = zooming.translate_coords(
        mouse_x, mouse_y, vid_width, vid_height, original_width, original_height
    )

    cropped_image, box_to_track = zooming.cropped_img(
        stopped_frame, stopped_boxes, new_x, new_y
    )

    cropped_frames = zooming.generate_cropped_frames(
        frames[current_frame:], bounding_boxes[current_frame:], box_to_track
    )
    output_video_name = "cropped_video_output"
    cropped_frames_path = (
        f"/Users/siddharth/Code/Python/Judo/static/imgs/{output_video_name}.mp4"
    )
    zooming.write_video(
        cropped_frames, cropped_frames_path, original_width, original_height, fps
    )

    processed_image = {
        "vid_path": url_for("static", filename="imgs/cropped_video_output.mp4")
    }

    return jsonify(processed_image)


@app.route("/delete-file", methods=["POST"])
def delete_file():
    file_path = "/Users/siddharth/Code/Python/Judo/static/imgs/cropped_video_output.mp4"  # Replace with the actual file path
    try:
        os.remove(file_path)
        return "File deleted successfully"
    except OSError as e:
        return f"Error deleting file: {e}", 500


if __name__ == "__main__":
    video_path = "starter_images/walking_vid.mp4"
    frames, bounding_boxes, original_width, original_height, fps = zooming.get_frames(
        video_path
    )  # used to run the model and get boxes for the video
    output_video_path = (
        "/Users/siddharth/Code/Python/Judo/static/imgs/walking_vid_output.mp4"
    )
    # zooming.write_video(frames, output_video_path, original_width, original_height, fps)
    app.run(port=8000)
