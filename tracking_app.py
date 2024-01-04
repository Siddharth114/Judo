from flask import Flask, render_template, request, jsonify, url_for
import tracking
import os
import datetime
import signal

app = Flask(__name__)

global INPUT_VIDEO_PATH
INPUT_VIDEO_PATH = "starter_images/walking_vid.mp4"

global INFERENCE_VIDEO_PATH
INFERENCE_VIDEO_PATH = "static/imgs/person_tracking_output.mp4"

global ZOOMED_VIDEO_NAME
ZOOMED_VIDEO_NAME = 'zoomed_tracked_output'

global pattern
pattern = f'static/imgs/{ZOOMED_VIDEO_NAME}*.mp4'

#rendering the home page
@app.route("/")
def home():
    data = {'fps': fps}
    return render_template("tracking_app.html", data=data)

@app.route("/process", methods=["POST"])
def process():
    pattern = f'static/imgs/{ZOOMED_VIDEO_NAME}*.mp4'
    tracking.wildcard_delete(pattern)
    
    output = request.get_json()
    mouse_x = output["mouseX"]
    mouse_y = output["mouseY"]
    vid_width = output["vidWidth"]
    vid_height = output["vidHeight"]
    current_frame = output["frameNumber"]

    stopped_frame = frames[current_frame]
    stopped_boxes = bounding_boxes[current_frame]

    new_x, new_y = tracking.translate_coords(
        mouse_x, mouse_y, vid_width, vid_height, original_width, original_height
    )

    clicked_inside_box = tracking.check_inside_box(stopped_boxes, new_x, new_y)
    
    if not clicked_inside_box:
        processed_result = {
            'status': 'False'
        }
        return jsonify(processed_result)
    
    cropped_frames = tracking.generate_cropped_frames(
        frames[current_frame:], bounding_boxes[current_frame:], clicked_inside_box
    )

    unique_suffix = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_video_name = f"tracked_video_output{unique_suffix}"

    global cropped_frames_path
    cropped_frames_path = (
        f"static/imgs/{output_video_name}.mp4"
    )

    tracking.write_video(
        cropped_frames, cropped_frames_path, original_width, original_height, fps
    )

    processed_image = {
        "vid_path": url_for('static', filename=f'imgs/{output_video_name}.mp4'),
        'status': 'True'
    }

    return jsonify(processed_image)

@app.route("/delete-file", methods=["POST"])
def delete_file():
    pattern = 'static/imgs/cropped_video_output*.mp4'
    tracking.wildcard_delete(pattern)
    # removing the file when the user clicks on the refresh button
    try:
        os.remove(cropped_frames_path)
        return "File deleted successfully"
    except OSError as e:
        return f"Error deleting file: {e}", 500
    
def cleanup_on_exit(signum, frame):
    tracking.wildcard_delete(pattern)
    exit(0)

signal.signal(signal.SIGINT, cleanup_on_exit)

if __name__ == '__main__':
    video_path = INPUT_VIDEO_PATH
    global fps
    frames, bounding_boxes, original_width, original_height, fps = tracking.get_frames(
        video_path
    )

    output_video_path = INFERENCE_VIDEO_PATH
    # writing the output video
    tracking.write_video(frames, output_video_path, original_width, original_height, fps)
    app.run(port=8000)