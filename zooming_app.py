from flask import Flask, render_template, request, jsonify, url_for
import base64
import zooming
import cv2
import os
import time

app = Flask(__name__)


# converting a numpy array of an image to base64 encoding
def numpy_array_to_base64(frame):
    # encoding the numpy array of the frame to jpg format
    success, encoded_bytes = cv2.imencode(".jpg", frame)
    # encoding it into base64
    base64_string = base64.b64encode(encoded_bytes).decode("utf-8")
    # formatting it for js
    data_uri = f"data:image/jpeg;base64,{base64_string}"
    return data_uri


#rendering the home page
@app.route("/")
def home():
    data = {'fps': fps}
    return render_template("index.html", data=data)


# creating the video with cropped frames when the user clicks on a person
@app.route("/process", methods=["POST"])
def process():
    # getting the json variable of the response when the user clicks on the video
    output = request.get_json()
    mouse_x = output["mouseX"]
    mouse_y = output["mouseY"]
    vid_width = output["vidWidth"]
    vid_height = output["vidHeight"]
    current_frame = output["frameNumber"]

    # getting the frame and bounding boxes of that frame at which the user clicked the video
    stopped_frame = frames[current_frame]
    stopped_boxes = bounding_boxes[current_frame]

    # translating coordinates from small video frame to original size
    new_x, new_y = zooming.translate_coords(
        mouse_x, mouse_y, vid_width, vid_height, original_width, original_height
    )

    # getting the box inside which the user clicked
    cropped_image, box_to_track, cropped_status = zooming.cropped_img(
        stopped_frame, stopped_boxes, new_x, new_y
    )
    
    if not cropped_status:
        processed_result = {
            'status': 'False'
        }
        return jsonify(processed_result)

    # generating cropped frames given the box to track across the video
    cropped_frames = zooming.generate_cropped_frames(
        frames[current_frame:], bounding_boxes[current_frame:], box_to_track
    )

    # saving output video with timestamp to avoid browser caching
    unique_suffix = str(time.time())
    output_video_name = f"cropped_video_output{unique_suffix}"
    global cropped_frames_path
    # setting the path of the saved video
    cropped_frames_path = (
        f"/Users/siddharth/Code/Python/Judo/static/imgs/{output_video_name}.mp4"
    )
    # writing the zoomed frames into a video
    zooming.write_video(
        cropped_frames, cropped_frames_path, original_width, original_height, fps
    )

    # sending a response to the js file with the path to the video
    processed_image = {
        "vid_path": url_for('static', filename=f'imgs/{output_video_name}.mp4'),
        'status': 'True'
    }

    return jsonify(processed_image)


# deleting the created video after the user presses the refresh button
@app.route("/delete-file", methods=["POST"])
def delete_file():
    # removing the file when the user clicks on the refresh button
    try:
        os.remove(cropped_frames_path)
        return "File deleted successfully"
    except OSError as e:
        return f"Error deleting file: {e}", 500


if __name__ == "__main__":
    # video path of the video to be used
    video_path = "starter_images/walking_vid.mp4"
    # getting data after the yolo v8 mdoel is run on the video
    global fps
    frames, bounding_boxes, original_width, original_height, fps = zooming.get_frames(
        video_path
    )
    # path of the output video
    output_video_path = (
        "/Users/siddharth/Code/Python/Judo/static/imgs/walking_vid_output.mp4"
    )
    # writing the output video
    # zooming.write_video(frames, output_video_path, original_width, original_height, fps)
    app.run(port=8000)
