var video = document.getElementById("video");
var output_vid = document.getElementById("output-video");
const frameNumberElement = document.getElementById("frameNumber");
const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");
const reloadButton = document.getElementById("reload");

// console.log(original_fps) // variable from flask app when the site loads

// play/pause button behaviour
function play() {
  if (video.paused) {
    video.play();
    output_vid.play();
  } else {
    video.pause();
    output_vid.pause();
  }
}

function nextFrame() {
  video.currentTime += 1 / original_fps;
  output_vid.currentTime += 1/original_fps;
}

//deleting the generated video with zoomed frames when the user presses the refresh button
function deleteFileAndReload() {
  fetch("/delete-file", {
    method: "POST",
  })
    .then((response) => {
      if (!response.ok) {
        location.reload();
        throw new Error("File deletion failed");
      }
      return response.text();
    })
    .then((data) => {
      console.log("File deleted successfully:", data);
      location.reload();
    })
    .catch((error) => {
      console.error("Error deleting file:", error);
    });
}

// sending current frame number to backend when the user clicks on the video
video.addEventListener("click", (event) => {
  is_paused = video.paused;
  if (!is_paused) {
    video.pause();
    output_vid.pause();
  }
  document.getElementById('loading-screen').style.display = 'block'; // Show loading screen
  document.getElementById('playpause').disabled = true;
  document.getElementById('refresh').disabled = true;
  const currentTime = video.currentTime;
  const fps = original_fps;
  const currentFrame = Math.floor(currentTime * fps);

  var rect = event.target.getBoundingClientRect();
  var x = event.clientX - rect.left;
  var y = event.clientY - rect.top;
  var vid_width = rect.width;
  var vid_height = rect.height;

  $.ajax({
    url: "/process",
    type: "POST",
    contentType: "application/json",
    data: JSON.stringify({
      mouseX: x,
      mouseY: y,
      vidWidth: vid_width,
      vidHeight: vid_height,
      frameNumber: currentFrame,
    }),
    success: function (response) {
      if (response.status=='False') {
        alert('Click inside a bounding box');
      }
      output_vid.src = response.vid_path;
      document.getElementById('loading-screen').style.display = 'none';
      document.getElementById('playpause').disabled = false;
      document.getElementById('refresh').disabled = false;
      if (!is_paused) {
        video.play();
        output_vid.play();
      }
    },
    error: function (error) {
      console.error(error);
      document.getElementById('loading-screen').style.display = 'none';
      document.getElementById('playpause').disabled = false;
      document.getElementById('refresh').disabled = false;
      if (!is_paused) {
        video.play();
        output_vid.play();
      }
    },
  });
});

video.addEventListener("ended", () => {
  reloadButton.style.display = "block";
});

reloadButton.addEventListener("click", () => {
  deleteFileAndReload();
});