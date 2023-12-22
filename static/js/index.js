var video = document.getElementById("video");
var output_vid = document.getElementById("output-video");
const frameNumberElement = document.getElementById("frameNumber");
const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");

function play() {
  if (video.paused) {
    video.play();
    output_vid.play();
  } else {
    video.pause();
    output_vid.pause();
  }
}

function deleteFileAndReload() {
  fetch("/delete-file", {
    method: "POST",
  })
    .then((response) => {
      if (!response.ok) {
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

video.addEventListener("click", (event) => {
  const currentTime = video.currentTime;
  // console.log(video.videoTracks[0])
  // const fps = video.playbackRate * video.videoTracks[0].fps;
  const fps = 25;
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
        console.log(response.vid_path)
      output_vid.src = response.vid_path;
    },
    error: function (error) {
      console.error(error);
    },
  });
});