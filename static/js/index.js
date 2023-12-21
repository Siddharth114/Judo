var video = document.getElementById("video");
var output_vid = document.getElementById("output-video")
const frameNumberElement = document.getElementById('frameNumber');
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');

function play() {
    if(video.paused) {
        video.play();
    }
    else {
        video.pause();
    }
}

video.addEventListener('click', (event) => {

    const currentTime = video.currentTime;
    // console.log(video.videoTracks[0])
    // const fps = video.playbackRate * video.videoTracks[0].fps;
    const fps = 25;
    const currentFrame = Math.floor(currentTime * fps);


    var rect = event.target.getBoundingClientRect();
    var x = event.clientX - rect.left;
    var y = event.clientY - rect.top;
    var vid_width = rect.width
    var vid_height = rect.height
    // document.getElementById('vid-x-value').textContent = x      //display x coordinates of mouse click
    // document.getElementById('vid-y-value').textContent = y      //display y coordinates of mouse click

    $.ajax({
        url:'/process',
        type:'POST',
        contentType: 'application/json',
        data: JSON.stringify({'mouseX':x, 'mouseY':y, 'vidWidth':vid_width, 'vidHeight':vid_height, 'frameNumber':currentFrame}),
        success: function (response) {
            document.getElementById('snapshot').src = response.image
            document.getElementById('output-video-path').src = response.output_vid_path
        },
        error: function(error) {
            console.error(error);
          }
    });
  });