var video = document.getElementById("video");
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

    // Get video dimensions
    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;
  
    // Set canvas size to match video dimensions
    canvas.width = videoWidth;
    canvas.height = videoHeight;
  
    // Draw current frame onto canvas
    ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
    // Extract image data as Base64 string
    const imageData = canvas.toDataURL('image/png');

    // Use the image data (e.g., display it in an image element)
    const snapshotElement = document.getElementById('snapshot');
    snapshotElement.src = imageData;

    var data = imageData.split(',')[1];

    $.ajax({
        url:'/process',
        type:'POST',
        contentType: 'application/json',
        data: JSON.stringify({ 'img': data , 'mouseX':x, 'mouseY':y, 'vidWidth':vid_width, 'vidHeight':vid_height, 'frameNumber':currentFrame}),
        success: function (response) {
            document.getElementById('output-image').src = response.image
        },
        error: function(error) {
            console.error(error);
          }
    });
  });