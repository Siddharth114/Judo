var video = document.getElementById("video");
const frameNumberElement = document.getElementById('frameNumber');
const fps = 20
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



document.getElementById("video").onclick = function(e) {
    var rect = e.target.getBoundingClientRect();
    var x = e.clientX - rect.left;
    var y = e.clientY - rect.top;
    document.getElementById('vid-x-value').textContent = x
    document.getElementById('vid-y-value').textContent = y
}

video.addEventListener('click', (event) => {
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
  });