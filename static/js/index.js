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

document.getElementById("video").onclick = function(e) {
    
}

video.addEventListener('click', (event) => {

    var rect = event.target.getBoundingClientRect();
    var x = event.clientX - rect.left;
    var y = event.clientY - rect.top;
    document.getElementById('vid-x-value').textContent = x
    document.getElementById('vid-y-value').textContent = y

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
    console.log(x, y)

    $.ajax({
        url:'/process',
        type:'POST',
        contentType: 'application/json',
        data: JSON.stringify({ 'value': data , 'mouseX':x, 'mouseY':y})
    });
  });