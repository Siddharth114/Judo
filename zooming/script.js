const imageList = [];

let currentFrame = 0;

function nextFrame() {
  if (currentFrame >= imageList.length) return;

  const imageData = imageList[currentFrame];

  // Create a temporary video element and set source
  const video = document.createElement("video");
  video.src = `data:image/png;base64,${imageData}`;

  // Play the video and destroy it after ending
  video.play().then(() => video.remove());

  currentFrame++;
  setTimeout(nextFrame, 100); // Set frame rate (100ms = 10fps)
}

nextFrame()