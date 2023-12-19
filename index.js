window.addEventListener('mousemove', function(e){
    document.getElementById('x-value').textContent = e.x
    document.getElementById('y-value').textContent = e.y
});
var video = document.getElementById("vid");

function play() {
    console.log(video)
    if(video.paused) {
        video.play();
    }
    else {
        video.pause();
    }
}

document.getElementById("vid").onclick = function(e) {
    var rect = e.target.getBoundingClientRect();
    var x = e.clientX - rect.left;
    var y = e.clientY - rect.top;
    document.getElementById('vid-x-value').textContent = x
    document.getElementById('vid-y-value').textContent = y

}