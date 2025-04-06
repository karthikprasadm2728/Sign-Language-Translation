const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');
const predictionText = document.getElementById('prediction');

// Access webcam
navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
  video.srcObject = stream;
});

// Send a frame every 300ms
setInterval(() => {
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  const imageData = canvas.toDataURL('image/jpeg');

  fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: imageData })
  })
  .then(res => res.json())
  .then(data => {
    predictionText.innerText = `Prediction: ${data.prediction}`;
  });
}, 300);
