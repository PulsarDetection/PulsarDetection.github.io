document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const imageInput = document.getElementById('imageInput');
    const formData = new FormData();
    formData.append('image', imageInput.files[0]);

    fetch('192.168.23.114/cnn_predict/', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const resultDiv = document.getElementById('result');
        resultDiv.innerHTML = `<p>Prediction: ${data.prediction}</p><p>Probability: ${data.probability}</p>`;
    })
    .catch(error => {
        console.error('Error:', error);
        const resultDiv = document.getElementById('result');
        resultDiv.innerHTML = `<p>There was an error processing your request.</p>`;
    });
});
