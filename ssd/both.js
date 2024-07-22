const uploadContainer = document.getElementById('uploadContainer');
const fileInput = document.getElementById('imageUpload');
const numericalForm = document.getElementById('numericalForm');
const submitButton = document.getElementById('submitButton');
const outputDiv = document.getElementById('output');
const errorMessage = document.getElementById('errorMessage');

let uploadedFile = null;
let numericalValues = [];

// Handle image file selection or drag-and-drop
function handleFiles(files) {
  const file = files[0];
  const img = new Image();
  img.src = URL.createObjectURL(file);
  img.onload = function() {
    uploadContainer.innerHTML = '<p class="message">Photo uploaded successfully!</p>';
    uploadedFile = file;
    errorMessage.style.display = 'none';
  };
  img.onerror = function() {
    alert('Invalid image file.');
  };
}

// Event listeners for image upload
uploadContainer.addEventListener('click', () => fileInput.click());
uploadContainer.addEventListener('dragover', (event) => {
  event.preventDefault();
  uploadContainer.classList.add('dragover');
});
uploadContainer.addEventListener('dragleave', () => {
  uploadContainer.classList.remove('dragover');
});
uploadContainer.addEventListener('drop', (event) => {
  event.preventDefault();
  uploadContainer.classList.remove('dragover');
  const files = event.dataTransfer.files;
  handleFiles(files);
});
fileInput.addEventListener('change', () => {
  const files = fileInput.files;
  handleFiles(files);
});

// Retrieve numerical values from the form and format them as a JSON array of strings
function getNumericalValues() {
  const inputs = numericalForm.querySelectorAll('input[type="number"]');
  numericalValues = []; // Reset the array
  inputs.forEach(input => {
    numericalValues.push(input.value); // Collect values as strings
  });
}

// Submit form data and make API call
submitButton.addEventListener('click', () => {
  if (uploadedFile && numericalValues.length === 8) {
    // Convert the numerical values array to a JSON string
    const numericalData = JSON.stringify(numericalValues);

    // Prepare form data for image file
    const formData = new FormData();
    formData.append('image', uploadedFile);
    formData.append('data', numericalData); // Append numerical data as a string

    // Make the API call to upload the image and numerical data
    fetch('https://pulsardetection.pythonanywhere.com/api/merged-predict/?', {
      method: 'POST',
      body: formData
    })
    .then(response => response.json())
    .then(result => {
      // Display the uploaded image and numerical values
      const img = new Image();
      img.src = URL.createObjectURL(uploadedFile);

      outputDiv.innerHTML = `
        <h2>Uploaded Image:</h2>
        <img src="${img.src}" alt="Uploaded Image" style="width: 300px;"/>
        <h2>Numerical Values:</h2>
        <p><b>Mean of the Integrated Profile:</b> ${numericalValues[0]}</p>
        <p><b>Standard deviation of the Integrated Profile:</b> ${numericalValues[1]}</p>
        <p><b>Excess kurtosis of the Integrated Profile:</b> ${numericalValues[2]}</p>
        <p><b>Skewness of the Integrated Profile:</b> ${numericalValues[3]}</p>
        <p><b>Mean of the DM-SNR Curve:</b> ${numericalValues[4]}</p>
        <p><b>Standard deviation of the DM-SNR Curve:</b> ${numericalValues[5]}</p>
        <p><b>Excess kurtosis of the DM-SNR Curve:</b> ${numericalValues[6]}</p>
        <p><b>Skewness of the DM-SNR Curve:</b> ${numericalValues[7]}</p>
        <h2>Prediction:</h2>
        <p><b>CNN Prediction:</b> ${result.cnn_prediction}</p>
        <p><b>CNN Probability:</b> ${result.cnn_probability}</p>
        <p><b>ANN Prediction:</b> ${result.ann_prediction}</p>
        <p><b>ANN Probability:</b> ${result.ann_probability}</p>
        <p><b>Merged Prediction:</b> ${result.merged_prediction}</p>
        <p><b>Merged Probability:</b> ${result.merged_probability}</p>
      `;
      document.getElementById('section2').style.display = 'block';
      window.scrollTo({
        top: outputDiv.offsetTop,
        behavior: 'smooth'
      });
    })
    .catch(error => {
      console.error('Error:', error);
      outputDiv.innerHTML = `<p style="color: red;">An error occurred: ${error.message}</p>`;
      document.getElementById('section2').style.display = 'block';
      window.scrollTo({
        top: outputDiv.offsetTop,
        behavior: 'smooth'
      });
    });
  } else {
    alert('Please ensure both image and all numerical values are entered.');
  }
});

// Update numerical values on input change
numericalForm.addEventListener('input', getNumericalValues);
