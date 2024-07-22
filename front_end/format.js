const uploadContainer = document.getElementById('uploadContainer');
const fileInput = document.getElementById('fileUpload');
const uploadForm = document.getElementById('uploadForm');
const outputDiv = document.getElementById('output');
const errorMessage = document.getElementById('errorMessage');
const uploadButton = document.getElementById('uploadButton');
let uploadedFile = null;

function handleFiles(files) {
  const file = files[0];
  if (file.name.endsWith('.phcx')) {
      uploadContainer.innerHTML = '<p class="message">.phcx file uploaded successfully!</p>';
      uploadedFile = file;
      errorMessage.style.display = 'none';
  } else {
      alert('Please upload a valid .phcx file.');
  }
}

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

uploadForm.addEventListener('submit', function(event) {
  event.preventDefault();
  if (!uploadedFile) {
      errorMessage.style.display = 'block';
  } else {
      errorMessage.style.display = 'none';

      // Prepare form data for submission
      const formData = new FormData();
      formData.append('file', uploadedFile);

      // Make the API call to upload the .phcx file
      fetch('https://pulsardetection.pythonanywhere.com/api/phcx-predict/', {
        method: 'POST',
        body: formData
      })
      .then(response => response.json())
      .then(result => {
        // Display the response
        const outputDiv1 = document.getElementById('output');
        const outputDiv2 = document.getElementById('output1');
        
        // Create an image element if base64 image is present in response
        let imgHtml = '';
        if (result.image_base64) {
          const imgSrc = `data:image/png;base64,${result.image_base64}`;
          imgHtml = `
            <p><b>Generated Image:</b></p>
            <img src="${imgSrc}" alt="Generated Image" style="width: 300px;"/>
          `;
        }

        outputDiv1.innerHTML = `
            <p><b>Generated Data:</b> ${result.generated_data}</p>
            ${imgHtml}
        `;
        outputDiv2.innerHTML = `
            <h2>Prediction:</h2>
            <p><b>CNN Prediction:</b> ${result.cnn_prediction}</p>
            <p><b>CNN Probability:</b> ${result.cnn_probability}</p>
            <p><b>ANN Prediction:</b> ${result.ann_prediction}</p>
            <p><b>ANN Probability:</b> ${result.ann_probability}</p>
            <p><b>Merged Prediction:</b> ${result.merged_prediction}</p>
            <p><b>Merged Probability:</b> ${result.merged_probability}</p>
        `; // Adjust as needed
        document.getElementById('section2').style.display = 'block';
        window.scrollTo({
          top: outputDiv1.offsetTop,
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
  }
});

function validateForm() {
  if (!uploadedFile) {
    alert('Please upload a document first.');
    return false;
  }
  return true;
}

uploadButton.addEventListener('click', (event) => {
  if (!validateForm()) {
    event.preventDefault();
  }
});
