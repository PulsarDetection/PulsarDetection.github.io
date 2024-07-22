document.getElementById('pulsarForm').addEventListener('submit', function(event) {
    event.preventDefault();
  
    const data = [
      document.getElementById('meanIntegrated').value,
      document.getElementById('stdIntegrated').value,
      document.getElementById('kurtIntegrated').value,
      document.getElementById('skewIntegrated').value,
      document.getElementById('meanDMSNR').value,
      document.getElementById('stdDMSNR').value,
      document.getElementById('kurtDMSNR').value,
      document.getElementById('skewDMSNR').value
    ];
  
    fetch('https://pulsardetection.pythonanywhere.com/api/ann-predict/?', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ data: JSON.stringify(data) })
    })
    .then(response => response.json())
    .then(result => {
      const outputDiv = document.getElementById('output');
      outputDiv.innerHTML = `
        <p><b>Mean of the Integrated Profile:</b> ${document.getElementById('meanIntegrated').value}</p>
        <p><b>Standard deviation of the Integrated Profile:</b> ${document.getElementById('stdIntegrated').value}</p>
        <p><b>Excess kurtosis of the Integrated Profile:</b> ${document.getElementById('kurtIntegrated').value}</p>
        <p><b>Skewness of the Integrated Profile:</b> ${document.getElementById('skewIntegrated').value}</p>
        <p><b>Mean of the DM-SNR Curve:</b> ${document.getElementById('meanDMSNR').value}</p>
        <p><b>Standard deviation of the DM-SNR Curve:</b> ${document.getElementById('stdDMSNR').value}</p>
        <p><b>Excess kurtosis of the DM-SNR Curve:</b> ${document.getElementById('kurtDMSNR').value}</p>
        <p><b>Skewness of the DM-SNR Curve:</b> ${document.getElementById('skewDMSNR').value}</p>
        <p><b>Prediction:</b> ${result.prediction}</p>
        <p><b>Probability:</b> ${result.probability}</p>
      `;
      document.getElementById('section2').style.display = 'block';
      document.querySelector('.section2').scrollIntoView({ behavior: 'smooth' });
    })
    .catch(error => {
      console.error('Error:', error);
      const outputDiv = document.getElementById('output');
      outputDiv.innerHTML = `<p style="color: red;">An error occurred: ${error.message}</p>`;
      document.getElementById('section2').style.display = 'block';
      document.querySelector('.section2').scrollIntoView({ behavior: 'smooth' });
    });
  });
  