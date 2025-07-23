// script.js

document.getElementById('predictionForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const crop = document.getElementById('crop').value;
    const month = document.getElementById('month').value;
    const city = document.getElementById('city').value;
    const state = document.getElementById('state').value;
    const year = document.getElementById('year').value;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ crop, month, city, state, year })
        });

        const data = await response.json();
        const resultDiv = document.getElementById('result');

        if (data.price) {
            resultDiv.innerHTML = `<h3>Predicted Price: ₹${data.price}</h3>`;
        } else {
            resultDiv.innerHTML = `<h3>Error: ${data.error}</h3>`;
        }
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('result').innerHTML = `<h3>An error occurred while fetching the prediction.</h3>`;
    }
});