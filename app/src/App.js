import './App.css';
import React, { useState } from 'react';

function App() {
  const [file, setFile] = useState(null);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const transcode = async () => {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://127.0.0.1:8000/transcode', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Error: ${await response.text()}`);
      }

      setProgress(0);
      setResult(await response.text());
    } catch (error) {
      console.error('Error:', error);
      alert('An error occurred. Please try again.');
    }
  };

  return (
    <div>
      <h1>Whisper API Client</h1>
      <input type="file" onChange={handleFileChange} />
      <button onClick={transcode}>Transcode</button>

      {progress > 0 && (
        <div>
          <div>
            Progress: {Math.round(progress)}%
          </div>
          <div style={{ width: `${progress}%`, height: '20px', backgroundColor: '#4caf50' }} />
        </div>
      )}

      {result && (
        <div>
          <h2>Transcode Result</h2>
          <pre>{result}</pre>
        </div>
      )}
    </div>
  );
}

export default App;
