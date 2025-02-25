import React, { useState, ChangeEvent, FormEvent } from 'react';
import axios from 'axios';
import './styles.css';

interface FileWithPath extends File {
  path?: string;
}

const App: React.FC = () => {
  const [urls, setUrls] = useState<string[]>(['']);
  const [files, setFiles] = useState<FileWithPath[]>([]);
  const [message, setMessage] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isDragging, setIsDragging] = useState(false);

  const handleUrlChange = (index: number, value: string): void => {
    const newUrls = [...urls];
    newUrls[index] = value;
    setUrls(newUrls);
  };

  const addUrlField = (): void => {
    setUrls([...urls, '']);
  };

  const removeUrlField = (index: number): void => {
    const newUrls = urls.filter((_, i) => i !== index);
    setUrls(newUrls.length ? newUrls : ['']);
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>): void => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files) as FileWithPath[];
      setFiles([...files, ...newFiles]);
    }
  };

  const removeFile = (index: number): void => {
    setFiles(files.filter((_, i) => i !== index));
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files) {
      const newFiles = Array.from(e.dataTransfer.files) as FileWithPath[];
      const pdfFiles = newFiles.filter(file => file.type === 'application/pdf');
      setFiles([...files, ...pdfFiles]);
    }
  };

  const handleSubmit = async (e: FormEvent<HTMLFormElement>): Promise<void> => {
    e.preventDefault();
    setIsLoading(true);
    const formData = new FormData();

    urls.forEach((url, index) => {
      if (url.trim()) formData.append(`urls[${index}]`, url.trim());
    });

    files.forEach((file) => formData.append('files', file));

    try {
      const response = await axios.post('http://localhost:5000/generate', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setMessage('success');
      console.log(response.data);
    } catch (error) {
      setMessage('error');
      console.error(error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-800 to-gray-900 py-8">
      <div className="container">
        <div className="content-wrapper">
          <div className="landing-content">
            <h1>AI-Powered Question Generator</h1>
            <h3 className="subtitle">Generate AI-Powered Questions from URLs & PDFs Instantly!</h3>
            
            <div className="intro">
              <p className="highlight">🚀 Transform any webpage or document into an interactive quiz with <strong>AI-powered automation</strong>!</p>
            </div>

            <div className="features-section">
              <h2>How It Works?</h2>
              <div className="steps">
                <div className="step">✅ <strong>Step 1:</strong> Upload a URL or PDF 📄</div>
                <div className="step">✅ <strong>Step 2:</strong> AI extracts content & generates <strong>smart questions</strong> ✨</div>
                <div className="step">✅ <strong>Step 3:</strong> Download the results in <strong>CSV format</strong> 📥</div>
              </div>
            </div>

            <div className="features-grid">
              <div className="feature-card">
                <h3>🎯 Features</h3>
                <ul>
                  <li>✅ Supports <strong>multiple URLs & PDFs</strong> 📑</li>
                  <li>✅ Uses <strong>Gemini AI</strong> for question generation 🧠</li>
                  <li>✅ Append & export all questions in a single CSV</li>
                  <li>✅ Fast, accurate, and user-friendly</li>
                </ul>
              </div>

              <div className="feature-card">
                <h3>📌 Use Cases</h3>
                <ul>
                  <li><strong>Educators & Trainers</strong> – Create quizzes from study materials</li>
                  <li><strong>Businesses & Researchers</strong> – Extract key insights from reports</li>
                  <li><strong>Students & Self-Learners</strong> – Test knowledge on any topic</li>
                </ul>
              </div>

              <div className="feature-card">
                <h3>🚀 Built with</h3>
                <ul>
                  <li><strong>TypeScript & Python</strong> for seamless performance</li>
                  <li><strong>Crew AI Agents</strong> for intelligent automation</li>
                  <li><strong>Gemini AI</strong> for advanced question generation</li>
                </ul>
              </div>
            </div>

            <div className="cta-section">
              <p className="highlight">💡 Try it now & revolutionize content learning!</p>
            </div>
          </div>

          <div className="divider"></div>

          <form onSubmit={handleSubmit}>
            <div className="input-section">
              <h2>
                <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10"/>
                  <path d="M2 12h20"/>
                  <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
                </svg>
                Input URLs
              </h2>
              
              {urls.map((url, index) => (
                <div key={index} className="url-input-group">
                  <input
                    type="url"
                    value={url}
                    onChange={(e) => handleUrlChange(index, e.target.value)}
                    placeholder="Enter a URL"
                    className="input-field"
                  />
                  {urls.length > 1 && (
                    <button
                      type="button"
                      onClick={() => removeUrlField(index)}
                      className="remove-btn"
                    >
                      <svg className="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M18 6L6 18M6 6l12 12"/>
                      </svg>
                    </button>
                  )}
                </div>
              ))}
              
              <button type="button" onClick={addUrlField} className="add-btn">
                <svg className="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 5v14M5 12h14"/>
                </svg>
                Add Another URL
              </button>
            </div>

            <div className="input-section">
              <h2>
                <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                  <path d="M14 2v6h6"/>
                  <path d="M16 13H8"/>
                  <path d="M16 17H8"/>
                  <path d="M10 9H8"/>
                </svg>
                Upload PDFs
              </h2>

              <div 
                className={`file-drop-zone ${isDragging ? 'dragging' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => document.getElementById('file-upload')?.click()}
              >
                <input
                  type="file"
                  id="file-upload"
                  accept=".pdf"
                  multiple
                  onChange={handleFileChange}
                  className="hidden"
                />
                <svg className="icon-lg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                  <polyline points="17 8 12 3 7 8"/>
                  <line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
                <p>Drop PDF files here or click to upload</p>
              </div>

              {files.length > 0 && (
                <div className="file-list">
                  {files.map((file, index) => (
                    <div key={index} className="file-item">
                      <span>{file.name}</span>
                      <button
                        type="button"
                        onClick={() => removeFile(index)}
                        className="remove-btn"
                      >
                        <svg className="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M18 6L6 18M6 6l12 12"/>
                        </svg>
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="submit-btn"
            >
              {isLoading ? (
                <>
                  <svg className="loading-spinner icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10" strokeOpacity="0.25"/>
                    <path d="M12 2a10 10 0 0 1 10 10" strokeOpacity="0.75"/>
                  </svg>
                  Generating Questions...
                </>
              ) : (
                'Generate Questions'
              )}
            </button>
          </form>

          {message && (
            <div className={`message ${message === 'success' ? 'success' : 'error'}`}>
              <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                {message === 'success' ? (
                  <path d="M20 6L9 17l-5-5"/>
                ) : (
                  <path d="M18 6L6 18M6 6l12 12"/>
                )}
              </svg>
              <div>
                <h3 className="font-semibold">
                  {message === 'success' ? 'Success!' : 'Error'}
                </h3>
                <p className="text-sm">
                  {message === 'success'
                    ? 'Questions generated successfully! Check backend response.'
                    : 'Error generating questions. Please try again.'}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;