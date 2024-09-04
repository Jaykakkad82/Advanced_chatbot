import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './SessionPage.css';
import { v4 as uuidv4 } from 'uuid';  // Import UUID generator

// Define the base URL for the backend
const BASE_URL = 'http://localhost:5000/api/v1';  // Change this during deployment

const SessionPage = () => {
  const [apiKey, setApiKey] = useState('');
  const [pdfFiles, setPdfFiles] = useState(['']);
  const [statusMessage, setStatusMessage] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [chatMessages, setChatMessages] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [sessionId, setSessionId] = useState(uuidv4());  // Generate a unique session ID

  const chatWindowRef = useRef(null);  // Create a ref for the chat window

  // Automatically scroll to the bottom of the chat window when a new message is added
  useEffect(() => {
    if (chatWindowRef.current) {
      chatWindowRef.current.scrollTop = chatWindowRef.current.scrollHeight;
    }
  }, [chatMessages]);

  const handleApiKeySubmit = async () => {
    try {
      const response = await axios.post(`${BASE_URL}/store-api-key`, {
        session_id: sessionId,
        api_key: apiKey,
      });
      setStatusMessage('API Key stored for the session');
    } catch (error) {
      console.error('Error storing API key', error);
      setStatusMessage('Failed to store API key');
    }
  };

  const handlePdfUpload = (index, e) => {
    const files = [...pdfFiles];
    files[index] = e.target.files[0];
    setPdfFiles(files);
  };

  const handleAddFile = () => {
    setPdfFiles([...pdfFiles, '']);
  };

  const handlePdfSubmit = async () => {
    setIsProcessing(true);
    try {
      const formData = new FormData();
      pdfFiles.forEach((file) => {
        if (file) {
          formData.append('pdfs', file);
        }
      });
      formData.append('session_id', sessionId);
      formData.append('api_key', apiKey);

      const response = await axios.post(`${BASE_URL}/upload-pdfs`, formData);
      setStatusMessage('Upload complete, you can now chat with this PDF');
    } catch (error) {
      console.error('Error uploading PDFs', error);
      setStatusMessage('Failed to upload PDFs');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleAskQuestion = async () => {
    if (!userInput.trim()) return;

    const newChatMessages = [...chatMessages, { sender: 'user', message: userInput }];
    setChatMessages(newChatMessages);
    setUserInput('');

    try {
      const response = await axios.post(`${BASE_URL}/ask-question`, {
        session_id: sessionId,
        question: userInput,
      });
      
      const botResponse = response.data.answer;
      setChatMessages([...newChatMessages, { sender: 'bot', message: botResponse }]);
    } catch (error) {
      console.error('Error asking question', error);
      setChatMessages([...newChatMessages, { sender: 'bot', message: 'Failed to get response' }]);
    }
  };

  return (
    <div className="session-container">
      <div className="sidebar">
        <h3>Session Sidebar</h3>
        
        <div className="step">
          <h4>1. Add OpenAI Key</h4>
          <input
            type="text"
            placeholder="OpenAI API Key"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
          />
          <button onClick={handleApiKeySubmit}>Submit API Key</button>
        </div>

        <div className="step">
          <h4>2. Upload PDF Documents</h4>
          {pdfFiles.map((file, index) => (
            <input
              key={index}
              type="file"
              accept="application/pdf"
              onChange={(e) => handlePdfUpload(index, e)}
            />
          ))}
          <button onClick={handleAddFile}>Add More</button>
          <button onClick={handlePdfSubmit} disabled={isProcessing}>Submit PDFs</button>
        </div>

        {statusMessage && <p>{statusMessage}</p>}
        {isProcessing && <p>Processing...</p>}
      </div>
      
      <div className="chatbox">
        <h3>Chat with the PDF</h3>
        <div className="chat-window" ref={chatWindowRef}> {/* Attach the ref to the chat window */}
          {chatMessages.map((msg, index) => (
            <div key={index} className={msg.sender === 'user' ? 'user-message' : 'bot-message'}>
              {msg.message}
            </div>
          ))}
        </div>
        <div className="chat-input">
          <input 
            type="text" 
            placeholder="Type your message here..." 
            value={userInput} 
            onChange={(e) => setUserInput(e.target.value)} 
          />
          <button onClick={handleAskQuestion}>Send</button>
        </div>
      </div>
    </div>
  );
};

export default SessionPage;
