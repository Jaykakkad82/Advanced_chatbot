import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './login.css';  // Import the CSS file

const LoginPage = () => {
  const [id, setId] = useState('');
  const [password, setPassword] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const navigate = useNavigate();

  const handleLogin = () => {
    // Implement your login logic here
    // If login fails, set an error message:
    // setErrorMessage('Login failed. Please check your credentials.');

    // If login is successful:
    navigate('/session');
  };

  return (
    <div className="login-container">
      <div className="login-box">
        <h2>Login</h2>
        <label htmlFor="userid">UserID:</label>
        <input
          id="userid"
          type="text"
          placeholder="Enter UserID"
          value={id}
          onChange={(e) => setId(e.target.value)}
        />
        <label htmlFor="password">Password:</label>
        <input
          id="password"
          type="password"
          placeholder="Enter Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <button onClick={handleLogin}>Login</button>
        {errorMessage && <p className="error-message">{errorMessage}</p>}
      </div>
    </div>
  );
};

export default LoginPage;
