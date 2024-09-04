// import logo from './logo.svg';
import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import LoginPage from './components/LoginPage';
import SessionPage from './components/SessionPage';
import Navbar from './components/Navbar';

function App() {
  return (
    <Router>
      <div>
        <Navbar />  {/* Navbar will be displayed on every page */}
        <Routes>
          <Route path="/" element={<LoginPage />} />
          <Route path="/session" element={<SessionPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;

