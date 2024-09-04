// src/components/Navbar.js
import React from 'react';
import './Navbar.css';  // You can add styles here or inline

const Navbar = () => {
  return (
    <div className="navbar">
      <img src="/truodds-logo.png" alt="Logo" className="logo" />
      <h1 className="navbar-title">EquiAssist</h1>
    </div>
  );
};

export default Navbar;