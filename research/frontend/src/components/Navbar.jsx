import React from "react";
import { Link } from "react-router-dom";
import "./Navbar.css";

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="navbar-logo">HiperHealth</div>
      <div className="navbar-links">
        <Link to="/dashboard">Patient Dashboard</Link>
        <Link to="/diagnosis">Resume Diagnosis</Link>
      </div>
    </nav>
  );
};

export default Navbar;
