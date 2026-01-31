import React from "react";
import { Link, useInRouterContext } from "react-router-dom";
import "./Navbar.css";

/** Navbar
@returns {JSX.Element}
*/
const Navbar = () => {
  const inRouter = useInRouterContext();

  return (
    <nav className="navbar">
      <div className="navbar-logo">HiperHealth</div>
      <div className="navbar-links">
        {inRouter ? (
          <>
            <Link to="/dashboard">Patient Dashboard</Link>
            <Link to="/diagnosis">Resume Diagnosis</Link>
          </>
        ) : (
          <>
            <a href="/dashboard">Patient Dashboard</a>
            <a href="/diagnosis">Resume Diagnosis</a>
          </>
        )}
      </div>
    </nav>
  );
};

export default Navbar;
