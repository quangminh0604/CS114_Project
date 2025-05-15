import React from 'react';
import { Link } from 'react-router-dom';  // Link is used for navigation in React Router
import './Navbar.css';
const Navbar: React.FC = () => {
  return (
    <nav className="navbar">
      <div className="logo">
      <Link to="/">
          <img src="/uit.svg" alt="MyLogo" className="logo-image" />
        </Link>
      </div>
      <ul className="nav-links">
        <li><Link to="/">Home</Link></li>
        <li><Link to="/EDA">EDA</Link></li>
        <li><Link to="/preprocessing">Preprocessing</Link></li>
        <li><Link to="/models">Models</Link></li>
      </ul>
    </nav>
  );
}

export default Navbar;