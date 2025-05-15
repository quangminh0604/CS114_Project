import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import "./App.css";
import logo from './logo.svg';
import Home from './Pages/HomePage/Home';
import Navbar from './components/Navbar/Navbar'
import EDA from './Pages/EDA/EDA'
import Preprocessing from './Pages/Preprocessing/Preprocessing'
import Models from './Pages/Models/Models';
import SVM from './Pages/Models/SVM/SVM';
import DecisionTree from './Pages/Models/DecisionTree/DecisionTree';
import RandomForest from './Pages/Models/RandomForest/RandomForest';
function App() {
  return (
    <Router> {/* Wrap your routes in BrowserRouter */}
    <Navbar /> {/* Add Navbar at the top */}
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/EDA" element={<EDA />} />
        
				<Route path="/models" element={<Models />}>
          <Route path="SVM" element={<SVM />} />
          <Route path="DecisionTree" element={<DecisionTree />} />
          <Route path="RandomForest" element={<RandomForest />} />
        </Route>
        
				<Route path="/Preprocessing" element={<Preprocessing />} />

        {/* Add more routes here */}
      </Routes>
    </Router>
  );
}

export default App;
