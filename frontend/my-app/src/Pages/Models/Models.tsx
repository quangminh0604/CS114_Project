import { TIMEOUT } from "dns";
import React from 'react';
import { Outlet, Link } from 'react-router-dom';
import './Models.css'
/**
 * @component
 * @returns {React.ReactElement} A home page with full responsive css
 */
export default function Models() {
    return (
        <div className="models">
            <h1>Models</h1>
            <nav>
                <ul>
                    <li><Link to="SVM">SVM</Link></li>
                    <li><Link to="DecisionTree">Decision Tree</Link></li>
                    <li><Link to="RandomForest">Random Forest</Link></li>
                </ul>
            </nav>
            <div>
            {/* This will render the nested route components */}
            <Outlet />
            </div>
        </div>
    );
}
