import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import EcsMapping from "./pages/EcsMapping";

function App() {
    return (
        <Router>
            <Routes>
                <Route path="/" element={<EcsMapping />} />
            </Routes>
        </Router>
    );
}

export default App;
