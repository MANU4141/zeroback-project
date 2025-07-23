import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import OOTDForm from "./components/OOTDForm";
import ResultPage from "./components/ResultPage";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<OOTDForm />} />
        <Route path="/result" element={<ResultPage />} />
      </Routes>
    </Router>
  );
}

export default App;
