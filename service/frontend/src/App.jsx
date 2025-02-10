import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Main from "./Pages/Main";
import Result1 from "./Pages/Result1";
import Result2 from "./Pages/Result2";
import Header from "./components/Header";
import SimilarSearch from "./Pages/SimilarSearch";
import SimilarSearchResult from "./Pages/SimilarSearchResult";
import "./App.css";

function App() {
  return (
    <BrowserRouter>
      <Header />

      <div className="container">
        <Routes>
          <Route path="/" element={<Main />} />
          <Route path="/result1" element={<Result1 />} />
          <Route path="/result2" element={<Result2 />} />
          <Route path="/similarsearch" element={<SimilarSearch />} />
          <Route
            path="/similarsearchresult"
            element={<SimilarSearchResult />}
          />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
