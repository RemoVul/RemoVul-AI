import "./App.css";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import ScanForm from "./pages/scanform";
import ScanResult from "./pages/scanresult";
import ShowFile from "./pages/showfile";

function App() {
  return (
    <Router>
      <Routes>
        <Route exact path="/" element={<ScanForm/>}/>
        <Route path="/scan-result" element={<ScanResult/>}/>
        <Route path="/show-file" element={<ShowFile/>}/>
      </Routes>
    </Router>
  );
}

export default App;
