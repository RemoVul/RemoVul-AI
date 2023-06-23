import React, { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import RenderVulLines from "../../components/rendervullines";
import Spinner from "react-bootstrap/Spinner";

function ScanResult() {
  const [vulLines, setVulLines] = useState(null);
  const location = useLocation();
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const searchParams = new URLSearchParams(location.search);
    const githubUrl = searchParams.get("github_link");
    setIsLoading(true); // Set loading state to true before fetching data
    console.log(isLoading);

    fetch(`http://localhost:5000/api/vul_lines?github_link=${githubUrl}`)
      .then((response) => response.json())
      .then((data) => {
        setVulLines(data);
        setIsLoading(false); // Set loading state to false after fetching data
        console.log(data);
      });
  }, [location.search]);

  return (
    <div>
      <h1>Scan Result</h1>
      {isLoading ? (
        <div
          className="container mx-auto p-4 d-flex align-items-center justify-content-center"
          style={{ minHeight: "100vh" }}
        >
          <Spinner animation="border" variant="dark" role="status">
            <span className="visually-hidden">Loading...</span>
          </Spinner>
        </div>
      ) : (
        vulLines && <RenderVulLines vul_lines={vulLines.vul_lines} />
      )}
    </div>
  );
}

export default ScanResult;
