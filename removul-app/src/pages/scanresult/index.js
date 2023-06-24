import React, { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import RenderVulLines from "../../components/rendervullines";
import Spinner from "react-bootstrap/Spinner";
import Alert from "react-bootstrap/Alert";

function ScanResult() {
  const [vulLines, setVulLines] = useState(null);
  const location = useLocation();
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const searchParams = new URLSearchParams(location.search);
    const githubUrl = searchParams.get("github_link");
    // if not github url, redirect to home page
    if (!githubUrl) {
      window.location.href = "/";
    }
    setIsLoading(true); // Set loading state to true before fetching data
    setError(null); // Clear any previous error

    fetch(`http://localhost:5000/api/vul_lines?github_link=${githubUrl}`)
      .then((response) => {
        if (response.ok) {
          return response.json();
        } else {
          return response.json().then((data) => {
            throw new Error(data.error);
          });
        }
      })
      .then((data) => {
        console.log(data);
        setVulLines(data);
        setIsLoading(false); // Set loading state to false after fetching data
      })
      .catch((error) => {
        console.log(error);
        setError(error.message);
        setIsLoading(false); // Set loading state to false in case of error
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
      ) : error ? (
        <Alert variant="danger">{error}</Alert>
      ) : (
        vulLines && <RenderVulLines vul_lines={vulLines.vul_lines} />
      )}
    </div>
  );
}

export default ScanResult;
