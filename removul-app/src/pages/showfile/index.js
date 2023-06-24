import { CopyBlock, dracula } from "react-code-blocks";
import React, { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import Spinner from "react-bootstrap/Spinner";
import Alert from "react-bootstrap/Alert";

function ShowFile() {
  const [fileContent, setFileContent] = useState("");
  const location = useLocation();
  const [vulLines, setVulLines] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const searchParams = new URLSearchParams(location.search);
    const fileUrl = searchParams.get("file_url");
    const vul_lines = searchParams.get("vul_lines");
    // if not file url or not vul_lines, redirect to home page
    if (!fileUrl || !vul_lines) {
      window.location.href = "/";
    }
    // check if vul_lines is empty string set vul_lines to "0"
    if (vul_lines === "") { 
        setVulLines("0");
    } else {
        setVulLines(vul_lines);
    }
    setIsLoading(true); // Set loading state to true before fetching data

    fetch(`http://localhost:5000/api/file_content?file_url=${fileUrl}`)
      .then((response) => {
        if (response.ok) {
          return response.json();
        } else {
          return response.json().then((data) => {
            throw new Error(data.error);
          });
        }
      }).then((data) => {
        console.log(data);
        setFileContent(data.file_content);
        setIsLoading(false); // Set loading state to false after fetching data
      }).catch((error) => {
        console.log(error);
        setError(error.message);
        setIsLoading(false); // Set loading state to false in case of error
      });
  }, [location.search]);

  return (
    <>
      {isLoading ? (
         <div className="container mx-auto p-4 d-flex align-items-center justify-content-center" style={{ minHeight: "100vh" }}>
            <Spinner  animation="border" variant="dark" role="status">
            <span className="visually-hidden">Loading...</span>
            </Spinner>
        </div>
       ) : error ? (
        <Alert variant="danger">{error}</Alert>
      ) : (
        <div className="demo">
          <CopyBlock
            language="cpp"
            text={fileContent}
            showLineNumbers={true}
            theme={dracula}
            highlight={vulLines}
            wrapLines
            codeBlock
          />
        </div>
      )}
    </>
  );
}

export default ShowFile;
