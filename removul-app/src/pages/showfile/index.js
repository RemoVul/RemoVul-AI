import { CopyBlock, dracula } from "react-code-blocks";
import React, { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import Spinner from "react-bootstrap/Spinner";

function ShowFile() {
  const [fileContent, setFileContent] = useState("");
  const location = useLocation();
  const [vulLines, setVulLines] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const searchParams = new URLSearchParams(location.search);
    const fileUrl = searchParams.get("file_url");
    const vul_lines = searchParams.get("vul_lines");
    // check if vul_lines is empty string set vul_lines to "0"
    if (vul_lines === "") { 
        setVulLines("0");
    } else {
        setVulLines(vul_lines);
    }
    setIsLoading(true); // Set loading state to true before fetching data

    fetch(`http://localhost:5000/api/file_content?file_url=${fileUrl}`)
      .then((response) => response.json())
      .then((data) => {
        setFileContent(data.file_content);
        setIsLoading(false); // Set loading state to false after fetching data
        
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
