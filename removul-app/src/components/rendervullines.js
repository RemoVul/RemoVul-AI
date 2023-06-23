import React from "react";
import { Link } from 'react-router-dom';


function RenderVulLines(props) {

   const vulLines = props.vul_lines;  
  // Check if vulLines is empty
  if (Object.keys(vulLines).length === 0) {
    return null;
  }

  // Render each name_of_folder and its contents recursively
  return Object.entries(vulLines).map(([folderName, folderContents]) => {
    // Check if folderContents contains another folder
    // folder content not has fileurl as key
    const isFile =
      "vul_lines" in folderContents && "file_url" in folderContents;

    return (
      <div key={folderName} style={{ marginLeft: "16px" }}>
        {!isFile ? (
          <>
            <h3>{folderName}</h3> {<RenderVulLines vul_lines={folderContents} />}
          </>
        ) : (
          <>
            <Link to={'/show-file?file_url=' + folderContents.file_url+ '&vul_lines='+folderContents.vul_lines.join(", ")}>
              <h3>{folderName}</h3>
            </Link>
            {folderContents.vul_lines.length!==0 ? (
              <span>Vulnerable lines: {folderContents.vul_lines.join(", ")}</span>
            ) : (
              <span>No Vul</span>
            )}
          </>
        )}
      </div>
    );
  });
}

export default RenderVulLines;