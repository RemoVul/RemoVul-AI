import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import Form from "react-bootstrap/Form";
import Button from "react-bootstrap/Button";
import Container from "react-bootstrap/Container";

function ScanForm() {
  const [githubUrl, setGithubUrl] = useState("");
  const [isValidUrl, setIsValidUrl] = useState(true);
  const navigate = useNavigate();

  const handleSubmit = (event) => {
    event.preventDefault();
    if (isValidUrl) {
      const encodedUrl = encodeURIComponent(githubUrl);
      navigate(`/scan-result?github_link=${encodedUrl}`);
    }
  };

  const handleGithubUrlChange = (event) => {
    const url = event.target.value;
    setGithubUrl(url);
    setIsValidUrl(validateGithubUrl(url));
  };

  const validateGithubUrl = (url) => {
    const regex = /^(https?:\/\/)?(www\.)?github\.com\/[a-zA-Z0-9-]+\/[a-zA-Z0-9-]+$/;
    return regex.test(url);
  };

  return (
    <>
      <h1 className="text-center mt-5 mb-5">Removul</h1>
      <h3 className="text-center mb-5">
        A tool to remove vulnerabilities from your code
      </h3>
      <Container
        className="d-flex align-items-center justify-content-center"
        style={{ minHeight: "50vh" }}
      >
        <Form
          onSubmit={handleSubmit}
          className="text-center"
          style={{ width: "100%" }}
        >
          <Form.Group controlId="githubUrl">
            <Form.Label className="mb-3">GitHub URL</Form.Label>
            <Form.Control
              type="text"
              placeholder="Enter GitHub URL"
              value={githubUrl}
              onChange={handleGithubUrlChange}
              isInvalid={!isValidUrl}
            />
            {!isValidUrl && (
              <Form.Control.Feedback type="invalid">
                Please enter a valid GitHub URL.
              </Form.Control.Feedback>
            )}
          </Form.Group>
          <Button
            className="mt-3"
            size="lg"
            variant="secondary"
            type="submit"
            disabled={!isValidUrl}
          >
            Start Scan
          </Button>
        </Form>
      </Container>
    </>
  );
}

export default ScanForm;
