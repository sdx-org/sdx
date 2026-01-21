import React, { useState, useRef, useEffect} from 'react';
import { useForm } from 'react-hook-form';
import {
  Form,
  Button,
  Container,
  Card,
  ProgressBar,
  Alert,
  Spinner,
  ListGroup,
  Row,
  Col,
} from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useConsultation, consultationActions } from '../../context/ConsultationContext';
import consultationAPI from '../../services/api';

export default function MedicalReport() {
  const navigate = useNavigate();
  const { t } = useTranslation();
  const { state, dispatch } = useConsultation();
  const fileInputRef = useRef(null);
  const {
    handleSubmit,
    formState: { isSubmitting },
  } = useForm();
  useEffect(()=>{
    if(!state.formData.medicalReports){
      dispatch(consultationActions.updateMedicalReports({
        files:[],
        skipped:false,
      }));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  },[]);
  const [apiError, setApiError] = useState(null);
  const [selectedFiles, setSelectedFiles] = useState(
    state.formData.medicalReports?.files || []
  );
  const acceptedFormats = [
    'application/pdf',
    'image/jpeg',
    'image/png',
    'image/jpg',
  ];

  const acceptedExtensions = '.pdf, .jpg, .jpeg, .png';
  const maxFileSize = 20 * 1024 * 1024; // 20MB
  const maxFiles = 5;
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes, k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
  };

  const handleFileSelect = (event) => {
    const files = Array.from(event.target.files);
    setApiError(null);

    // Validate file count
    if (selectedFiles.length + files.length > maxFiles) {
      setApiError(`Maximum ${maxFiles} files allowed`);
      return;
    }

    // Validate each file
    const validFiles = [];
    for (const file of files) {
      // Check format
      if (!acceptedFormats.includes(file.type)) {
        setApiError(
          `Invalid file format: ${file.name}. Accepted: PDF, JPG, PNG`
        );
        continue;
      }

      // Check size
      if (file.size > maxFileSize) {
        setApiError(
          `File too large: ${file.name} (${formatFileSize(file.size)}). Max: 10MB`
        );
        continue;
      }

      validFiles.push(file);
    }

    if (validFiles.length > 0) {
      setSelectedFiles([...selectedFiles, ...validFiles]);
    }
  };

  const removeFile = (index) => {
    setSelectedFiles(selectedFiles.filter((_, i) => i !== index));
  };

  const clearAllFiles = () => {
    setSelectedFiles([]);
  };

  const handleSkip = async () => {
    try {
      setApiError(null);

      // Validate patient ID exists
      if (!state.patientId) {
        throw new Error('Patient ID not found. Please start over.');
      }

      // Update local state
      dispatch(
        consultationActions.updateMedicalReports({
          files: [],
          skipped: true,
        })
      );

      // Call backend API to skip
      await consultationAPI.skipMedicalReports(state.patientId);

      // Update current step in context
      dispatch(consultationActions.setCurrentStep('wearable-data'));

      // Navigate to next step
      navigate('/wearable-data');
    } catch (err) {
      console.error('Error skipping medical reports:', err);
      setApiError(err.message || 'Failed to skip medical reports. Please try again.');
      window.scrollTo(0, 0);
    }
  };
  const onSubmit = async () => {
    try {
      setApiError(null);

      // Validate patient ID exists
      if (!state.patientId) {
        throw new Error('Patient ID not found. Please start over.');
      }

      if (selectedFiles.length === 0) {
        setApiError('Please select files or click Skip');
        return;
      }

      // Create FormData for multipart upload
      const formData = new FormData();
      selectedFiles.forEach((file) => {
        formData.append('files', file);
      });

      // Update local state
      dispatch(
        consultationActions.updateMedicalReports({
          files: selectedFiles.map((f) => ({
            name: f.name,
            size: f.size,
            type: f.type,
          })),
          skipped: false,
        })
      );

      // Call backend API to upload
      await consultationAPI.uploadMedicalReports(state.patientId, formData);

      // Update current step in context
      dispatch(consultationActions.setCurrentStep('wearable-data'));

      // Navigate to next step
      navigate('/wearable-data');
    } catch (err) {
      console.error('Error uploading medical reports:', err);
      setApiError(err.message || 'Failed to upload medical reports. Please try again.');
      window.scrollTo(0, 0);
    }
  };
  const hasFiles = selectedFiles.length > 0;

  return (
    <div className="bg-light min-vh-100 py-4">
      <Container>
        {/* Progress Section */}
        <div className="mb-4">
          <div className="d-flex justify-content-between align-items-center mb-2">
            <small className="text-muted fw-semibold">Step 5 of 10</small>
            <small className="text-muted">50% Complete</small>
          </div>
          <ProgressBar now={50} style={{ height: '8px' }} className="mb-3" />
        </div>

        {/* Error Alert */}
        {apiError && (
          <Alert
            variant="danger"
            dismissible
            onClose={() => setApiError(null)}
            className="mb-4"
          >
            <Alert.Heading>Error</Alert.Heading>
            <p className="mb-0">{apiError}</p>
          </Alert>
        )}

        {/* Main Card */}
        <Card className="border-0 shadow-sm">
          <Card.Body className="p-4 p-md-5">
            {/* Header */}
            <div className="mb-5">
              <h1 className="display-6 fw-bold text-primary mb-2">
                📄 Medical Reports
              </h1>
              <p className="text-muted lead mb-0">
                Upload previous medical test results, reports, or documentation
              </p>
            </div>

            {/* Form */}
            <Form onSubmit={handleSubmit(onSubmit)}>
              {/* File Upload Area */}
              <div className="mb-5">
                <Form.Group>
                  <Form.Label className="fw-semibold mb-3">
                    Upload Medical Documents
                  </Form.Label>

                  {/* Drag & Drop Area */}
                  <div
                    className="border-2 border-dashed rounded-3 p-5 text-center bg-light cursor-pointer transition"
                    onClick={() => fileInputRef.current?.click()}
                    style={{
                      borderColor: '#dee2e6',
                      cursor: 'pointer',
                      transition: 'all 0.3s ease',
                    }}
                    onDragOver={(e) => {
                      e.preventDefault();
                      e.currentTarget.style.backgroundColor = '#e7f3ff';
                      e.currentTarget.style.borderColor = '#0d6efd';
                    }}
                    onDragLeave={(e) => {
                      e.currentTarget.style.backgroundColor = '#f8f9fa';
                      e.currentTarget.style.borderColor = '#dee2e6';
                    }}
                    onDrop={(e) => {
                      e.preventDefault();
                      e.currentTarget.style.backgroundColor = '#f8f9fa';
                      e.currentTarget.style.borderColor = '#dee2e6';
                      handleFileSelect({
                        target: { files: e.dataTransfer.files },
                      });
                    }}
                  >
                    <div style={{ fontSize: '2.5rem' }} className="mb-2">
                      📤
                    </div>
                    <p className="mb-2">
                      <strong>Click to upload or drag and drop</strong>
                    </p>
                    <p className="text-muted small mb-0">
                      Supported formats: PDF, JPG, PNG, DOC, DOCX (Max 10MB each)
                    </p>
                  </div>

                  {/* Hidden File Input */}
                  <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    accept={acceptedExtensions}
                    onChange={handleFileSelect}
                    style={{ display: 'none' }}
                  />
                </Form.Group>
              </div>

              {/* Selected Files List */}
              {hasFiles && (
                <div className="mb-5">
                  <div className="d-flex justify-content-between align-items-center mb-3">
                    <p className="fw-semibold mb-0">
                      Selected Files ({selectedFiles.length}/{maxFiles})
                    </p>
                    <Button
                      variant="outline-danger"
                      size="sm"
                      onClick={clearAllFiles}
                    >
                      Clear All
                    </Button>
                  </div>

                  <ListGroup>
                    {selectedFiles.map((file, idx) => (
                      <ListGroup.Item
                        key={idx}
                        className="d-flex justify-content-between align-items-center"
                      >
                        <div className="d-flex align-items-center gap-3">
                          <span style={{ fontSize: '1.5rem' }}>
                            {file.type.includes('pdf')
                              ? '📕'
                              : file.type.includes('image')
                              ? '🖼️'
                              : '📄'}
                          </span>
                          <div>
                            <p className="mb-1 fw-semibold">{file.name}</p>
                            <small className="text-muted">
                              {formatFileSize(file.size)}
                            </small>
                          </div>
                        </div>
                        <Button
                          variant="outline-danger"
                          size="sm"
                          onClick={() => removeFile(idx)}
                        >
                          ✕
                        </Button>
                      </ListGroup.Item>
                    ))}
                  </ListGroup>
                </div>
              )}

              {/* Info Alert */}
              <Alert variant="info" className="border-0 bg-info bg-opacity-10 mb-4">
                <div className="d-flex align-items-start">
                  <span className="me-2" style={{ fontSize: '1.2rem' }}>
                    💡
                  </span>
                  <div className="small">
                    <p className="mb-2 fw-semibold">Helpful information:</p>
                    <ul className="mb-0 ps-3">
                      <li>Blood tests, X-rays, MRI, ultrasound reports</li>
                      <li>Previous consultation summaries</li>
                      <li>Lab test results with dates</li>
                      <li>Any specialist recommendations</li>
                      <li>Hospital discharge summaries</li>
                    </ul>
                  </div>
                </div>
              </Alert>

              {/* File Guidelines */}
              <Card className="bg-light border-0 mb-4">
                <Card.Body className="p-3">
                  <Row className="g-3">
                    <Col md={6}>
                      <p className="text-muted small mb-2">
                        <strong>✅ Accepted Formats</strong>
                      </p>
                      <ul className="text-muted small mb-0 ps-3">
                        <li>PDF Documents</li>
                        <li>Images (JPG, PNG)</li>
                      </ul>
                    </Col>
                    <Col md={6}>
                      <p className="text-muted small mb-2">
                        <strong>📋 Guidelines</strong>
                      </p>
                      <ul className="text-muted small mb-0 ps-3">
                        <li>Maximum 5 files</li>
                        <li>Max 20MB per file</li>
                        <li>Clear, readable scans</li>
                      </ul>
                    </Col>
                  </Row>
                </Card.Body>
              </Card>

              {/* Privacy Alert */}
              <Alert variant="success" className="border-0 bg-success bg-opacity-10 mb-4">
                <div className="d-flex align-items-start">
                  <span className="me-2" style={{ fontSize: '1.2rem' }}>
                    🔒
                  </span>
                  <small>
                    Your medical documents are encrypted and securely stored. Only
                    authorized medical professionals can access them.
                  </small>
                </div>
              </Alert>

              {/* Action Buttons */}
              <div className="d-flex justify-content-between align-items-center pt-4 border-top">
                <Button
                  variant="outline-secondary"
                  onClick={() => navigate(-1)}
                  size="lg"
                  disabled={isSubmitting}
                >
                  ← Back
                </Button>

                <div className="d-flex gap-2">
                  <Button
                    variant="outline-warning"
                    size="lg"
                    onClick={handleSkip}
                    disabled={isSubmitting}
                    className="d-flex align-items-center gap-2"
                  >
                    <span>Skip</span>
                  </Button>

                  <Button
                    type="submit"
                    variant="primary"
                    size="lg"
                    disabled={isSubmitting || !hasFiles}
                    className="d-flex align-items-center gap-2"
                  >
                    {isSubmitting ? (
                      <>
                        <Spinner animation="border" size="sm" />
                        <span>Uploading...</span>
                      </>
                    ) : (
                      <>
                        <span>Upload & Continue</span>
                        <span>→</span>
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </Form>
          </Card.Body>
        </Card>

        {/* Footer */}
        <div className="text-center mt-4">
          <small className="text-muted">
            Medical reports help us provide better diagnostic recommendations
          </small>
        </div>
      </Container>
    </div>
  );
}
