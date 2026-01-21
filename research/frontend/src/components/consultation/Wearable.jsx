import React, { useState, useRef, useEffect } from 'react';
import { useForm } from 'react-hook-form';
import {
  Form,
  Button,
  Container,
  Card,
  ProgressBar,
  Alert,
  Spinner,
  Row,
  Col,
  Badge,
} from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useConsultation, consultationActions } from '../../context/ConsultationContext';
import consultationAPI from '../../services/api';

export default function Wearable() {
  const navigate = useNavigate();
  const { t } = useTranslation();
  const { state, dispatch } = useConsultation();
  const fileInputRef = useRef(null);
  const {
    handleSubmit,
    formState: { isSubmitting },
  } = useForm();

  useEffect(()=>{
    if(!state.formData.wearableData){
      dispatch(consultationActions.updateWearableData({
        file:null,
        skipped:false,
      }))
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  },[]);
  const [apiError, setApiError] = useState(null);
  const [selectedFile, setSelectedFile] = useState(
    state.formData.wearableData?.file || null
  );

  const acceptedFormats = [
    'text/csv',
    'application/json',
  ];

  const acceptedExtensions = '.csv, .json';
  const maxFileSize = 50 * 1024 * 1024; // 50MB for data exports
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes, k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
  };

  const getFileIcon = (type) => {
    if (type.includes('csv')) return '📊';
    if (type.includes('json')) return '📋';
  };

  const handleFileSelect = (event) => {
    const files = Array.from(event.target.files);
    setApiError(null);

    if (files.length === 0) return;

    const file = files[0]; // Only one file for wearable data

    // Check format
    if (!acceptedFormats.includes(file.type)) {
      setApiError(
        `Invalid file format: ${file.name}. Accepted: CSV, JSON`
      );
      return;
    }

    // Check size
    if (file.size > maxFileSize) {
      setApiError(
        `File too large: ${file.name} (${formatFileSize(file.size)}). Max: 50MB`
      );
      return;
    }

    setSelectedFile(file);
  };

  const removeFile = () => {
    setSelectedFile(null);
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
        consultationActions.updateWearableData({
          file: null,
          skipped: true,
        })
      );

      // Call backend API to skip
      await consultationAPI.skipWearableData(state.patientId);

      // Update current step in context
      dispatch(consultationActions.setCurrentStep('diagnosis'));

      // Navigate to next step
      navigate('/diagnosis');
    } catch (err) {
      console.error('Error skipping wearable data:', err);
      setApiError(err.message || 'Failed to skip wearable data. Please try again.');
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

      if (!selectedFile) {
        setApiError('Please select a file or click Skip');
        return;
      }

      // Create FormData for multipart upload
      const formData = new FormData();
      formData.append('file', selectedFile);

      // Update local state
      dispatch(
        consultationActions.updateWearableData({
          file: {
            name: selectedFile.name,
            size: selectedFile.size,
            type: selectedFile.type,
          },
          skipped: false,
        })
      );

      // Call backend API to upload
      await consultationAPI.uploadWearableData(state.patientId, formData);

      // Update current step in context
      dispatch(consultationActions.setCurrentStep('diagnosis'));

      // Navigate to next step
      navigate('/diagnosis');
    } catch (err) {
      console.error('Error uploading wearable data:', err);
      setApiError(err.message || 'Failed to upload wearable data. Please try again.');
      window.scrollTo(0, 0);
    }
  };
  const hasFile = selectedFile !== null;

  return (
    <div className="bg-light min-vh-100 py-4">
      <Container>
        {/* Progress Section */}
        <div className="mb-4">
          <div className="d-flex justify-content-between align-items-center mb-2">
            <small className="text-muted fw-semibold">Step 6 of 10</small>
            <small className="text-muted">60% Complete</small>
          </div>
          <ProgressBar now={60} style={{ height: '8px' }} className="mb-3" />
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
                ⌚ Wearable Device Data
              </h1>
              <p className="text-muted lead mb-0">
                Upload data from fitness trackers, smartwatches, or health monitoring apps
              </p>
            </div>

            {/* Form */}
            <Form onSubmit={handleSubmit(onSubmit)}>
              {/* Supported Devices Info */}
              <div className="mb-5">
                <p className="text-muted fw-semibold mb-3">
                  Supported devices & sources:
                </p>
                <Row className="g-2">
                  {[
                    { icon: '⌚', name: 'Apple Watch' },
                    { icon: '🔴', name: 'Fitbit' },
                    { icon: '📱', name: 'Samsung Health' },
                    { icon: '🎯', name: 'Garmin' },
                    { icon: '📊', name: 'Google Fit' },
                    { icon: '💓', name: 'Oura Ring' },
                    { icon: '🏃', name: 'Strava' },
                    { icon: '📈', name: 'MyFitnessPal' },
                  ].map((device, idx) => (
                    <Col xs={6} sm={4} md={3} key={idx} className="mb-2">
                      <div className="d-flex align-items-center gap-2 p-2 bg-light rounded">
                        <span style={{ fontSize: '1.5rem' }}>{device.icon}</span>
                        <small className="fw-semibold">{device.name}</small>
                      </div>
                    </Col>
                  ))}
                </Row>
              </div>

              {/* File Upload Area */}
              <div className="mb-5">
                <Form.Group>
                  <Form.Label className="fw-semibold mb-3">
                    Upload Wearable Data Export
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
                      Supported formats: CSV, JSON, PDF, XLSX, ZIP (Max 50MB)
                    </p>
                  </div>

                  {/* Hidden File Input */}
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept={acceptedExtensions}
                    onChange={handleFileSelect}
                    style={{ display: 'none' }}
                  />
                </Form.Group>
              </div>

              {/* Selected File Display */}
              {hasFile && (
                <div className="mb-5">
                  <p className="fw-semibold mb-3">Selected File</p>
                  <Card className="border-0 bg-light">
                    <Card.Body className="d-flex justify-content-between align-items-center p-3">
                      <div className="d-flex align-items-center gap-3">
                        <span style={{ fontSize: '2rem' }}>
                          {getFileIcon(selectedFile.type)}
                        </span>
                        <div>
                          <p className="mb-1 fw-semibold">{selectedFile.name}</p>
                          <small className="text-muted">
                            {formatFileSize(selectedFile.size)}
                          </small>
                        </div>
                      </div>
                      <Button
                        variant="outline-danger"
                        size="sm"
                        onClick={removeFile}
                      >
                        ✕
                      </Button>
                    </Card.Body>
                  </Card>
                </div>
              )}

              {/* Info Alert */}
              <Alert variant="info" className="border-0 bg-info bg-opacity-10 mb-4">
                <div className="d-flex align-items-start">
                  <span className="me-2" style={{ fontSize: '1.2rem' }}>
                    💡
                  </span>
                  <div className="small">
                    <p className="mb-2 fw-semibold">How to export data:</p>
                    <ul className="mb-0 ps-3">
                      <li>
                        <strong>Apple Health:</strong> Settings → Health Data → Export
                      </li>
                      <li>
                        <strong>Fitbit:</strong> Account → Download Your Data
                      </li>
                      <li>
                        <strong>Google Fit:</strong> Settings → Download Your Data
                      </li>
                      <li>
                        <strong>Garmin:</strong> Device → Export to CSV
                      </li>
                      <li>
                        <strong>Other apps:</strong> Look for Export or Share options
                      </li>
                    </ul>
                  </div>
                </div>
              </Alert>

              {/* Data Types Info */}
              <Card className="bg-light border-0 mb-4">
                <Card.Body className="p-3">
                  <p className="text-muted small mb-2">
                    <strong>Helpful data to include:</strong>
                  </p>
                  <ul className="text-muted small mb-0 ps-3">
                    <li>Daily steps and activity levels</li>
                    <li>Heart rate data (resting and active)</li>
                    <li>Sleep patterns and duration</li>
                    <li>Calorie burn information</li>
                    <li>Distance and workout data</li>
                    <li>Blood oxygen and stress levels (if available)</li>
                    <li>Any health metrics from past 1-3 months</li>
                  </ul>
                </Card.Body>
              </Card>

              {/* File Format Guidelines */}
              <Card className="bg-light border-0 mb-4">
                <Card.Body className="p-3">
                  <Row className="g-3">
                    <Col md={6}>
                      <p className="text-muted small mb-2">
                        <strong>✅ Accepted Formats</strong>
                      </p>
                      <ul className="text-muted small mb-0 ps-3">
                        <li>CSV (Comma Separated)</li>
                        <li>JSON (Data Export)</li>
                      </ul>
                    </Col>
                    <Col md={6}>
                      <p className="text-muted small mb-2">
                        <strong>📋 Guidelines</strong>
                      </p>
                      <ul className="text-muted small mb-0 ps-3">
                        <li>One file per upload</li>
                        <li>Max 50MB file size</li>
                        <li>Recent data preferred</li>
                        <li>Multiple metrics OK</li>
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
                    Your wearable data is encrypted and securely stored. We use it only
                    to provide personalized health insights and recommendations.
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
                    disabled={isSubmitting || !hasFile}
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
            Wearable data helps us understand your activity patterns and health trends
          </small>
        </div>
      </Container>
    </div>
  );
}
