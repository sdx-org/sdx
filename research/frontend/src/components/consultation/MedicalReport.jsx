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
      setApiError(t('medicalReports.errors.maxFiles', { max: maxFiles }));
      return;
    }

    // Validate each file
    const validFiles = [];
    for (const file of files) {
      // Check format
      if (!acceptedFormats.includes(file.type)) {
        setApiError(
          t('medicalReports.errors.invalidFormat', { name: file.name })
        );
        continue;
      }

      // Check size
      if (file.size > maxFileSize) {
        setApiError(
          t('medicalReports.errors.fileTooLarge', {
            name: file.name,
            size: formatFileSize(file.size),
            max: formatFileSize(maxFileSize),
          })
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
        throw new Error(t('errors.patientIdMissing'));
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
      setApiError(err.message || t('errors.skipMedicalReportsFailed'));
      window.scrollTo(0, 0);
    }
  };
  const onSubmit = async () => {
    try {
      setApiError(null);

      // Validate patient ID exists
      if (!state.patientId) {
        throw new Error(t('errors.patientIdMissing'));
      }

      if (selectedFiles.length === 0) {
        setApiError(t('medicalReports.errors.noFilesSelected'));
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
      setApiError(err.message || t('errors.uploadMedicalReportsFailed'));
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
            <small className="text-muted fw-semibold">
              {t('steps.ofTen', { step: 5 })}
            </small>
            <small className="text-muted">
              {t('steps.percentComplete', { percent: 50 })}
            </small>
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
            <Alert.Heading>{t('errors.title')}</Alert.Heading>
            <p className="mb-0">{apiError}</p>
          </Alert>
        )}

        {/* Main Card */}
        <Card className="border-0 shadow-sm">
          <Card.Body className="p-4 p-md-5">
            {/* Header */}
            <div className="mb-5">
              <h1 className="display-6 fw-bold text-primary mb-2">
                📄 {t('medicalReports.title')}
              </h1>
              <p className="text-muted lead mb-0">
                {t('medicalReports.subtitle')}
              </p>
            </div>

            {/* Form */}
            <Form onSubmit={handleSubmit(onSubmit)}>
              {/* File Upload Area */}
              <div className="mb-5">
                <Form.Group>
                  <Form.Label className="fw-semibold mb-3">
                    {t('medicalReports.uploadLabel')}
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
                      <strong>{t('medicalReports.dropTitle')}</strong>
                    </p>
                    <p className="text-muted small mb-0">
                      {t('medicalReports.dropSubtitle')}
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
                      {t('medicalReports.selectedFiles', {
                        count: selectedFiles.length,
                        max: maxFiles,
                      })}
                    </p>
                    <Button
                      variant="outline-danger"
                      size="sm"
                      onClick={clearAllFiles}
                    >
                      {t('common.clearAll')}
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
                    <p className="mb-2 fw-semibold">
                      {t('medicalReports.helpfulTitle')}
                    </p>
                    <ul className="mb-0 ps-3">
                      {t('medicalReports.helpfulList', { returnObjects: true }).map(
                        (item, idx) => (
                          <li key={idx}>{item}</li>
                        )
                      )}
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
                        <strong>{t('medicalReports.guidelinesFormatsTitle')}</strong>
                      </p>
                      <ul className="text-muted small mb-0 ps-3">
                        {t('medicalReports.guidelinesFormatsList', { returnObjects: true }).map(
                          (item, idx) => (
                            <li key={idx}>{item}</li>
                          )
                        )}
                      </ul>
                    </Col>
                    <Col md={6}>
                      <p className="text-muted small mb-2">
                        <strong>{t('medicalReports.guidelinesRulesTitle')}</strong>
                      </p>
                      <ul className="text-muted small mb-0 ps-3">
                        {t('medicalReports.guidelinesRulesList', { returnObjects: true }).map(
                          (item, idx) => (
                            <li key={idx}>{item}</li>
                          )
                        )}
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
                    {t('medicalReports.privacyText')}
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
                  ← {t('common.back')}
                </Button>

                <div className="d-flex gap-2">
                  <Button
                    variant="outline-warning"
                    size="lg"
                    onClick={handleSkip}
                    disabled={isSubmitting}
                    className="d-flex align-items-center gap-2"
                  >
                    <span>{t('common.skip')}</span>
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
                        <span>{t('common.uploading')}</span>
                      </>
                    ) : (
                      <>
                        <span>{t('common.uploadContinue')}</span>
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
            {t('medicalReports.footer')}
          </small>
        </div>
      </Container>
    </div>
  );
}
