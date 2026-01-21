import React, { useEffect } from 'react';
import {
  Container,
  Card,
  Button,
  Alert,
  ProgressBar,
  ListGroup,
  Row,
  Col,
} from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useConsultation, consultationActions } from '../../context/ConsultationContext';


export default function Confirmation() {
  const navigate = useNavigate();
  const { t } = useTranslation();
  const { state, dispatch } = useConsultation();

  useEffect(() => {
    // Verify patient is complete
    if (!state.isComplete || !state.patientId) {
      navigate('/');
    }
  }, [state.isComplete, state.patientId, navigate]);

  const handleViewDashboard = () => {
    navigate('/dashboard');
  };

  const handleCreateNewConsultation = async () => {
    try {
      // Reset consultation state for new patient
      dispatch(consultationActions.resetState());
      navigate('/');
    } catch (err) {
      console.error('Error creating new consultation:', err);
    }
  };

  const handleDownloadSummary = () => {
    // Generate summary text
    const summary = generateConsultationSummary();

    // Create blob and download
    const element = document.createElement('a');
    element.setAttribute(
      'href',
      'data:text/plain;charset=utf-8,' + encodeURIComponent(summary)
    );
    element.setAttribute('download', `consultation_${state.patientId}_summary.txt`);
    element.style.display = 'none';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const generateConsultationSummary = () => {
    const data = state.formData;
    let summary = `CONSULTATION SUMMARY\n`;
    summary += `Patient ID: ${state.patientId}\n`;
    summary += `Language: ${state.language}\n`;
    summary += `Status: COMPLETE\n\n`;

    summary += `=== DEMOGRAPHICS ===\n`;
    summary += `Age: ${data.demographics.age} years\n`;
    summary += `Gender: ${data.demographics.gender}\n`;
    summary += `Weight: ${data.demographics.weight} kg\n`;
    summary += `Height: ${data.demographics.height} cm\n\n`;

    summary += `=== LIFESTYLE ===\n`;
    summary += `Diet: ${data.lifestyle.diet}\n`;
    summary += `Sleep Hours: ${data.lifestyle.sleep_hours}\n`;
    summary += `Physical Activity: ${data.lifestyle.physical_activity}\n`;
    summary += `Mental Exercises: ${data.lifestyle.mental_exercises}\n\n`;

    summary += `=== SYMPTOMS ===\n`;
    summary += `${data.symptoms.symptoms}\n\n`;

    summary += `=== MENTAL HEALTH ===\n`;
    summary += `${data.mental.mental_health}\n\n`;

    summary += `=== MEDICAL REPORTS ===\n`;
    summary += `Files Uploaded: ${data.medicalReports.files.length}\n`;
    summary += `Skipped: ${data.medicalReports.skipped}\n\n`;

    summary += `=== WEARABLE DATA ===\n`;
    summary += `File Uploaded: ${data.wearableData.file ? 'Yes' : 'No'}\n`;
    summary += `Skipped: ${data.wearableData.skipped}\n\n`;

    summary += `=== SELECTED DIAGNOSES ===\n`;
    summary += `Total: ${data.diagnosis.selected.length}\n`;
    data.diagnosis.selected.forEach((name) => {
      const evaluations = data.diagnosis.evaluations[name];
      const score = evaluations
        ? Math.round(
            (evaluations.accuracy +
              evaluations.relevance +
              evaluations.usefulness +
              evaluations.coherence) /
              4
          )
        : 0;
      summary += `- ${name} (Score: ${score}/10)\n`;
    });
    summary += '\n';

    summary += `=== SELECTED EXAMS ===\n`;
    summary += `Total: ${data.exams.selected.length}\n`;
    data.exams.selected.forEach((name) => {
      const evaluations = data.exams.evaluations[name];
      const score = evaluations
        ? Math.round(
            (evaluations.accuracy +
              evaluations.relevance +
              evaluations.usefulness +
              evaluations.coherence +
              evaluations.safety) /
              5
          )
        : 0;
      summary += `- ${name} (Score: ${score}/10)\n`;
    });
    summary += '\n';

    summary += `=== PRIVACY & DEIDENTIFICATION ===\n`;
    summary += `Your data has been deidentified and will be securely archived.\n`;
    summary += `All personal identifiers have been removed.\n`;
    summary += `This data will be used solely for research purposes.\n`;

    return summary;
  };

  return (
    <div className="bg-light min-vh-100 py-5">
      <Container>
        {/* Progress Section */}
        <div className="mb-4">
          <div className="d-flex justify-content-between align-items-center mb-2">
            <small className="text-muted fw-semibold">Step 9 of 10</small>
            <small className="text-muted">100% Complete</small>
          </div>
          <ProgressBar
            now={100}
            style={{ height: '8px' }}
            className="mb-3"
            variant="success"
          />
        </div>

        {/* Completion Card */}
        <Card className="border-0 shadow-lg mb-4">
          <Card.Body className="p-5 text-center">
            {/* Success Icon */}
            <div style={{ fontSize: '4rem', marginBottom: '1rem' }}>
              ✅
            </div>

            {/* Header */}
            <h1 className="display-4 fw-bold text-success mb-2">
              Consultation Complete!
            </h1>
            <p className="text-muted lead mb-4">
              Thank you for completing your health consultation
            </p>

            {/* Patient ID */}
            <Card className="bg-light border-0 mb-4 mx-auto" style={{ maxWidth: '400px' }}>
              <Card.Body className="p-3">
                <small className="text-muted d-block mb-2">Patient ID</small>
                <code className="fs-6">{state.patientId}</code>
              </Card.Body>
            </Card>
          </Card.Body>
        </Card>

        {/* Summary Section */}
        <Row className="mb-4">
          <Col lg={8} className="mx-auto">
            {/* Completion Summary */}
            <Card className="border-0 shadow-sm mb-4">
              <Card.Body className="p-4">
                <h5 className="fw-bold mb-4">📋 Consultation Summary</h5>

                {/* Steps Completed */}
                <div className="mb-4">
                  <h6 className="fw-semibold mb-3">Steps Completed:</h6>
                  <ListGroup variant="flush">
                    <ListGroup.Item className="d-flex justify-content-between align-items-center border-0 py-2">
                      <span>
                        <span className="me-2">👤</span> Demographics
                      </span>
                      <span className="text-success fw-bold">✓</span>
                    </ListGroup.Item>
                    <ListGroup.Item className="d-flex justify-content-between align-items-center border-0 py-2">
                      <span>
                        <span className="me-2">🏃</span> Lifestyle
                      </span>
                      <span className="text-success fw-bold">✓</span>
                    </ListGroup.Item>
                    <ListGroup.Item className="d-flex justify-content-between align-items-center border-0 py-2">
                      <span>
                        <span className="me-2">🤒</span> Symptoms
                      </span>
                      <span className="text-success fw-bold">✓</span>
                    </ListGroup.Item>
                    <ListGroup.Item className="d-flex justify-content-between align-items-center border-0 py-2">
                      <span>
                        <span className="me-2">🧠</span> Mental Health
                      </span>
                      <span className="text-success fw-bold">✓</span>
                    </ListGroup.Item>
                    <ListGroup.Item className="d-flex justify-content-between align-items-center border-0 py-2">
                      <span>
                        <span className="me-2">📄</span> Medical Reports
                      </span>
                      <span className="text-success fw-bold">✓</span>
                    </ListGroup.Item>
                    <ListGroup.Item className="d-flex justify-content-between align-items-center border-0 py-2">
                      <span>
                        <span className="me-2">⌚</span> Wearable Data
                      </span>
                      <span className="text-success fw-bold">✓</span>
                    </ListGroup.Item>
                    <ListGroup.Item className="d-flex justify-content-between align-items-center border-0 py-2">
                      <span>
                        <span className="me-2">🔬</span> AI Diagnosis
                      </span>
                      <span className="text-success fw-bold">✓</span>
                    </ListGroup.Item>
                    <ListGroup.Item className="d-flex justify-content-between align-items-center border-0 py-2">
                      <span>
                        <span className="me-2">🧪</span> Recommended Exams
                      </span>
                      <span className="text-success fw-bold">✓</span>
                    </ListGroup.Item>
                  </ListGroup>
                </div>

                {/* Selected Items Summary */}
                <div className="border-top pt-4">
                  <h6 className="fw-semibold mb-3">Selections Summary:</h6>
                  <Row className="g-3">
                    <Col sm={6}>
                      <div className="p-3 bg-light rounded">
                        <small className="text-muted d-block mb-1">
                          Selected Diagnoses
                        </small>
                        <h5 className="mb-0 text-primary fw-bold">
                          {state.formData.diagnosis.selected.length}
                        </h5>
                      </div>
                    </Col>
                    <Col sm={6}>
                      <div className="p-3 bg-light rounded">
                        <small className="text-muted d-block mb-1">
                          Recommended Exams
                        </small>
                        <h5 className="mb-0 text-primary fw-bold">
                          {state.formData.exams.selected.length}
                        </h5>
                      </div>
                    </Col>
                  </Row>
                </div>
              </Card.Body>
            </Card>

            {/* Privacy & Deidentification */}
            <Card className="border-0 shadow-sm mb-4">
              <Card.Body className="p-4">
                <h5 className="fw-bold mb-4">🔒 Privacy & Data Protection</h5>

                <Alert variant="success" className="border-0 mb-3">
                  <div className="d-flex align-items-start">
                    <span className="me-2" style={{ fontSize: '1.2rem' }}>
                      ✅
                    </span>
                    <div>
                      <p className="fw-semibold mb-2">Data Deidentified</p>
                      <small>
                        Your consultation data has been automatically deidentified.
                        All personal identifiers have been removed and will not be
                        stored with your medical information.
                      </small>
                    </div>
                  </div>
                </Alert>

                <Alert variant="info" className="border-0 mb-3">
                  <div className="d-flex align-items-start">
                    <span className="me-2" style={{ fontSize: '1.2rem' }}>
                      📋
                    </span>
                    <div>
                      <p className="fw-semibold mb-2">Research Use Only</p>
                      <small>
                        Your deidentified data will be securely archived and used
                        solely for improving AI diagnostic and exam recommendation
                        systems. No clinical decisions will be made based on this
                        data.
                      </small>
                    </div>
                  </div>
                </Alert>

                <Alert variant="info" className="border-0">
                  <div className="d-flex align-items-start">
                    <span className="me-2" style={{ fontSize: '1.2rem' }}>
                      🛡️
                    </span>
                    <div>
                      <p className="fw-semibold mb-2">Secure Storage</p>
                      <small>
                        Your data is encrypted and stored in secure, HIPAA-compliant
                        facilities. Access is restricted to authorized research
                        personnel only.
                      </small>
                    </div>
                  </div>
                </Alert>
              </Card.Body>
            </Card>

            {/* What Happens Next */}
            <Card className="border-0 shadow-sm mb-4">
              <Card.Body className="p-4">
                <h5 className="fw-bold mb-4">📌 What Happens Next?</h5>

                <div className="mb-3">
                  <div className="d-flex align-items-start mb-3">
                    <div
                      className="bg-primary text-white rounded-circle d-flex align-items-center justify-content-center me-3"
                      style={{ width: '32px', height: '32px', minWidth: '32px' }}
                    >
                      1
                    </div>
                    <div>
                      <p className="fw-semibold mb-1">Review by Healthcare Professional</p>
                      <small className="text-muted">
                        Your AI-generated diagnoses and recommended exams will be
                        reviewed by a qualified healthcare professional.
                      </small>
                    </div>
                  </div>

                  <div className="d-flex align-items-start mb-3">
                    <div
                      className="bg-primary text-white rounded-circle d-flex align-items-center justify-content-center me-3"
                      style={{ width: '32px', height: '32px', minWidth: '32px' }}
                    >
                      2
                    </div>
                    <div>
                      <p className="fw-semibold mb-1">Clinical Feedback</p>
                      <small className="text-muted">
                        You will receive feedback from the healthcare professional
                        regarding the AI recommendations and suggested next steps.
                      </small>
                    </div>
                  </div>

                  <div className="d-flex align-items-start">
                    <div
                      className="bg-primary text-white rounded-circle d-flex align-items-center justify-content-center me-3"
                      style={{ width: '32px', height: '32px', minWidth: '32px' }}
                    >
                      3
                    </div>
                    <div>
                      <p className="fw-semibold mb-1">Improved AI Models</p>
                      <small className="text-muted">
                        Your ratings and evaluations help train better AI diagnostic
                        systems that will benefit future patients.
                      </small>
                    </div>
                  </div>
                </div>
              </Card.Body>
            </Card>

            {/* Action Buttons */}
            <div className="d-grid gap-2 mb-3">
              <Button
                variant="primary"
                size="lg"
                onClick={handleViewDashboard}
                className="d-flex align-items-center justify-content-center gap-2"
              >
                <span>📊 View Dashboard</span>
              </Button>

              <Button
                variant="outline-secondary"
                size="lg"
                onClick={handleDownloadSummary}
                className="d-flex align-items-center justify-content-center gap-2"
              >
                <span>⬇️ Download Summary</span>
              </Button>

              <Button
                variant="outline-secondary"
                size="lg"
                onClick={handleCreateNewConsultation}
                className="d-flex align-items-center justify-content-center gap-2"
              >
                <span>➕ New Consultation</span>
              </Button>
            </div>
          </Col>
        </Row>

        {/* Footer Section */}
        <Row className="mb-4">
          <Col lg={8} className="mx-auto">
            <Card className="border-0 bg-light">
              <Card.Body className="p-4 text-center">
                <h6 className="fw-bold mb-2">Thank You for Contributing to Research</h6>
                <small className="text-muted">
                  Your participation in this AI-driven consultation system helps us
                  improve healthcare diagnostics and benefit future patients worldwide.
                </small>
              </Card.Body>
            </Card>
          </Col>
        </Row>

        {/* Footer */}
        <div className="text-center mt-5">
          <small className="text-muted">
            Consultation ID: <code>{state.patientId}</code>
          </small>
          <br />
          <small className="text-muted">Status: ✅ COMPLETE</small>
        </div>
      </Container>
    </div>
  );
}
