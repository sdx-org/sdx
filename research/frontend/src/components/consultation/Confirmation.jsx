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
    let summary = `${t('confirmation.summary.header')}\n`;
    summary += `${t('confirmation.summary.patientId')}: ${state.patientId}\n`;
    summary += `${t('confirmation.summary.language')}: ${state.language}\n`;
    summary += `${t('confirmation.summary.status')}: ${t('confirmation.summary.statusComplete')}\n\n`;

    summary += `=== ${t('confirmation.summary.sections.demographics')} ===\n`;
    summary += `${t('confirmation.summary.age')}: ${data.demographics.age} ${t('confirmation.summary.years')}\n`;
    summary += `${t('confirmation.summary.gender')}: ${data.demographics.gender}\n`;
    summary += `${t('confirmation.summary.weight')}: ${data.demographics.weight} ${t('confirmation.summary.kg')}\n`;
    summary += `${t('confirmation.summary.height')}: ${data.demographics.height} ${t('confirmation.summary.cm')}\n\n`;

    summary += `=== ${t('confirmation.summary.sections.lifestyle')} ===\n`;
    summary += `${t('confirmation.summary.diet')}: ${data.lifestyle.diet}\n`;
    summary += `${t('confirmation.summary.sleepHours')}: ${data.lifestyle.sleep_hours}\n`;
    summary += `${t('confirmation.summary.physicalActivity')}: ${data.lifestyle.physical_activity}\n`;
    summary += `${t('confirmation.summary.mentalExercises')}: ${data.lifestyle.mental_exercises}\n\n`;

    summary += `=== ${t('confirmation.summary.sections.symptoms')} ===\n`;
    summary += `${data.symptoms.symptoms}\n\n`;

    summary += `=== ${t('confirmation.summary.sections.mentalHealth')} ===\n`;
    summary += `${data.mental.mental_health}\n\n`;

    summary += `=== ${t('confirmation.summary.sections.medicalReports')} ===\n`;
    summary += `${t('confirmation.summary.filesUploaded')}: ${data.medicalReports.files.length}\n`;
    summary += `${t('confirmation.summary.skipped')}: ${data.medicalReports.skipped}\n\n`;

    summary += `=== ${t('confirmation.summary.sections.wearable')} ===\n`;
    summary += `${t('confirmation.summary.fileUploaded')}: ${data.wearableData.file ? t('common.yes') : t('common.no')}\n`;
    summary += `${t('confirmation.summary.skipped')}: ${data.wearableData.skipped}\n\n`;

    summary += `=== ${t('confirmation.summary.sections.diagnoses')} ===\n`;
    summary += `${t('confirmation.summary.total')}: ${data.diagnosis.selected.length}\n`;
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
      summary += `- ${name} (${t('confirmation.summary.score')}: ${score}/10)\n`;
    });
    summary += '\n';

    summary += `=== ${t('confirmation.summary.sections.exams')} ===\n`;
    summary += `${t('confirmation.summary.total')}: ${data.exams.selected.length}\n`;
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
      summary += `- ${name} (${t('confirmation.summary.score')}: ${score}/10)\n`;
    });
    summary += '\n';

    summary += `=== ${t('confirmation.summary.sections.privacy')} ===\n`;
    summary += `${t('confirmation.privacy.deidentified')}\n`;
    summary += `${t('confirmation.privacy.identifiersRemoved')}\n`;
    summary += `${t('confirmation.privacy.researchUse')}\n`;

    return summary;
  };

  return (
    <div className="bg-light min-vh-100 py-5">
      <Container>
        {/* Progress Section */}
        <div className="mb-4">
          <div className="d-flex justify-content-between align-items-center mb-2">
            <small className="text-muted fw-semibold">
              {t('steps.ofTen', { step: 9 })}
            </small>
            <small className="text-muted">
              {t('steps.percentComplete', { percent: 100 })}
            </small>
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
              {t('confirmation.title')}
            </h1>
            <p className="text-muted lead mb-4">
              {t('confirmation.subtitle')}
            </p>

            {/* Patient ID */}
              <Card className="bg-light border-0 mb-4 mx-auto" style={{ maxWidth: '400px' }}>
              <Card.Body className="p-3">
                <small className="text-muted d-block mb-2">
                  {t('confirmation.patientId')}
                </small>
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
                <h5 className="fw-bold mb-4">
                  📋 {t('confirmation.summaryTitle')}
                </h5>

                {/* Steps Completed */}
                <div className="mb-4">
                  <h6 className="fw-semibold mb-3">
                    {t('confirmation.stepsCompleted')}
                  </h6>
                  <ListGroup variant="flush">
                    <ListGroup.Item className="d-flex justify-content-between align-items-center border-0 py-2">
                      <span>
                        <span className="me-2">👤</span> {t('steps.demographics')}
                      </span>
                      <span className="text-success fw-bold">✓</span>
                    </ListGroup.Item>
                    <ListGroup.Item className="d-flex justify-content-between align-items-center border-0 py-2">
                      <span>
                        <span className="me-2">🏃</span> {t('steps.lifestyle')}
                      </span>
                      <span className="text-success fw-bold">✓</span>
                    </ListGroup.Item>
                    <ListGroup.Item className="d-flex justify-content-between align-items-center border-0 py-2">
                      <span>
                        <span className="me-2">🤒</span> {t('steps.symptoms')}
                      </span>
                      <span className="text-success fw-bold">✓</span>
                    </ListGroup.Item>
                    <ListGroup.Item className="d-flex justify-content-between align-items-center border-0 py-2">
                      <span>
                        <span className="me-2">🧠</span> {t('steps.mentalHealth')}
                      </span>
                      <span className="text-success fw-bold">✓</span>
                    </ListGroup.Item>
                    <ListGroup.Item className="d-flex justify-content-between align-items-center border-0 py-2">
                      <span>
                        <span className="me-2">📄</span> {t('steps.medicalReports')}
                      </span>
                      <span className="text-success fw-bold">✓</span>
                    </ListGroup.Item>
                    <ListGroup.Item className="d-flex justify-content-between align-items-center border-0 py-2">
                      <span>
                        <span className="me-2">⌚</span> {t('steps.wearableData')}
                      </span>
                      <span className="text-success fw-bold">✓</span>
                    </ListGroup.Item>
                    <ListGroup.Item className="d-flex justify-content-between align-items-center border-0 py-2">
                      <span>
                        <span className="me-2">🔬</span> {t('steps.diagnosis')}
                      </span>
                      <span className="text-success fw-bold">✓</span>
                    </ListGroup.Item>
                    <ListGroup.Item className="d-flex justify-content-between align-items-center border-0 py-2">
                      <span>
                        <span className="me-2">🧪</span> {t('steps.exams')}
                      </span>
                      <span className="text-success fw-bold">✓</span>
                    </ListGroup.Item>
                  </ListGroup>
                </div>

                {/* Selected Items Summary */}
                <div className="border-top pt-4">
                  <h6 className="fw-semibold mb-3">
                    {t('confirmation.selectionsTitle')}
                  </h6>
                  <Row className="g-3">
                    <Col sm={6}>
                      <div className="p-3 bg-light rounded">
                        <small className="text-muted d-block mb-1">
                          {t('confirmation.selectedDiagnoses')}
                        </small>
                        <h5 className="mb-0 text-primary fw-bold">
                          {state.formData.diagnosis.selected.length}
                        </h5>
                      </div>
                    </Col>
                    <Col sm={6}>
                      <div className="p-3 bg-light rounded">
                        <small className="text-muted d-block mb-1">
                          {t('confirmation.selectedExams')}
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
                <h5 className="fw-bold mb-4">
                  🔒 {t('confirmation.privacy.title')}
                </h5>

                <Alert variant="success" className="border-0 mb-3">
                  <div className="d-flex align-items-start">
                    <span className="me-2" style={{ fontSize: '1.2rem' }}>
                      ✅
                    </span>
                    <div>
                      <p className="fw-semibold mb-2">
                        {t('confirmation.privacy.deidentifiedTitle')}
                      </p>
                      <small>
                        {t('confirmation.privacy.deidentifiedText')}
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
                      <p className="fw-semibold mb-2">
                        {t('confirmation.privacy.researchTitle')}
                      </p>
                      <small>
                        {t('confirmation.privacy.researchText')}
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
                      <p className="fw-semibold mb-2">
                        {t('confirmation.privacy.storageTitle')}
                      </p>
                      <small>
                        {t('confirmation.privacy.storageText')}
                      </small>
                    </div>
                  </div>
                </Alert>
              </Card.Body>
            </Card>

            {/* What Happens Next */}
            <Card className="border-0 shadow-sm mb-4">
              <Card.Body className="p-4">
                <h5 className="fw-bold mb-4">
                  📌 {t('confirmation.next.title')}
                </h5>

                <div className="mb-3">
                  <div className="d-flex align-items-start mb-3">
                    <div
                      className="bg-primary text-white rounded-circle d-flex align-items-center justify-content-center me-3"
                      style={{ width: '32px', height: '32px', minWidth: '32px' }}
                    >
                      1
                    </div>
                    <div>
                      <p className="fw-semibold mb-1">
                        {t('confirmation.next.step1Title')}
                      </p>
                      <small className="text-muted">
                        {t('confirmation.next.step1Text')}
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
                      <p className="fw-semibold mb-1">
                        {t('confirmation.next.step2Title')}
                      </p>
                      <small className="text-muted">
                        {t('confirmation.next.step2Text')}
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
                      <p className="fw-semibold mb-1">
                        {t('confirmation.next.step3Title')}
                      </p>
                      <small className="text-muted">
                        {t('confirmation.next.step3Text')}
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
                <span>📊 {t('confirmation.actions.viewDashboard')}</span>
              </Button>

              <Button
                variant="outline-secondary"
                size="lg"
                onClick={handleDownloadSummary}
                className="d-flex align-items-center justify-content-center gap-2"
              >
                <span>⬇️ {t('confirmation.actions.downloadSummary')}</span>
              </Button>

              <Button
                variant="outline-secondary"
                size="lg"
                onClick={handleCreateNewConsultation}
                className="d-flex align-items-center justify-content-center gap-2"
              >
                <span>➕ {t('confirmation.actions.newConsultation')}</span>
              </Button>
            </div>
          </Col>
        </Row>

        {/* Footer Section */}
        <Row className="mb-4">
          <Col lg={8} className="mx-auto">
            <Card className="border-0 bg-light">
              <Card.Body className="p-4 text-center">
                <h6 className="fw-bold mb-2">
                  {t('confirmation.thanksTitle')}
                </h6>
                <small className="text-muted">
                  {t('confirmation.thanksText')}
                </small>
              </Card.Body>
            </Card>
          </Col>
        </Row>

        {/* Footer */}
        <div className="text-center mt-5">
          <small className="text-muted">
            {t('confirmation.consultationId')}: <code>{state.patientId}</code>
          </small>
          <br />
          <small className="text-muted">
            {t('confirmation.statusLabel')}: ✅ {t('confirmation.statusComplete')}
          </small>
        </div>
      </Container>
    </div>
  );
}
