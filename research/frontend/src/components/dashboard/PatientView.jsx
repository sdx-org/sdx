import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Container,
  Card,
  Button,
  Spinner,
  Alert,
  Row,
  Col,
  Badge,
  Nav,
  Tab,
} from 'react-bootstrap';
import { useTranslation } from 'react-i18next';
import consultationAPI from '../../services/api';

export default function PatientView() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { t, i18n } = useTranslation();
  const [patient, setPatient] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadPatient = async () => {
      try {
        setIsLoading(true);
        setError(null);
        const data = await consultationAPI.getConsultationStatus(id);

        if (!data) {
          throw new Error(t('patientView.errors.notFound'));
        }

        setPatient(data);
      } catch (err) {
        console.error('Error loading patient:', err);
        setError(err.message || t('patientView.errors.loadFailed'));
      } finally {
        setIsLoading(false);
      }
    };

    loadPatient();
  }, [id]);

  const formatDate = (dateString) => {
    if (!dateString) return t('common.na');
    try {
      return new Date(dateString).toLocaleDateString(i18n.language || 'en', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
      });
    } catch {
      return dateString;
    }
  };

  const renderDemographics = (formData) => {
    if (!formData?.demographics) {
      return <p className="text-muted">{t('common.noData')}</p>;
    }

    const { age, gender, weight, height } = formData.demographics;
    return (
      <div>
        <Row className="g-3">
          <Col md={6}>
            <p>
              <strong>{t('patientView.demographics.age')}:</strong>{' '}
              {age ? t('patientView.demographics.ageValue', { age }) : t('common.notProvided')}
            </p>
          </Col>
          <Col md={6}>
            <p>
              <strong>{t('patientView.demographics.gender')}:</strong>{' '}
              {gender || t('common.notProvided')}
            </p>
          </Col>
          <Col md={6}>
            <p>
              <strong>{t('patientView.demographics.weight')}:</strong>{' '}
              {weight ? t('patientView.demographics.weightValue', { weight }) : t('common.notProvided')}
            </p>
          </Col>
          <Col md={6}>
            <p>
              <strong>{t('patientView.demographics.height')}:</strong>{' '}
              {height ? t('patientView.demographics.heightValue', { height }) : t('common.notProvided')}
            </p>
          </Col>
        </Row>
      </div>
    );
  };

  const renderLifestyle = (formData) => {
    if (!formData?.lifestyle) {
      return <p className="text-muted">{t('common.noData')}</p>;
    }

    const { diet, sleep_hours, physical_activity, mental_exercises } =
      formData.lifestyle;
    return (
      <div>
        <Row className="g-3">
          <Col md={6}>
            <p>
              <strong>{t('patientView.lifestyle.diet')}:</strong>{' '}
              {diet || t('common.notProvided')}
            </p>
          </Col>
          <Col md={6}>
            <p>
              <strong>{t('patientView.lifestyle.sleepHours')}:</strong>{' '}
              {sleep_hours
                ? t('patientView.lifestyle.sleepHoursValue', { hours: sleep_hours })
                : t('common.notProvided')}
            </p>
          </Col>
          <Col md={6}>
            <p>
              <strong>{t('patientView.lifestyle.physicalActivity')}:</strong>{' '}
              {physical_activity || t('common.notProvided')}
            </p>
          </Col>
          <Col md={6}>
            <p>
              <strong>{t('patientView.lifestyle.mentalExercises')}:</strong>{' '}
              {mental_exercises || t('common.notProvided')}
            </p>
          </Col>
        </Row>
      </div>
    );
  };

  const renderSymptoms = (formData) => {
    if (!formData?.symptoms) {
      return <p className="text-muted">{t('common.noData')}</p>;
    }

    return (
      <div>
        <p>
          <strong>{t('patientView.symptoms.title')}:</strong>
        </p>
        <p className="bg-light p-3 rounded">
          {formData.symptoms.symptoms || t('common.notProvided')}
        </p>
      </div>
    );
  };

  const renderMentalHealth = (formData) => {
    if (!formData?.mental) {
      return <p className="text-muted">{t('common.noData')}</p>;
    }

    return (
      <div>
        <p>
          <strong>{t('patientView.mental.title')}:</strong>
        </p>
        <p className="bg-light p-3 rounded">
          {formData.mental.mental_health || t('common.notProvided')}
        </p>
      </div>
    );
  };

  const renderMedicalReports = (formData) => {
    if (!formData?.medicalReports) {
      return <p className="text-muted">{t('common.noData')}</p>;
    }

    const { files, skipped } = formData.medicalReports;

    if (skipped) {
      return (
        <Alert variant="info">
          <strong>{t('patientView.skipped')}:</strong>{' '}
          {t('patientView.medicalReports.skipped')}
        </Alert>
      );
    }

    if (!files || files.length === 0) {
      return <p className="text-muted">{t('patientView.medicalReports.noFiles')}</p>;
    }

    return (
      <div>
        <p>
          <strong>{t('patientView.medicalReports.title')}:</strong>
        </p>
        <ul className="list-group">
          {files.map((file, idx) => (
            <li key={idx} className="list-group-item">
              📄 {file.name || t('patientView.fileFallback', { index: idx + 1 })}
            </li>
          ))}
        </ul>
      </div>
    );
  };

  const renderWearableData = (formData) => {
    if (!formData?.wearableData) {
      return <p className="text-muted">{t('common.noData')}</p>;
    }

    const { file, skipped } = formData.wearableData;

    if (skipped) {
      return (
        <Alert variant="info">
          <strong>{t('patientView.skipped')}:</strong>{' '}
          {t('patientView.wearable.skipped')}
        </Alert>
      );
    }

    if (!file) {
      return <p className="text-muted">{t('patientView.wearable.noFile')}</p>;
    }

    return (
      <div>
        <p>
          <strong>{t('patientView.wearable.title')}:</strong>
        </p>
        <p className="bg-light p-3 rounded">
          📊 {file.name || t('patientView.wearable.fallback')}
        </p>
      </div>
    );
  };

  const renderDiagnosis = (formData) => {
    if (!formData?.diagnosis) {
      return <p className="text-muted">{t('common.noData')}</p>;
    }

    const { suggestions, selected } = formData.diagnosis;

    return (
      <div>
        {suggestions && suggestions.length > 0 && (
          <>
            <p>
              <strong>{t('patientView.diagnosis.suggestions')}:</strong>
            </p>
            <ul className="list-group mb-3">
              {suggestions.map((diagnosis, idx) => (
                <li key={idx} className="list-group-item">
                  {diagnosis}
                </li>
              ))}
            </ul>
          </>
        )}

        {selected && selected.length > 0 && (
          <>
            <p>
              <strong>{t('patientView.diagnosis.selected')}:</strong>
            </p>
            <div className="d-flex flex-wrap gap-2">
              {selected.map((diagnosis, idx) => (
                <Badge key={idx} bg="primary">
                  {diagnosis}
                </Badge>
              ))}
            </div>
          </>
        )}

        {!suggestions?.length && !selected?.length && (
          <p className="text-muted">{t('patientView.diagnosis.none')}</p>
        )}
      </div>
    );
  };

  const renderExams = (formData) => {
    if (!formData?.exams) {
      return <p className="text-muted">{t('common.noData')}</p>;
    }

    const { suggestions, selected } = formData.exams;

    return (
      <div>
        {suggestions && suggestions.length > 0 && (
          <>
            <p>
              <strong>{t('patientView.exams.suggestions')}:</strong>
            </p>
            <ul className="list-group mb-3">
              {suggestions.map((exam, idx) => (
                <li key={idx} className="list-group-item">
                  {exam}
                </li>
              ))}
            </ul>
          </>
        )}

        {selected && selected.length > 0 && (
          <>
            <p>
              <strong>{t('patientView.exams.selected')}:</strong>
            </p>
            <div className="d-flex flex-wrap gap-2">
              {selected.map((exam, idx) => (
                <Badge key={idx} bg="success">
                  {exam}
                </Badge>
              ))}
            </div>
          </>
        )}

        {!suggestions?.length && !selected?.length && (
          <p className="text-muted">{t('patientView.exams.none')}</p>
        )}
      </div>
    );
  };

  if (isLoading) {
    return (
      <Container className="py-5">
        <div className="text-center">
          <Spinner animation="border" role="status" className="mb-3">
          <span className="visually-hidden">{t('common.loading')}</span>
        </Spinner>
        <p className="text-muted">{t('patientView.loading')}</p>
      </div>
    </Container>
  );
  }

  if (error) {
    return (
      <Container className="py-5">
        <Alert variant="danger">
          <Alert.Heading>{t('patientView.errorTitle')}</Alert.Heading>
          <p>{error}</p>
        </Alert>
        <Button
          variant="outline-primary"
          onClick={() => navigate(-1)}
          className="me-2"
        >
          ← {t('common.goBack')}
        </Button>
        <Button variant="primary" onClick={() => navigate('/')}>
          {t('patientView.backToDashboard')}
        </Button>
      </Container>
    );
  }

  if (!patient) {
    return (
      <Container className="py-5">
        <Alert variant="warning">
          <Alert.Heading>{t('patientView.notFoundTitle')}</Alert.Heading>
          <p>{t('patientView.notFoundMessage')}</p>
        </Alert>
        <Button variant="primary" onClick={() => navigate('/')}>
          {t('patientView.backToDashboard')}
        </Button>
      </Container>
    );
  }

  return (
    <Container className="py-4">
      {/* Header */}
      <div className="mb-4 d-flex justify-content-between align-items-center">
        <div>
          <h1 className="display-6 fw-bold text-primary mb-2">
            {t('patientView.title')}
          </h1>
          <p className="text-muted">
            {t('patientView.subtitle')}{' '}
            <code>{patient.patient_id}</code>
          </p>
        </div>
        <Button
          variant="outline-secondary"
          onClick={() => navigate(-1)}
          size="lg"
        >
          ← {t('common.back')}
        </Button>
      </div>

      {/* Summary Card */}
      <Card className="border-0 shadow-sm mb-4">
        <Card.Body className="p-4">
          <Row className="g-4">
            <Col md={6} lg={3}>
              <p className="text-muted small mb-1">
                {t('patientView.summary.patientId')}
              </p>
              <code className="d-block fs-6">{patient.patient_id}</code>
            </Col>
            <Col md={6} lg={3}>
              <p className="text-muted small mb-1">
                {t('patientView.summary.language')}
              </p>
              <p className="mb-0 fs-6">
                <strong>{patient.lang?.toUpperCase() || t('common.na')}</strong>
              </p>
            </Col>
            <Col md={6} lg={3}>
              <p className="text-muted small mb-1">
                {t('patientView.summary.created')}
              </p>
              <p className="mb-0 fs-6">
                <strong>{formatDate(patient.created_at)}</strong>
              </p>
            </Col>
            <Col md={6} lg={3}>
              <p className="text-muted small mb-1">
                {t('patientView.summary.status')}
              </p>
              <Badge bg={patient.is_complete ? 'success' : 'warning'} className="fs-6">
                {patient.is_complete ? t('status.complete') : t('status.inProgress')}
              </Badge>
            </Col>
          </Row>
        </Card.Body>
      </Card>

      {/* Tabs with Consultation Data */}
      <Card className="border-0 shadow-sm">
        <Card.Body className="p-0">
          <Tab.Container defaultActiveKey="demographics">
            <Nav variant="pills" className="border-bottom p-3">
                <Nav.Item>
                  <Nav.Link eventKey="demographics" className="rounded-0">
                  📋 {t('patientView.tabs.demographics')}
                  </Nav.Link>
                </Nav.Item>
                <Nav.Item>
                  <Nav.Link eventKey="lifestyle" className="rounded-0">
                  🏃 {t('patientView.tabs.lifestyle')}
                  </Nav.Link>
                </Nav.Item>
                <Nav.Item>
                  <Nav.Link eventKey="symptoms" className="rounded-0">
                  🤒 {t('patientView.tabs.symptoms')}
                  </Nav.Link>
                </Nav.Item>
                <Nav.Item>
                  <Nav.Link eventKey="mental" className="rounded-0">
                  🧠 {t('patientView.tabs.mental')}
                  </Nav.Link>
                </Nav.Item>
                <Nav.Item>
                  <Nav.Link eventKey="medical" className="rounded-0">
                  📄 {t('patientView.tabs.medical')}
                  </Nav.Link>
                </Nav.Item>
                <Nav.Item>
                  <Nav.Link eventKey="wearable" className="rounded-0">
                  ⌚ {t('patientView.tabs.wearable')}
                  </Nav.Link>
                </Nav.Item>
                <Nav.Item>
                  <Nav.Link eventKey="diagnosis" className="rounded-0">
                  🔍 {t('patientView.tabs.diagnosis')}
                  </Nav.Link>
                </Nav.Item>
                <Nav.Item>
                  <Nav.Link eventKey="exams" className="rounded-0">
                  🩺 {t('patientView.tabs.exams')}
                  </Nav.Link>
                </Nav.Item>
              </Nav>

            <Tab.Content className="p-4">
              <Tab.Pane eventKey="demographics">
                {renderDemographics(patient.formData)}
              </Tab.Pane>
              <Tab.Pane eventKey="lifestyle">
                {renderLifestyle(patient.formData)}
              </Tab.Pane>
              <Tab.Pane eventKey="symptoms">
                {renderSymptoms(patient.formData)}
              </Tab.Pane>
              <Tab.Pane eventKey="mental">
                {renderMentalHealth(patient.formData)}
              </Tab.Pane>
              <Tab.Pane eventKey="medical">
                {renderMedicalReports(patient.formData)}
              </Tab.Pane>
              <Tab.Pane eventKey="wearable">
                {renderWearableData(patient.formData)}
              </Tab.Pane>
              <Tab.Pane eventKey="diagnosis">
                {renderDiagnosis(patient.formData)}
              </Tab.Pane>
              <Tab.Pane eventKey="exams">
                {renderExams(patient.formData)}
              </Tab.Pane>
            </Tab.Content>
          </Tab.Container>
        </Card.Body>
      </Card>

      {/* Raw JSON Data (for debugging) */}
        <Card className="border-0 shadow-sm mt-4">
        <Card.Header className="bg-light border-bottom">
          <h6 className="mb-0">{t('patientView.rawData')}</h6>
        </Card.Header>
        <Card.Body>
          <pre
            style={{
              whiteSpace: 'pre-wrap',
              wordWrap: 'break-word',
              fontSize: '0.85rem',
              backgroundColor: '#f8f9fa',
              padding: '15px',
              borderRadius: '6px',
              maxHeight: '400px',
              overflowY: 'auto',
            }}
          >
            {JSON.stringify(patient, null, 2)}
          </pre>
        </Card.Body>
      </Card>
    </Container>
  );
}
