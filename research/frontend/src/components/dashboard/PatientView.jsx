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
import consultationAPI from '../../services/api';

export default function PatientView() {
  const { id } = useParams();
  const navigate = useNavigate();
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
          throw new Error('Patient record not found');
        }

        setPatient(data);
      } catch (err) {
        console.error('Error loading patient:', err);
        setError(err.message || 'Failed to load patient record');
      } finally {
        setIsLoading(false);
      }
    };

    loadPatient();
  }, [id]);

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    try {
      return new Date(dateString).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
      });
    } catch {
      return dateString;
    }
  };

  const renderDemographics = (formData) => {
    if (!formData?.demographics) return <p className="text-muted">No data</p>;

    const { age, gender, weight, height } = formData.demographics;
    return (
      <div>
        <Row className="g-3">
          <Col md={6}>
            <p>
              <strong>Age:</strong> {age ? `${age} years` : 'Not provided'}
            </p>
          </Col>
          <Col md={6}>
            <p>
              <strong>Gender:</strong> {gender || 'Not provided'}
            </p>
          </Col>
          <Col md={6}>
            <p>
              <strong>Weight:</strong> {weight ? `${weight} kg` : 'Not provided'}
            </p>
          </Col>
          <Col md={6}>
            <p>
              <strong>Height:</strong> {height ? `${height} cm` : 'Not provided'}
            </p>
          </Col>
        </Row>
      </div>
    );
  };

  const renderLifestyle = (formData) => {
    if (!formData?.lifestyle) return <p className="text-muted">No data</p>;

    const { diet, sleep_hours, physical_activity, mental_exercises } =
      formData.lifestyle;
    return (
      <div>
        <Row className="g-3">
          <Col md={6}>
            <p>
              <strong>Diet:</strong> {diet || 'Not provided'}
            </p>
          </Col>
          <Col md={6}>
            <p>
              <strong>Sleep Hours:</strong>{' '}
              {sleep_hours ? `${sleep_hours} hours/day` : 'Not provided'}
            </p>
          </Col>
          <Col md={6}>
            <p>
              <strong>Physical Activity:</strong>{' '}
              {physical_activity || 'Not provided'}
            </p>
          </Col>
          <Col md={6}>
            <p>
              <strong>Mental Exercises:</strong>{' '}
              {mental_exercises || 'Not provided'}
            </p>
          </Col>
        </Row>
      </div>
    );
  };

  const renderSymptoms = (formData) => {
    if (!formData?.symptoms) return <p className="text-muted">No data</p>;

    return (
      <div>
        <p>
          <strong>Symptoms:</strong>
        </p>
        <p className="bg-light p-3 rounded">{formData.symptoms.symptoms || 'Not provided'}</p>
      </div>
    );
  };

  const renderMentalHealth = (formData) => {
    if (!formData?.mental) return <p className="text-muted">No data</p>;

    return (
      <div>
        <p>
          <strong>Mental Health Notes:</strong>
        </p>
        <p className="bg-light p-3 rounded">{formData.mental.mental_health || 'Not provided'}</p>
      </div>
    );
  };

  const renderMedicalReports = (formData) => {
    if (!formData?.medicalReports) return <p className="text-muted">No data</p>;

    const { files, skipped } = formData.medicalReports;

    if (skipped) {
      return (
        <Alert variant="info">
          <strong>Skipped:</strong> User skipped uploading medical reports
        </Alert>
      );
    }

    if (!files || files.length === 0) {
      return <p className="text-muted">No files uploaded</p>;
    }

    return (
      <div>
        <p>
          <strong>Medical Reports:</strong>
        </p>
        <ul className="list-group">
          {files.map((file, idx) => (
            <li key={idx} className="list-group-item">
              📄 {file.name || `File ${idx + 1}`}
            </li>
          ))}
        </ul>
      </div>
    );
  };

  const renderWearableData = (formData) => {
    if (!formData?.wearableData) return <p className="text-muted">No data</p>;

    const { file, skipped } = formData.wearableData;

    if (skipped) {
      return (
        <Alert variant="info">
          <strong>Skipped:</strong> User skipped uploading wearable data
        </Alert>
      );
    }

    if (!file) {
      return <p className="text-muted">No file uploaded</p>;
    }

    return (
      <div>
        <p>
          <strong>Wearable Data File:</strong>
        </p>
        <p className="bg-light p-3 rounded">📊 {file.name || 'Wearable data'}</p>
      </div>
    );
  };

  const renderDiagnosis = (formData) => {
    if (!formData?.diagnosis) return <p className="text-muted">No data</p>;

    const { suggestions, selected } = formData.diagnosis;

    return (
      <div>
        {suggestions && suggestions.length > 0 && (
          <>
            <p>
              <strong>Diagnosis Suggestions:</strong>
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
              <strong>Selected Diagnoses:</strong>
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
          <p className="text-muted">No diagnosis data</p>
        )}
      </div>
    );
  };

  const renderExams = (formData) => {
    if (!formData?.exams) return <p className="text-muted">No data</p>;

    const { suggestions, selected } = formData.exams;

    return (
      <div>
        {suggestions && suggestions.length > 0 && (
          <>
            <p>
              <strong>Exam Suggestions:</strong>
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
              <strong>Selected Exams:</strong>
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
          <p className="text-muted">No exam data</p>
        )}
      </div>
    );
  };

  if (isLoading) {
    return (
      <Container className="py-5">
        <div className="text-center">
          <Spinner animation="border" role="status" className="mb-3">
            <span className="visually-hidden">Loading...</span>
          </Spinner>
          <p className="text-muted">Loading patient record...</p>
        </div>
      </Container>
    );
  }

  if (error) {
    return (
      <Container className="py-5">
        <Alert variant="danger">
          <Alert.Heading>Error Loading Patient</Alert.Heading>
          <p>{error}</p>
        </Alert>
        <Button
          variant="outline-primary"
          onClick={() => navigate(-1)}
          className="me-2"
        >
          ← Go Back
        </Button>
        <Button variant="primary" onClick={() => navigate('/')}>
          Back to Dashboard
        </Button>
      </Container>
    );
  }

  if (!patient) {
    return (
      <Container className="py-5">
        <Alert variant="warning">
          <Alert.Heading>Patient Not Found</Alert.Heading>
          <p>The requested patient record could not be found.</p>
        </Alert>
        <Button variant="primary" onClick={() => navigate('/')}>
          Back to Dashboard
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
            Patient Record
          </h1>
          <p className="text-muted">
            View complete consultation details for patient{' '}
            <code>{patient.patient_id}</code>
          </p>
        </div>
        <Button
          variant="outline-secondary"
          onClick={() => navigate(-1)}
          size="lg"
        >
          ← Back
        </Button>
      </div>

      {/* Summary Card */}
      <Card className="border-0 shadow-sm mb-4">
        <Card.Body className="p-4">
          <Row className="g-4">
            <Col md={6} lg={3}>
              <p className="text-muted small mb-1">Patient ID</p>
              <code className="d-block fs-6">{patient.patient_id}</code>
            </Col>
            <Col md={6} lg={3}>
              <p className="text-muted small mb-1">Language</p>
              <p className="mb-0 fs-6">
                <strong>{patient.lang?.toUpperCase() || 'N/A'}</strong>
              </p>
            </Col>
            <Col md={6} lg={3}>
              <p className="text-muted small mb-1">Created</p>
              <p className="mb-0 fs-6">
                <strong>{formatDate(patient.created_at)}</strong>
              </p>
            </Col>
            <Col md={6} lg={3}>
              <p className="text-muted small mb-1">Status</p>
              <Badge bg={patient.is_complete ? 'success' : 'warning'} className="fs-6">
                {patient.is_complete ? '✓ Complete' : '⏳ In Progress'}
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
                  📋 Demographics
                </Nav.Link>
              </Nav.Item>
              <Nav.Item>
                <Nav.Link eventKey="lifestyle" className="rounded-0">
                  🏃 Lifestyle
                </Nav.Link>
              </Nav.Item>
              <Nav.Item>
                <Nav.Link eventKey="symptoms" className="rounded-0">
                  🤒 Symptoms
                </Nav.Link>
              </Nav.Item>
              <Nav.Item>
                <Nav.Link eventKey="mental" className="rounded-0">
                  🧠 Mental Health
                </Nav.Link>
              </Nav.Item>
              <Nav.Item>
                <Nav.Link eventKey="medical" className="rounded-0">
                  📄 Medical Reports
                </Nav.Link>
              </Nav.Item>
              <Nav.Item>
                <Nav.Link eventKey="wearable" className="rounded-0">
                  ⌚ Wearable Data
                </Nav.Link>
              </Nav.Item>
              <Nav.Item>
                <Nav.Link eventKey="diagnosis" className="rounded-0">
                  🔍 Diagnosis
                </Nav.Link>
              </Nav.Item>
              <Nav.Item>
                <Nav.Link eventKey="exams" className="rounded-0">
                  🩺 Exams
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
          <h6 className="mb-0">Raw Data (JSON)</h6>
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
