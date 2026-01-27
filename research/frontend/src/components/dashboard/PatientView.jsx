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

// Tab Components
import DemographicsTab from './tabs/DemographicsTab';
import LifestyleTab from './tabs/LifestyleTab';
import SymptomsTab from './tabs/SymptomsTab';
import MentalHealthTab from './tabs/MentalHealthTab';
import MedicalReportsTab from './tabs/MedicalReportsTab';
import WearableDataTab from './tabs/WearableDataTab';
import DiagnosisTab from './tabs/DiagnosisTab';
import ExamsTab from './tabs/ExamsTab';

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

  const formData = patient?.formData ?? {};

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
                <DemographicsTab data={formData?.demographics} />
              </Tab.Pane>
              <Tab.Pane eventKey="lifestyle">
                <LifestyleTab data={formData?.lifestyle} />
              </Tab.Pane>
              <Tab.Pane eventKey="symptoms">
                <SymptomsTab data={formData?.symptoms} />
              </Tab.Pane>
              <Tab.Pane eventKey="mental">
                <MentalHealthTab data={formData?.mental} />
              </Tab.Pane>
              <Tab.Pane eventKey="medical">
                <MedicalReportsTab data={formData?.medicalReports} />
              </Tab.Pane>
              <Tab.Pane eventKey="wearable">
                <WearableDataTab data={formData?.wearableData} />
              </Tab.Pane>
              <Tab.Pane eventKey="diagnosis">
                <DiagnosisTab data={formData?.diagnosis} />
              </Tab.Pane>
              <Tab.Pane eventKey="exams">
                <ExamsTab data={formData?.exams} />
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
