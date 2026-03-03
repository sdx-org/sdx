import { useEffect, useMemo, useState } from 'react';
import {
  Navbar,
  Nav,
  Container,
  NavDropdown,
  Spinner,
  Badge,
  Button,
} from 'react-bootstrap';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import consultationAPI from '../services/api';
import { useConsultation, consultationActions } from '../context/ConsultationContext';
import { resumeConsultationForPatient } from '../utils/consultationNavigation';

export default function AppNavbar() {
  const location = useLocation();
  const navigate = useNavigate();
  const { i18n, t } = useTranslation();
  const { state, dispatch } = useConsultation();

  const [patients, setPatients] = useState([]);
  const [isLoadingPatients, setIsLoadingPatients] = useState(false);
  const [resumeError, setResumeError] = useState('');

  const isConsultationRoute = useMemo(() => {
    const consultationPaths = [
      '/language',
      '/demographics',
      '/lifestyle',
      '/symptoms',
      '/mental',
      '/medical-reports',
      '/wearable-data',
      '/diagnosis',
      '/exams',
      '/confirmation',
    ];
    return consultationPaths.includes(location.pathname);
  }, [location.pathname]);

  const incompletePatients = useMemo(
    () => patients.filter((patient) => patient.is_complete !== true),
    [patients]
  );

  const loadPatients = async () => {
    try {
      setIsLoadingPatients(true);
      const patientsData = await consultationAPI.getPatients();
      setPatients(Array.isArray(patientsData) ? patientsData : []);
    } catch (error) {
      console.error('Error loading patients for navbar:', error);
      setPatients([]);
    } finally {
      setIsLoadingPatients(false);
    }
  };

  useEffect(() => {
    loadPatients();
  }, []);

  const handleStartNewPatient = () => {
    Object.keys(localStorage).forEach((key) => {
      if (key.startsWith('consultationState_')) {
        localStorage.removeItem(key);
      }
    });
    dispatch(consultationActions.resetState());
    navigate('/language');
  };

  const handleResumeDiagnosis = async (patientId) => {
    try {
      setResumeError('');
      await resumeConsultationForPatient({ patientId, dispatch, navigate });
    } catch (error) {
      console.error('Error resuming diagnosis from navbar:', error);
      setResumeError(error.message || 'Failed to resume diagnosis.');
    }
  };

  return (
    <Navbar bg="primary" variant="dark" expand="lg" className="shadow-sm">
      <Container fluid="lg">
        <Navbar.Brand
          as={Link}
          to="/"
          className="fw-bold text-white"
        >
          TeleHealthCareAI
        </Navbar.Brand>

        <Navbar.Toggle aria-controls="app-navbar-nav" />
        <Navbar.Collapse id="app-navbar-nav">
          <Nav className="me-auto align-items-lg-center gap-lg-2">
            <Nav.Link as={Link} to="/" className="text-white">
              {t('navbar.patientDashboard')}
            </Nav.Link>

            <NavDropdown
              title={t('navbar.switchPatient')}
              id="switch-patient-dropdown"
              menuVariant="light"
            >
              {isLoadingPatients ? (
                <NavDropdown.ItemText className="small text-muted">
                  <Spinner animation="border" size="sm" className="me-2" />
                  {t('navbar.loadingPatients')}
                </NavDropdown.ItemText>
              ) : patients.length === 0 ? (
                <NavDropdown.ItemText className="small text-muted">
                  {t('navbar.noPatientRecords')}
                </NavDropdown.ItemText>
              ) : (
                patients.map((patient) => (
                  <NavDropdown.Item
                    key={patient.patient_id}
                    onClick={() => navigate(`/patients/${patient.patient_id}`)}
                  >
                    <div className="d-flex justify-content-between align-items-center gap-3">
                      <code>{patient.patient_id}</code>
                      <Badge bg={patient.is_complete ? 'success' : 'warning'}>
                        {patient.is_complete ? t('navbar.complete') : t('navbar.inProgress')}
                      </Badge>
                    </div>
                  </NavDropdown.Item>
                ))
              )}
            </NavDropdown>

            <NavDropdown
              title={t('navbar.resumeDiagnosis')}
              id="resume-diagnosis-dropdown"
              menuVariant="light"
            >
              {isLoadingPatients ? (
                <NavDropdown.ItemText className="small text-muted">
                  <Spinner animation="border" size="sm" className="me-2" />
                  {t('navbar.loadingResumablePatients')}
                </NavDropdown.ItemText>
              ) : incompletePatients.length === 0 ? (
                <NavDropdown.ItemText className="small text-muted">
                  {t('navbar.noConsultationsInProgress')}
                </NavDropdown.ItemText>
              ) : (
                incompletePatients.map((patient) => (
                  <NavDropdown.Item
                    key={patient.patient_id}
                    onClick={() => handleResumeDiagnosis(patient.patient_id)}
                  >
                    <div className="d-flex justify-content-between align-items-center gap-3">
                      <code>{patient.patient_id}</code>
                      <small className="text-muted">
                        {patient.current_step || 'demographics'}
                      </small>
                    </div>
                  </NavDropdown.Item>
                ))
              )}
            </NavDropdown>

            {isConsultationRoute && state.patientId ? (
              <Nav.Item className="d-flex align-items-center">
                <Badge bg="light" text="dark">
                  Active: {state.patientId}
                </Badge>
              </Nav.Item>
            ) : null}
          </Nav>

          <div className="d-flex align-items-center gap-2 mt-3 mt-lg-0">
            <Button
              variant="light"
              size="sm"
              onClick={handleStartNewPatient}
            >
              + {t('navbar.newPatient')}
            </Button>
            <Button
              variant="outline-light"
              size="sm"
              onClick={loadPatients}
              disabled={isLoadingPatients}
            >
              {t('navbar.refresh')}
            </Button>
            <NavDropdown
              title={`🌐 ${i18n.language.toUpperCase()}`}
              id="language-dropdown"
              align="end"
              menuVariant="light"
            >
              <NavDropdown.Item onClick={() => i18n.changeLanguage('en')}>
                {t('navbar.english')}
              </NavDropdown.Item>
              <NavDropdown.Item onClick={() => i18n.changeLanguage('es')}>
                {t('navbar.spanish')}
              </NavDropdown.Item>
            </NavDropdown>
          </div>
        </Navbar.Collapse>
      </Container>

      {resumeError ? (
        <div className="position-absolute top-100 start-50 translate-middle-x mt-2">
          <Badge bg="danger">{resumeError}</Badge>
        </div>
      ) : null}
    </Navbar>
  );
}
