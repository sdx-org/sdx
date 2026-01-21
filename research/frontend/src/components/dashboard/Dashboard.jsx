import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Button,
  Container,
  Card,
  Row,
  Col,
  Table,
  Badge,
  Spinner,
  Alert,
} from 'react-bootstrap';
import { useTranslation } from 'react-i18next';
import ReactPaginate from 'react-paginate';
import consultationAPI from '../../services/api';
import { useConsultation, consultationActions } from '../../context/ConsultationContext';

export default function Dashboard() {
  const navigate = useNavigate();
  const { t } = useTranslation();
  const { dispatch } = useConsultation();
  const [patients, setPatients] = useState([]);
  const [stats, setStats] = useState({
    total_patients: 0,
    active_records: 0,
    this_month: 0,
  });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [itemOffset, setItemOffset] = useState(0);
  const itemsPerPage = 5;
  const endOffset = itemOffset + itemsPerPage;
  const currentItems = patients.slice(itemOffset, endOffset);
  const pageCount = Math.ceil(patients.length / itemsPerPage);
  const loadDashboardData = async () => {
    try {
      setIsLoading(true);
      setError(null);

      // Fetch all patients
      const patientsData = await consultationAPI.getPatients();
      setPatients(Array.isArray(patientsData) ? patientsData : []);

      // Calculate stats
      const totalPatients = patientsData?.length || 0;
      const completedCount = patientsData?.filter(
        (p) => p.is_complete === true
      ).length || 0;

      // Count patients created this month (simple check)
      const thisMonth = patientsData?.filter((p) => {
        if (!p.created_at) return false;
        const createdDate = new Date(p.created_at);
        const now = new Date();
        return (
          createdDate.getMonth() === now.getMonth() &&
          createdDate.getFullYear() === now.getFullYear()
        );
      }).length || 0;

      setStats({
        total_patients: totalPatients,
        active_records: totalPatients - completedCount,
        this_month: thisMonth,
      });
    } catch (err) {
      console.error('Error loading dashboard data:', err);
      setError(err.message || 'Failed to load dashboard data');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadDashboardData();

    // Refetch when window regains focus
    const handleFocus = () => {
      if (document.visibilityState === 'visible') {
        loadDashboardData();
      }
    };

    window.addEventListener('focus', handleFocus);
    document.addEventListener('visibilitychange', handleFocus);

    return () => {
      window.removeEventListener('focus', handleFocus);
      document.removeEventListener('visibilitychange', handleFocus);
    };
  }, []);

  const handleAddPatient = () => {
    // Navigate to language selection to create new patient
    Object.keys(localStorage).forEach((key) => {
      if (key.startsWith('consultationState_')) {
        localStorage.removeItem(key);
      }
    });
    dispatch(consultationActions.resetState());
    navigate('/language');
  };

  const handleViewPatient = (patientId) => {
    navigate(`/patients/${patientId}`);
  };

  const handleResumePatient = async (patientId) => {
    try{
      const consultationData=await consultationAPI.getConsultationStatus(patientId);
      if(!consultationData){
        alert('No consultation data found for this patient.');
        return;
      }
      const currentStep=consultationData.current_step || 'demographics';
      const language=consultationData.lang || 'en';
      dispatch(consultationActions.initConsultation(
        patientId,
        language,
        currentStep
      ));

      const shouldPrefillFromLocalStorage=[
        'demographics',
        'lifestyle',
        'symptoms',
        'mental',
      ].includes(currentStep);

      if(shouldPrefillFromLocalStorage){
        const savedState=localStorage.getItem(`consultationState_${patientId}`);
        if(savedState){
          const parsedState=JSON.parse(savedState);
          if(parsedState.formData.demographics){
            dispatch(consultationActions.updateDemographics(parsedState.formData.demographics));
          }
          if(parsedState.formData.lifestyle){
            dispatch(consultationActions.updateLifestyle(parsedState.formData.lifestyle));
          }
          if(parsedState.formData.symptoms){
            dispatch(consultationActions.updateSymptoms(parsedState.formData.symptoms));
          }
          if(parsedState.formData.mental){
            dispatch(consultationActions.updateMentalHealth(parsedState.formData.mental));
          }
        }
      }
        navigate(`/${currentStep}`);
    }catch(err){
      console.error('Error resuming patient consultation:',err);
      alert(`Failed to resume consultation: ${err.message}`);
    }
  };

  const handleDeletePatient = async (patientId, patientName) => {
    if (
      !window.confirm(
        `Delete patient "${patientName || patientId}"? This action cannot be undone.`
      )
    ) {
      return;
    }

    try {
      await consultationAPI.deletePatient(patientId);

      // Remove from local list
      setPatients((prev) => prev.filter((p) => p.patient_id !== patientId));

      // Reload stats
      await loadDashboardData();
    } catch (err) {
      console.error('Error deleting patient:', err);
      alert(`Failed to delete patient: ${err.message}`);
    }
  };

  const handlePageClick = (event) => {
    const newOffset = (event.selected * itemsPerPage) % patients.length;
    setItemOffset(newOffset);
  };

  const hasPatients = patients && patients.length > 0;

  return (
    <div className="bg-light min-vh-100 py-4">
      <Container>
        {/* Header */}
        <div className="mb-4 d-flex justify-content-between align-items-center">
          <div>
            <h1 className="display-6 fw-bold text-primary mb-2">
              {t('dashboard.title')}
            </h1>
            <p className="text-muted lead mb-0">
              Manage and track patient records efficiently
            </p>
          </div>

          <Button
            onClick={handleAddPatient}
            variant="primary"
            size="lg"
            className="d-flex align-items-center gap-2"
          >
            <span style={{ fontSize: '1.2rem' }}>➕</span>
            <span>Add Patient</span>
          </Button>
        </div>

        {/* Error Alert */}
        {error && (
          <Alert variant="danger" dismissible onClose={() => setError(null)}>
            <Alert.Heading>Error</Alert.Heading>
            <p>{error}</p>
          </Alert>
        )}

        {/* Stats Cards */}
        <Row className="g-4 mb-4">
          {/* Total Patients */}
          <Col md={6} lg={4}>
            <Card className="border-0 shadow-sm h-100">
              <Card.Body className="p-4">
                <div className="d-flex align-items-center">
                  <div className="bg-primary bg-opacity-10 p-3 rounded-3 me-3">
                    <span style={{ fontSize: '1.5rem' }}>📋</span>
                  </div>
                  <div>
                    <h3 className="mb-0 fw-bold">
                      {isLoading ? (
                        <Spinner animation="border" size="sm" />
                      ) : (
                        stats.total_patients
                      )}
                    </h3>
                    <p className="text-muted mb-0 small">Total Patients</p>
                  </div>
                </div>
              </Card.Body>
            </Card>
          </Col>

          {/* Active Records */}
          <Col md={6} lg={4}>
            <Card className="border-0 shadow-sm h-100">
              <Card.Body className="p-4">
                <div className="d-flex align-items-center">
                  <div className="bg-success bg-opacity-10 p-3 rounded-3 me-3">
                    <span style={{ fontSize: '1.5rem' }}>✅</span>
                  </div>
                  <div>
                    <h3 className="mb-0 fw-bold">
                      {isLoading ? (
                        <Spinner animation="border" size="sm" />
                      ) : (
                        stats.active_records
                      )}
                    </h3>
                    <p className="text-muted mb-0 small">Active Records</p>
                  </div>
                </div>
              </Card.Body>
            </Card>
          </Col>

          {/* This Month */}
          <Col md={6} lg={4}>
            <Card className="border-0 shadow-sm h-100">
              <Card.Body className="p-4">
                <div className="d-flex align-items-center">
                  <div className="bg-info bg-opacity-10 p-3 rounded-3 me-3">
                    <span style={{ fontSize: '1.5rem' }}>📊</span>
                  </div>
                  <div>
                    <h3 className="mb-0 fw-bold">
                      {isLoading ? (
                        <Spinner animation="border" size="sm" />
                      ) : (
                        stats.this_month
                      )}
                    </h3>
                    <p className="text-muted mb-0 small">This Month</p>
                  </div>
                </div>
              </Card.Body>
            </Card>
          </Col>
        </Row>

        {/* Patients Table */}
        {isLoading && !hasPatients ? (
          <Card className="border-0 shadow-sm">
            <Card.Body className="p-5 text-center">
              <Spinner animation="border" role="status">
                <span className="visually-hidden">Loading...</span>
              </Spinner>
              <p className="text-muted mt-3">Loading patients...</p>
            </Card.Body>
          </Card>
        ) : !hasPatients ? (
          <Card className="border-0 shadow-sm">
            <Card.Body className="p-5 text-center">
              <h5 className="mb-3 fw-semibold">
                {t('dashboard.noRecords')}
              </h5>
              <p className="text-muted mb-4">
                Get started by adding your first patient record
              </p>
              <Button
                onClick={handleAddPatient}
                variant="primary"
                size="lg"
                className="px-4"
              >
                <span className="me-2">➕</span>
                Add New Patient
              </Button>
            </Card.Body>
          </Card>
        ) : (
          <Card className="border-0 shadow-sm">
            <Card.Body className="p-4">
              <h5 className="mb-4 fw-semibold">
                Recent Patients ({patients.length})
              </h5>

              <div className="table-responsive">
                <Table hover className="align-middle">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Patient ID</th>
                      <th>Language</th>
                      <th>Status</th>
                      <th>Created</th>
                      <th>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {currentItems.map((patient, index) => (
                      <tr key={patient.patient_id}>
                        <td>{itemOffset + index + 1}</td>
                        <td>
                          <code className="text-truncate" style={{ maxWidth: '150px' }}>
                            {patient.patient_id}
                          </code>
                        </td>
                        <td>{patient.lang || 'N/A'}</td>
                        <td>
                          <Badge
                            bg={
                              patient.is_complete === true
                                ? 'success'
                                : 'warning'
                            }
                          >
                            {patient.is_complete === true
                              ? '✓ Complete'
                              : '⏳ In Progress'}
                          </Badge>
                        </td>
                        <td className="small">
                          {patient.created_at
                            ? new Date(patient.created_at).toLocaleDateString()
                            : 'N/A'}
                        </td>
                        <td>
                          <div className="d-flex gap-2">
                            {patient.is_complete === true ? (
                              <Button
                                variant="outline-primary"
                                size="sm"
                                onClick={() =>
                                  handleViewPatient(patient.patient_id)
                                }
                              >
                                View
                              </Button>
                            ) : (
                              <Button
                                variant="outline-primary"
                                size="sm"
                                onClick={() =>
                                  handleResumePatient(patient.patient_id)
                                }
                              >
                                Resume
                              </Button>
                            )}
                            <Button
                              variant="outline-danger"
                              size="sm"
                              onClick={() =>
                                handleDeletePatient(
                                  patient.patient_id,
                                  patient.patient_id
                                )
                              }
                            >
                              Delete
                            </Button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              </div>

              {/* Pagination */}
              {pageCount > 1 && (
                <ReactPaginate
                  breakLabel="..."
                  nextLabel="›"
                  previousLabel="‹"
                  onPageChange={handlePageClick}
                  pageRangeDisplayed={3}
                  pageCount={pageCount}
                  renderOnZeroPageCount={null}
                  containerClassName="pagination justify-content-center mt-4"
                  pageClassName="page-item"
                  pageLinkClassName="page-link"
                  previousClassName="page-item"
                  previousLinkClassName="page-link"
                  nextClassName="page-item"
                  nextLinkClassName="page-link"
                  activeClassName="active"
                />
              )}
            </Card.Body>
          </Card>
        )}
      </Container>
    </div>
  );
}
