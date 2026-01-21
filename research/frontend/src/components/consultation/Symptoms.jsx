import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import {
  Form,
  Button,
  Container,
  Row,
  Col,
  Card,
  ProgressBar,
  Alert,
  Spinner,
  ListGroup,
} from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useConsultation, consultationActions } from '../../context/ConsultationContext';
import consultationAPI from '../../services/api';

export default function Symptoms() {
  const navigate = useNavigate();
  const { t } = useTranslation();
  const { state, dispatch } = useConsultation();
  const {
    register,
    handleSubmit,
    watch,
    formState: { errors, isSubmitting },
  } = useForm({
    defaultValues: {
      symptoms: state.formData.symptoms.symptoms || '',
    },
  });

  const [apiError, setApiError] = useState(null);
  const watchSymptoms = watch('symptoms');
  const getCharacterCount = () => watchSymptoms?.length || 0;

  const getSeverityHint = (length) => {
    if (length < 10) return { status: 'Too short', color: 'danger' };
    if (length < 50) return { status: 'Brief', color: 'warning' };
    if (length < 150) return { status: 'Good', color: 'info' };
    return { status: 'Detailed', color: 'success' };
  };

  const commonSymptoms = [
    '🤕 Headache / Migraine',
    '🤒 Fever / Chills',
    '🫁 Cough / Breathing issues',
    '🤢 Nausea / Vomiting',
    '🫘 Chest pain / Discomfort',
    '😴 Fatigue / Weakness',
    '🦴 Joint / Muscle pain',
    '🧠 Dizziness / Vertigo',
    '👁️ Vision problems',
    '💪 Weakness',
  ];

  const isFormComplete = watchSymptoms && watchSymptoms.length >= 10;
  const charCount = getCharacterCount();
  const severityHint = getSeverityHint(charCount);
  const onSubmit = async (data) => {
    try {
      setApiError(null);

      // Validate patient ID exists
      if (!state.patientId) {
        throw new Error('Patient ID not found. Please start over.');
      }

      // Update local state
      dispatch(
        consultationActions.updateSymptoms({
          symptoms: data.symptoms,
        })
      );

      // Call backend API
      await consultationAPI.updateConsultationSymptoms(state.patientId, {
        symptoms: data.symptoms,
      });

      // Update current step in context
      dispatch(consultationActions.setCurrentStep('mental'));

      // Navigate to next step
      navigate('/mental');
    } catch (err) {
      console.error('Error saving symptoms data:', err);
      setApiError(err.message || 'Failed to save symptoms. Please try again.');
      window.scrollTo(0, 0);
    }
  };

  return (
    <div className="bg-light min-vh-100 py-4">
      <Container>
        {/* Progress Section */}
        <div className="mb-4">
          <div className="d-flex justify-content-between align-items-center mb-2">
            <small className="text-muted fw-semibold">Step 3 of 10</small>
            <small className="text-muted">30% Complete</small>
          </div>
          <ProgressBar now={30} style={{ height: '8px' }} className="mb-3" />
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
                🤒 Symptoms & Concerns
              </h1>
              <p className="text-muted lead mb-0">
                Describe any current symptoms or health concerns you're experiencing
              </p>
            </div>

            {/* Form */}
            <Form onSubmit={handleSubmit(onSubmit)}>
              {/* Common Symptoms Reference */}
              <div className="mb-5">
                <p className="text-muted fw-semibold mb-3">
                  Common symptoms (for reference):
                </p>
                <div className="d-flex flex-wrap gap-2 mb-4">
                  {commonSymptoms.map((symptom, idx) => (
                    <span
                      key={idx}
                      className="badge bg-light text-dark border border-secondary"
                      style={{ fontSize: '0.85rem', padding: '0.5rem 0.75rem' }}
                    >
                      {symptom}
                    </span>
                  ))}
                </div>
              </div>

              {/* Symptoms Textarea */}
              <Form.Group className="mb-4">
                <Form.Label className="fw-semibold mb-2">
                  Describe Your Symptoms <span className="text-danger">*</span>
                </Form.Label>
                <Form.Control
                  as="textarea"
                  rows={6}
                  placeholder="Be specific about:
• What symptoms you're experiencing
• When they started
• How often they occur
• Severity (mild/moderate/severe)
• What triggers or relieves them
• Any related activities or conditions"
                  className={`${errors.symptoms ? 'is-invalid' : ''}`}
                  style={{ resize: 'vertical' }}
                  {...register('symptoms', {
                    required: 'Please describe your symptoms',
                    minLength: {
                      value: 10,
                      message: 'Please provide more details (minimum 10 characters)',
                    },
                  })}
                />
                {errors.symptoms && (
                  <Form.Control.Feedback type="invalid" className="d-block mt-2">
                    {errors.symptoms.message}
                  </Form.Control.Feedback>
                )}

                {/* Character Count & Status */}
                <div className="mt-3 d-flex justify-content-between align-items-center">
                  <small className="text-muted">
                    {charCount} characters entered
                  </small>
                  {watchSymptoms && (
                    <span
                      className={`badge bg-${severityHint.color}`}
                      style={{
                        fontSize: '0.85rem',
                        padding: '0.4rem 0.8rem',
                      }}
                    >
                      {severityHint.status}
                    </span>
                  )}
                </div>
              </Form.Group>

              {/* Example Box */}
              <Card className="bg-light border-0 mb-4">
                <Card.Body className="p-3">
                  <p className="text-muted small mb-2">
                    <strong>📝 Example of detailed symptom description:</strong>
                  </p>
                  <p className="text-muted small mb-0" style={{ fontSize: '0.9rem' }}>
                    "Recurring headaches for 2 weeks, mostly in afternoon, moderate
                    severity. Triggered by stress and long screen time. Pain is located
                    in temples and forehead. Relieved by rest and paracetamol. No
                    associated fever or nausea."
                  </p>
                </Card.Body>
              </Card>

              {/* Helpful Tips */}
              <Alert variant="info" className="border-0 bg-info bg-opacity-10 mb-4">
                <div className="d-flex align-items-start">
                  <span className="me-2" style={{ fontSize: '1.2rem' }}>
                    💡
                  </span>
                  <div className="small">
                    <p className="mb-2 fw-semibold">Tips for describing symptoms:</p>
                    <ul className="mb-0 ps-3">
                      <li>Be as specific as possible about location and timing</li>
                      <li>Include severity and impact on daily activities</li>
                      <li>Mention any known triggers or relief measures</li>
                      <li>Note any other related symptoms</li>
                    </ul>
                  </div>
                </div>
              </Alert>

              {/* Emergency Notice */}
              <Alert variant="warning" className="border-0 bg-warning bg-opacity-10 mb-4">
                <div className="d-flex align-items-start">
                  <span className="me-2" style={{ fontSize: '1.2rem' }}>
                    ⚠️
                  </span>
                  <small>
                    <strong>Emergency:</strong> If you experience severe chest pain,
                    difficulty breathing, loss of consciousness, or other life-threatening
                    symptoms, please contact emergency services or visit the nearest hospital
                    immediately.
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

                <Button
                  type="submit"
                  variant="primary"
                  size="lg"
                  disabled={isSubmitting || !isFormComplete}
                  className="d-flex align-items-center gap-2"
                >
                  {isSubmitting ? (
                    <>
                      <Spinner animation="border" size="sm" />
                      <span>Saving...</span>
                    </>
                  ) : (
                    <>
                      <span>Next: Mental Health</span>
                      <span>→</span>
                    </>
                  )}
                </Button>
              </div>
            </Form>
          </Card.Body>
        </Card>

        {/* Footer */}
        <div className="text-center mt-4">
          <small className="text-muted">
            Your symptom information is completely confidential and encrypted
          </small>
        </div>
      </Container>
    </div>
  );
}
