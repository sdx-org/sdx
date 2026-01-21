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
} from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useConsultation, consultationActions } from '../../context/ConsultationContext';
import consultationAPI from '../../services/api';

export default function Mental() {
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
      mental_health: state.formData.mental.mental_health || '',
    },
  });

  const [apiError, setApiError] = useState(null);
  const watchMentalHealth = watch('mental_health');
  const getCharacterCount = () => watchMentalHealth?.length || 0;

  const getDetailLevel = (length) => {
    if (length < 10) return { status: 'Too short', color: 'danger' };
    if (length < 50) return { status: 'Brief', color: 'warning' };
    if (length < 150) return { status: 'Good', color: 'info' };
    return { status: 'Comprehensive', color: 'success' };
  };

  const stressAreas = [
    '💼 Work / Career',
    '👨‍👩‍👧‍👦 Family / Relationships',
    '💰 Financial',
    '🏥 Health concerns',
    '🏠 Home / Living situation',
    '👥 Social / Isolation',
    '🎓 Education / Learning',
    '🎯 Life goals / Purpose',
  ];
  const copingStrategies = [
    '🧘 Meditation / Mindfulness',
    '💪 Exercise / Physical activity',
    '👥 Social support / Friends',
    '🗣️ Therapy / Counseling',
    '📖 Reading / Creative pursuits',
    '🎵 Music / Entertainment',
    '😴 Sleep / Rest',
    '🧠 Problem-solving',
    '🙏 Spiritual / Religious practices',
  ];
  const isFormComplete = watchMentalHealth && watchMentalHealth.length >= 10;
  const charCount = getCharacterCount();
  const detailLevel = getDetailLevel(charCount);
  const onSubmit = async (data) => {
    try {
      setApiError(null);

      // Validate patient ID exists
      if (!state.patientId) {
        throw new Error('Patient ID not found. Please start over.');
      }

      // Update local state
      dispatch(
        consultationActions.updateMentalHealth({
          mental_health: data.mental_health,
        })
      );

      // Call backend API
      await consultationAPI.updateConsultationMentalHealth(state.patientId, {
        mental_health: data.mental_health,
      });

      // Update current step in context
      dispatch(consultationActions.setCurrentStep('medical-reports'));

      // Navigate to next step
      navigate('/medical-reports');
    } catch (err) {
      console.error('Error saving mental health data:', err);
      setApiError(err.message || 'Failed to save mental health information. Please try again.');
      window.scrollTo(0, 0);
    }
  };

  return (
    <div className="bg-light min-vh-100 py-4">
      <Container>
        {/* Progress Section */}
        <div className="mb-4">
          <div className="d-flex justify-content-between align-items-center mb-2">
            <small className="text-muted fw-semibold">Step 4 of 10</small>
            <small className="text-muted">40% Complete</small>
          </div>
          <ProgressBar now={40} style={{ height: '8px' }} className="mb-3" />
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
                🧠 Mental Health & Wellness
              </h1>
              <p className="text-muted lead mb-0">
                Share information about your mental health, stress levels, and emotional well-being
              </p>
            </div>

            {/* Form */}
            <Form onSubmit={handleSubmit(onSubmit)}>
              {/* Common Stress Areas */}
              <div className="mb-5">
                <p className="text-muted fw-semibold mb-3">
                  Common stress areas (for reference):
                </p>
                <div className="d-flex flex-wrap gap-2 mb-4">
                  {stressAreas.map((area, idx) => (
                    <span
                      key={idx}
                      className="badge bg-light text-dark border border-secondary"
                      style={{ fontSize: '0.85rem', padding: '0.5rem 0.75rem' }}
                    >
                      {area}
                    </span>
                  ))}
                </div>
              </div>

              {/* Mental Health Textarea */}
              <Form.Group className="mb-4">
                <Form.Label className="fw-semibold mb-2">
                  Mental Health & Stress <span className="text-danger">*</span>
                </Form.Label>
                <Form.Control
                  as="textarea"
                  rows={6}
                  placeholder="Share information about:
• Current stress levels (low/moderate/high)
• Anxiety or worry patterns
• Mood changes or concerns
• Sleep quality and patterns
• Emotional challenges
• Work-life balance
• Coping strategies you use
• Any history of mental health conditions
• Current support systems"
                  className={`${errors.mental_health ? 'is-invalid' : ''}`}
                  style={{ resize: 'vertical' }}
                  {...register('mental_health', {
                    required: 'Please share your mental health information',
                    minLength: {
                      value: 10,
                      message: 'Please provide more details (minimum 10 characters)',
                    },
                  })}
                />
                {errors.mental_health && (
                  <Form.Control.Feedback type="invalid" className="d-block mt-2">
                    {errors.mental_health.message}
                  </Form.Control.Feedback>
                )}

                {/* Character Count & Status */}
                <div className="mt-3 d-flex justify-content-between align-items-center">
                  <small className="text-muted">
                    {charCount} characters entered
                  </small>
                  {watchMentalHealth && (
                    <span
                      className={`badge bg-${detailLevel.color}`}
                      style={{
                        fontSize: '0.85rem',
                        padding: '0.4rem 0.8rem',
                      }}
                    >
                      {detailLevel.status}
                    </span>
                  )}
                </div>
              </Form.Group>

              {/* Coping Strategies Reference */}
              <div className="mb-4">
                <p className="text-muted fw-semibold mb-3">
                  Healthy coping strategies (for reference):
                </p>
                <div className="d-flex flex-wrap gap-2">
                  {copingStrategies.map((strategy, idx) => (
                    <span
                      key={idx}
                      className="badge bg-success bg-opacity-25 text-dark"
                      style={{ fontSize: '0.85rem', padding: '0.5rem 0.75rem' }}
                    >
                      {strategy}
                    </span>
                  ))}
                </div>
              </div>

              {/* Example Box */}
              <Card className="bg-light border-0 mb-4">
                <Card.Body className="p-3">
                  <p className="text-muted small mb-2">
                    <strong>📝 Example of comprehensive mental health information:</strong>
                  </p>
                  <p className="text-muted small mb-0" style={{ fontSize: '0.9rem' }}>
                    "Experiencing moderate work-related stress with occasional anxiety.
                    Sleep quality is affected, typically 6-7 hours per night. I cope by
                    exercising 3x/week and meditation on weekdays. Have supportive family
                    and friends. Mood generally stable but sometimes feel overwhelmed. No
                    previous mental health diagnosis. Looking to improve work-life balance."
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
                    <p className="mb-2 fw-semibold">Tips for sharing mental health information:</p>
                    <ul className="mb-0 ps-3">
                      <li>Be honest about your stress levels and emotional state</li>
                      <li>Include both challenges and positive coping mechanisms</li>
                      <li>Mention any significant life events or changes</li>
                      <li>Share what helps you feel better and what worsens stress</li>
                      <li>Include any history of anxiety, depression, or other conditions</li>
                    </ul>
                  </div>
                </div>
              </Alert>

              {/* Privacy & Confidentiality */}
              <Alert variant="success" className="border-0 bg-success bg-opacity-10 mb-4">
                <div className="d-flex align-items-start">
                  <span className="me-2" style={{ fontSize: '1.2rem' }}>
                    🔒
                  </span>
                  <small>
                    <strong>Your Privacy:</strong> All mental health information is completely
                    confidential and protected. This information is used only to provide
                    appropriate medical care and support.
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
                      <span>Next: Medical Reports</span>
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
            Mental health information is crucial for comprehensive care and is always confidential
          </small>
        </div>
      </Container>
    </div>
  );
}
