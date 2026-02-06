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
    if (length < 10) return { status: t('symptoms.severity.tooShort'), color: 'danger' };
    if (length < 50) return { status: t('symptoms.severity.brief'), color: 'warning' };
    if (length < 150) return { status: t('symptoms.severity.good'), color: 'info' };
    return { status: t('symptoms.severity.detailed'), color: 'success' };
  };

  const commonSymptoms = t('symptoms.commonList', { returnObjects: true });
  const tips = t('symptoms.tips', { returnObjects: true });

  const isFormComplete = watchSymptoms && watchSymptoms.length >= 10;
  const charCount = getCharacterCount();
  const severityHint = getSeverityHint(charCount);
  const onSubmit = async (data) => {
    try {
      setApiError(null);

      // Validate patient ID exists
      if (!state.patientId) {
        throw new Error(t('errors.patientIdMissing'));
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
      setApiError(err.message || t('errors.saveSymptomsFailed'));
      window.scrollTo(0, 0);
    }
  };

  return (
    <div className="bg-light min-vh-100 py-4">
      <Container>
        {/* Progress Section */}
        <div className="mb-4">
          <div className="d-flex justify-content-between align-items-center mb-2">
            <small className="text-muted fw-semibold">
              {t('steps.ofTen', { step: 3 })}
            </small>
            <small className="text-muted">
              {t('steps.percentComplete', { percent: 30 })}
            </small>
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
                🤒 {t('symptoms.title')}
              </h1>
              <p className="text-muted lead mb-0">
                {t('symptoms.subtitle')}
              </p>
            </div>

            {/* Form */}
            <Form onSubmit={handleSubmit(onSubmit)}>
              {/* Common Symptoms Reference */}
              <div className="mb-5">
                <p className="text-muted fw-semibold mb-3">
                  {t('symptoms.commonTitle')}
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
                  {t('symptoms.describeLabel')} <span className="text-danger">*</span>
                </Form.Label>
                <Form.Control
                  as="textarea"
                  rows={6}
                  placeholder={t('symptoms.placeholder')}
                  className={`${errors.symptoms ? 'is-invalid' : ''}`}
                  style={{ resize: 'vertical' }}
                  {...register('symptoms', {
                    required: t('validation.symptomsRequired'),
                    minLength: {
                      value: 10,
                      message: t('validation.minDetails'),
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
                    {t('common.charactersEntered', { count: charCount })}
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
                    <strong>📝 {t('symptoms.exampleTitle')}</strong>
                  </p>
                  <p className="text-muted small mb-0" style={{ fontSize: '0.9rem' }}>
                    {t('symptoms.exampleText')}
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
                    <p className="mb-2 fw-semibold">
                      {t('symptoms.tipsTitle')}
                    </p>
                    <ul className="mb-0 ps-3">
                      {Array.isArray(tips) &&
                        tips.map((tip, idx) => <li key={idx}>{tip}</li>)}
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
                    <strong>{t('symptoms.emergencyTitle')}:</strong>{' '}
                    {t('symptoms.emergencyText')}
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
                      <span>{t('common.saving')}</span>
                    </>
                  ) : (
                    <>
                      <span>{t('actions.nextTo', { step: t('steps.mentalHealth') })}</span>
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
            {t('symptoms.footer')}
          </small>
        </div>
      </Container>
    </div>
  );
}
