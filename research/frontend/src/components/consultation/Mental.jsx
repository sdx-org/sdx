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
    if (length < 10) return { status: t('mental.detailLevel.tooShort'), color: 'danger' };
    if (length < 50) return { status: t('mental.detailLevel.brief'), color: 'warning' };
    if (length < 150) return { status: t('mental.detailLevel.good'), color: 'info' };
    return { status: t('mental.detailLevel.comprehensive'), color: 'success' };
  };

  const stressAreas = t('mental.stressAreas', { returnObjects: true });
  const copingStrategies = t('mental.copingStrategies', { returnObjects: true });
  const tips = t('mental.tips', { returnObjects: true });
  const isFormComplete = watchMentalHealth && watchMentalHealth.length >= 10;
  const charCount = getCharacterCount();
  const detailLevel = getDetailLevel(charCount);
  const onSubmit = async (data) => {
    try {
      setApiError(null);

      // Validate patient ID exists
      if (!state.patientId) {
        throw new Error(t('errors.patientIdMissing'));
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
      setApiError(err.message || t('errors.saveMentalFailed'));
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
              {t('steps.ofTen', { step: 4 })}
            </small>
            <small className="text-muted">
              {t('steps.percentComplete', { percent: 40 })}
            </small>
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
                🧠 {t('mental.title')}
              </h1>
              <p className="text-muted lead mb-0">
                {t('mental.subtitle')}
              </p>
            </div>

            {/* Form */}
            <Form onSubmit={handleSubmit(onSubmit)}>
              {/* Common Stress Areas */}
              <div className="mb-5">
                <p className="text-muted fw-semibold mb-3">
                  {t('mental.stressTitle')}
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
                  {t('mental.label')} <span className="text-danger">*</span>
                </Form.Label>
                <Form.Control
                  as="textarea"
                  rows={6}
                  placeholder={t('mental.placeholder')}
                  className={`${errors.mental_health ? 'is-invalid' : ''}`}
                  style={{ resize: 'vertical' }}
                  {...register('mental_health', {
                    required: t('validation.mentalHealthRequired'),
                    minLength: {
                      value: 10,
                      message: t('validation.minDetails'),
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
                    {t('common.charactersEntered', { count: charCount })}
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
                  {t('mental.copingTitle')}
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
                    <strong>📝 {t('mental.exampleTitle')}</strong>
                  </p>
                  <p className="text-muted small mb-0" style={{ fontSize: '0.9rem' }}>
                    {t('mental.exampleText')}
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
                    <p className="mb-2 fw-semibold">{t('mental.tipsTitle')}</p>
                    <ul className="mb-0 ps-3">
                      {Array.isArray(tips) &&
                        tips.map((tip, idx) => <li key={idx}>{tip}</li>)}
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
                    <strong>{t('mental.privacyTitle')}:</strong>{' '}
                    {t('mental.privacyText')}
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
                      <span>{t('actions.nextTo', { step: t('steps.medicalReports') })}</span>
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
            {t('mental.footer')}
          </small>
        </div>
      </Container>
    </div>
  );
}
