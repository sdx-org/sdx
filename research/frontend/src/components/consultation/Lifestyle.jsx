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
  Badge,
} from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useConsultation, consultationActions } from '../../context/ConsultationContext';
import consultationAPI from '../../services/api';

export default function Lifestyle() {
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
      diet: state.formData.lifestyle.diet || '',
      sleep_hours: state.formData.lifestyle.sleep_hours || '',
      physical_activity: state.formData.lifestyle.physical_activity || '',
      mental_exercises: state.formData.lifestyle.mental_exercises || '',
    },
  });

  const [apiError, setApiError] = useState(null);
  const watchDiet = watch('diet');
  const watchSleep = watch('sleep_hours');
  const watchExercise = watch('physical_activity');
  const watchMental = watch('mental_exercises');

  const getSleepQuality = () => {
    const sleep = parseFloat(watchSleep);
    if (sleep < 5) return { label: t('lifestyle.sleepQuality.poor'), color: 'danger' };
    if (sleep < 7) return { label: t('lifestyle.sleepQuality.fair'), color: 'warning' };
    if (sleep <= 9) return { label: t('lifestyle.sleepQuality.good'), color: 'success' };
    return { label: t('lifestyle.sleepQuality.excessive'), color: 'warning' };
  };

  const allFieldsFilled = watchDiet && watchSleep && watchExercise && watchMental;
  const onSubmit = async (data) => {
    try {
      setApiError(null);

      // Validate patient ID exists
      if (!state.patientId) {
        throw new Error(t('errors.patientIdMissing'));
      }

      // Update local state
      dispatch(
        consultationActions.updateLifestyle({
          diet: data.diet,
          sleep_hours: parseFloat(data.sleep_hours),
          physical_activity: data.physical_activity,
          mental_exercises: data.mental_exercises,
        })
      );

      // Call backend API
      await consultationAPI.updateConsultationLifestyle(state.patientId, {
        diet: data.diet,
        sleep_hours: parseFloat(data.sleep_hours),
        physical_activity: data.physical_activity,
        mental_exercises: data.mental_exercises,
      });

      // Update current step in context
      dispatch(consultationActions.setCurrentStep('symptoms'));

      // Navigate to next step
      navigate('/symptoms');
    } catch (err) {
      console.error('Error saving lifestyle data:', err);
      setApiError(err.message || t('errors.saveLifestyleFailed'));
      window.scrollTo(0, 0);
    }
  };

  const sleepQuality = watchSleep ? getSleepQuality() : null;
  return (
    <div className="bg-light min-vh-100 py-4">
      <Container>
        {/* Progress Section */}
        <div className="mb-4">
          <div className="d-flex justify-content-between align-items-center mb-2">
            <small className="text-muted fw-semibold">
              {t('steps.ofTen', { step: 2 })}
            </small>
            <small className="text-muted">
              {t('steps.percentComplete', { percent: 20 })}
            </small>
          </div>
          <ProgressBar now={20} style={{ height: '8px' }} className="mb-3" />
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
                🏃 {t('lifestyle.title')}
              </h1>
              <p className="text-muted lead mb-0">
                {t('lifestyle.subtitle')}
              </p>
            </div>

            {/* Form */}
            <Form onSubmit={handleSubmit(onSubmit)}>
              {/* Diet & Sleep Row */}
              <Row className="g-4 mb-4">
                <Col md={6}>
                  <Form.Group>
                    <Form.Label className="fw-semibold mb-2">
                      {t('lifestyle.dietLabel')} <span className="text-danger">*</span>
                    </Form.Label>
                    <Form.Control
                      placeholder={t('lifestyle.dietPlaceholder')}
                      className={`py-3 ${errors.diet ? 'is-invalid' : ''}`}
                      {...register('diet', {
                        required: t('validation.dietRequired'),
                        minLength: {
                          value: 3,
                          message: t('validation.dietMin'),
                        },
                      })}
                    />
                    {errors.diet && (
                      <Form.Control.Feedback type="invalid" className="d-block mt-2">
                        {errors.diet.message}
                      </Form.Control.Feedback>
                    )}
                    {watchDiet && !errors.diet && (
                      <small className="text-success d-block mt-2">
                        ✓ {watchDiet}
                      </small>
                    )}
                  </Form.Group>
                </Col>

                <Col md={6}>
                  <Form.Group>
                    <Form.Label className="fw-semibold mb-2">
                      {t('lifestyle.sleepLabel')} <span className="text-danger">*</span>
                    </Form.Label>
                    <div className="input-group">
                      <Form.Control
                        type="number"
                        step="0.5"
                        placeholder={t('lifestyle.sleepPlaceholder')}
                        className={`py-3 ${errors.sleep_hours ? 'is-invalid' : ''}`}
                        {...register('sleep_hours', {
                          required: t('validation.sleepRequired'),
                          min: {
                            value: 1,
                            message: t('validation.sleepMin'),
                          },
                          max: {
                            value: 24,
                            message: t('validation.sleepMax'),
                          },
                          pattern: {
                            value: /^[0-9]+\.?[0-9]*$/,
                            message: t('validation.sleepNumber'),
                          },
                        })}
                      />
                      <span className="input-group-text bg-light fw-semibold">
                        {t('lifestyle.sleepUnit')}
                      </span>
                    </div>
                    {errors.sleep_hours && (
                      <Form.Control.Feedback type="invalid" className="d-block mt-2">
                        {errors.sleep_hours.message}
                      </Form.Control.Feedback>
                    )}
                    {watchSleep && !errors.sleep_hours && sleepQuality && (
                      <div className="mt-2 d-flex align-items-center gap-2">
                        <small className="text-success">
                          ✓ {t('lifestyle.sleepHoursValue', { hours: watchSleep })}
                        </small>
                        <Badge bg={sleepQuality.color}>{sleepQuality.label}</Badge>
                      </div>
                    )}
                  </Form.Group>
                </Col>
              </Row>

              {/* Exercise & Mental Activities Row */}
              <Row className="g-4 mb-4">
                <Col md={6}>
                  <Form.Group>
                    <Form.Label className="fw-semibold mb-2">
                      {t('lifestyle.exerciseLabel')} <span className="text-danger">*</span>
                    </Form.Label>
                    <Form.Control
                      placeholder={t('lifestyle.exercisePlaceholder')}
                      className={`py-3 ${errors.physical_activity ? 'is-invalid' : ''}`}
                      {...register('physical_activity', {
                        required: t('validation.exerciseRequired'),
                        minLength: {
                          value: 3,
                          message: t('validation.exerciseMin'),
                        },
                      })}
                    />
                    {errors.physical_activity && (
                      <Form.Control.Feedback type="invalid" className="d-block mt-2">
                        {errors.physical_activity.message}
                      </Form.Control.Feedback>
                    )}
                    {watchExercise && !errors.physical_activity && (
                      <small className="text-success d-block mt-2">
                        ✓ {watchExercise}
                      </small>
                    )}
                  </Form.Group>
                </Col>

                <Col md={6}>
                  <Form.Group>
                    <Form.Label className="fw-semibold mb-2">
                      {t('lifestyle.mentalLabel')} <span className="text-danger">*</span>
                    </Form.Label>
                    <Form.Control
                      placeholder={t('lifestyle.mentalPlaceholder')}
                      className={`py-3 ${errors.mental_exercises ? 'is-invalid' : ''}`}
                      {...register('mental_exercises', {
                        required: t('validation.mentalRequired'),
                        minLength: {
                          value: 3,
                          message: t('validation.mentalMin'),
                        },
                      })}
                    />
                    {errors.mental_exercises && (
                      <Form.Control.Feedback type="invalid" className="d-block mt-2">
                        {errors.mental_exercises.message}
                      </Form.Control.Feedback>
                    )}
                    {watchMental && !errors.mental_exercises && (
                      <small className="text-success d-block mt-2">
                        ✓ {watchMental}
                      </small>
                    )}
                  </Form.Group>
                </Col>
              </Row>

              {/* Info Alert */}
              <Alert variant="info" className="border-0 bg-info bg-opacity-10 mb-4">
                <div className="d-flex align-items-start">
                  <span className="me-2" style={{ fontSize: '1.2rem' }}>
                    ℹ️
                  </span>
                  <small>
                    {t('lifestyle.info')}
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
                  disabled={isSubmitting || !allFieldsFilled}
                  className="d-flex align-items-center gap-2"
                >
                  {isSubmitting ? (
                    <>
                      <Spinner animation="border" size="sm" />
                      <span>{t('common.saving')}</span>
                    </>
                  ) : (
                    <>
                      <span>{t('actions.nextTo', { step: t('steps.symptoms') })}</span>
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
            {t('lifestyle.footer')}
          </small>
        </div>
      </Container>
    </div>
  );
}
