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
    if (sleep < 5) return { label: 'Poor', color: 'danger' };
    if (sleep < 7) return { label: 'Fair', color: 'warning' };
    if (sleep <= 9) return { label: 'Good', color: 'success' };
    return { label: 'Excessive', color: 'warning' };
  };

  const allFieldsFilled = watchDiet && watchSleep && watchExercise && watchMental;
  const onSubmit = async (data) => {
    try {
      setApiError(null);

      // Validate patient ID exists
      if (!state.patientId) {
        throw new Error('Patient ID not found. Please start over.');
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
      setApiError(err.message || 'Failed to save lifestyle data. Please try again.');
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
            <small className="text-muted fw-semibold">Step 2 of 10</small>
            <small className="text-muted">20% Complete</small>
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
                🏃 {t('lifestyle.title')}
              </h1>
              <p className="text-muted lead mb-0">
                Tell us about your daily habits and routines
              </p>
            </div>

            {/* Form */}
            <Form onSubmit={handleSubmit(onSubmit)}>
              {/* Diet & Sleep Row */}
              <Row className="g-4 mb-4">
                <Col md={6}>
                  <Form.Group>
                    <Form.Label className="fw-semibold mb-2">
                      Diet Type <span className="text-danger">*</span>
                    </Form.Label>
                    <Form.Control
                      placeholder="e.g., Balanced, Keto, Vegetarian, Vegan"
                      className={`py-3 ${errors.diet ? 'is-invalid' : ''}`}
                      {...register('diet', {
                        required: 'Diet type is required',
                        minLength: {
                          value: 3,
                          message: 'Please describe your diet pattern',
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
                      Sleep Hours/Night <span className="text-danger">*</span>
                    </Form.Label>
                    <div className="input-group">
                      <Form.Control
                        type="number"
                        step="0.5"
                        placeholder="e.g., 7.5"
                        className={`py-3 ${errors.sleep_hours ? 'is-invalid' : ''}`}
                        {...register('sleep_hours', {
                          required: 'Sleep duration is required',
                          min: {
                            value: 1,
                            message: 'Sleep must be at least 1 hour',
                          },
                          max: {
                            value: 24,
                            message: 'Sleep cannot exceed 24 hours',
                          },
                          pattern: {
                            value: /^[0-9]+\.?[0-9]*$/,
                            message: 'Sleep must be a valid number',
                          },
                        })}
                      />
                      <span className="input-group-text bg-light fw-semibold">
                        hrs
                      </span>
                    </div>
                    {errors.sleep_hours && (
                      <Form.Control.Feedback type="invalid" className="d-block mt-2">
                        {errors.sleep_hours.message}
                      </Form.Control.Feedback>
                    )}
                    {watchSleep && !errors.sleep_hours && sleepQuality && (
                      <div className="mt-2 d-flex align-items-center gap-2">
                        <small className="text-success">✓ {watchSleep} hours</small>
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
                      Physical Activity <span className="text-danger">*</span>
                    </Form.Label>
                    <Form.Control
                      placeholder="e.g., Running 3x/week, Gym daily, Walks"
                      className={`py-3 ${errors.physical_activity ? 'is-invalid' : ''}`}
                      {...register('physical_activity', {
                        required: 'Physical activity is required',
                        minLength: {
                          value: 3,
                          message: 'Please describe your exercise routine',
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
                      Mental Exercises <span className="text-danger">*</span>
                    </Form.Label>
                    <Form.Control
                      placeholder="e.g., Meditation, Reading, Yoga, Journaling"
                      className={`py-3 ${errors.mental_exercises ? 'is-invalid' : ''}`}
                      {...register('mental_exercises', {
                        required: 'Mental activities are required',
                        minLength: {
                          value: 3,
                          message: 'Please describe your mental activities',
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
                    Lifestyle factors like sleep, exercise, and diet significantly impact your
                    overall health and help us provide better assessments.
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
                  disabled={isSubmitting || !allFieldsFilled}
                  className="d-flex align-items-center gap-2"
                >
                  {isSubmitting ? (
                    <>
                      <Spinner animation="border" size="sm" />
                      <span>Saving...</span>
                    </>
                  ) : (
                    <>
                      <span>Next: Symptoms</span>
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
            Your health information is encrypted and securely stored
          </small>
        </div>
      </Container>
    </div>
  );
}
