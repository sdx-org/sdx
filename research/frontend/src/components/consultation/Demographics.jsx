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

export default function Demographics() {
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
      age: state.formData.demographics.age || '',
      gender: state.formData.demographics.gender || '',
      weight_kg: state.formData.demographics.weight_kg || '',
      height_cm: state.formData.demographics.height_cm || '',
    },
  });

  const [apiError, setApiError] = useState(null);
  const watchAge = watch('age');
  const watchGender = watch('gender');
  const watchWeight = watch('weight_kg');
  const watchHeight = watch('height_cm');

  const calculateBMI = () => {
    if (watchHeight && watchWeight) {
      const heightInMeters = watchHeight / 100;
      const bmi = watchWeight / (heightInMeters * heightInMeters);
      return bmi.toFixed(1);
    }
    return null;
  };

  const getBMICategory = () => {
    const bmi = calculateBMI();
    if (!bmi) return null;

    if (bmi < 18.5) return { label: 'Underweight', color: 'warning' };
    if (bmi < 25) return { label: 'Normal', color: 'success' };
    if (bmi < 30) return { label: 'Overweight', color: 'warning' };
    return { label: 'Obese', color: 'danger' };
  };

  const onSubmit = async (data) => {
    try {
      setApiError(null);

      // Validate patient ID exists
      if (!state.patientId) {
        throw new Error('Patient ID not found. Please start over.');
      }

      // Update local state
      dispatch(
        consultationActions.updateDemographics({
          age: parseInt(data.age),
          gender: data.gender,
          weight_kg: parseFloat(data.weight_kg),
          height_cm: parseFloat(data.height_cm),
        })
      );

      // Call backend API
      await consultationAPI.updateConsultationDemographics(
        state.patientId,
        {
          age: parseInt(data.age),
          gender: data.gender,
          weight_kg: parseFloat(data.weight_kg),
          height_cm: parseFloat(data.height_cm),
        }
      );

      // Update current step in context
      dispatch(consultationActions.setCurrentStep('lifestyle'));

      // Navigate to next step
      navigate('/lifestyle');
    } catch (err) {
      console.error('Error saving demographics:', err);
      setApiError(err.message || 'Failed to save demographics. Please try again.');
      window.scrollTo(0, 0);
    }
  };

  const bmiValue = calculateBMI();
  const bmiCategory = getBMICategory();
  const allFieldsFilled = watchAge && watchGender && watchWeight && watchHeight;
  return (
    <div className="bg-light min-vh-100 py-4">
      <Container>
        {/* Progress Section */}
        <div className="mb-4">
          <div className="d-flex justify-content-between align-items-center mb-2">
            <small className="text-muted fw-semibold">Step 1 of 10</small>
            <small className="text-muted">10% Complete</small>
          </div>
          <ProgressBar now={10} style={{ height: '8px' }} className="mb-3" />
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
                📋 {t('demographics.title')}
              </h1>
              <p className="text-muted lead mb-0">
                Please provide your basic demographic information
              </p>
            </div>

            {/* Form */}
            <Form onSubmit={handleSubmit(onSubmit)}>
              {/* Age & Gender Row */}
              <Row className="g-4 mb-4">
                <Col md={6}>
                  <Form.Group>
                    <Form.Label className="fw-semibold mb-2">
                      Age <span className="text-danger">*</span>
                    </Form.Label>
                    <Form.Control
                      type="number"
                      placeholder="e.g., 30"
                      className={`py-3 ${errors.age ? 'is-invalid' : ''}`}
                      {...register('age', {
                        required: 'Age is required',
                        min: {
                          value: 1,
                          message: 'Age must be between 1 and 120',
                        },
                        max: {
                          value: 120,
                          message: 'Age must be between 1 and 120',
                        },
                        pattern: {
                          value: /^[0-9]+$/,
                          message: 'Age must be a valid number',
                        },
                      })}
                    />
                    {errors.age && (
                      <Form.Control.Feedback type="invalid" className="d-block mt-2">
                        {errors.age.message}
                      </Form.Control.Feedback>
                    )}
                    {watchAge && !errors.age && (
                      <small className="text-success d-block mt-2">
                        ✓ {watchAge} years old
                      </small>
                    )}
                  </Form.Group>
                </Col>

                <Col md={6}>
                  <Form.Group>
                    <Form.Label className="fw-semibold mb-2">
                      Gender <span className="text-danger">*</span>
                    </Form.Label>
                    <Form.Select
                      className={`py-3 ${errors.gender ? 'is-invalid' : ''}`}
                      {...register('gender', {
                        required: 'Gender is required',
                      })}
                    >
                      <option value="">Select your gender</option>
                      <option value="male">♂️ Male</option>
                      <option value="female">♀️ Female</option>
                      <option value="other">Other</option>
                      <option value="prefer-not-to-say">
                        Prefer not to say
                      </option>
                    </Form.Select>
                    {errors.gender && (
                      <Form.Control.Feedback type="invalid" className="d-block mt-2">
                        {errors.gender.message}
                      </Form.Control.Feedback>
                    )}
                    {watchGender && !errors.gender && (
                      <small className="text-success d-block mt-2">
                        ✓ {watchGender.charAt(0).toUpperCase() + watchGender.slice(1)}
                      </small>
                    )}
                  </Form.Group>
                </Col>
              </Row>

              {/* Weight & Height Row */}
              <Row className="g-4 mb-4">
                <Col md={6}>
                  <Form.Group>
                    <Form.Label className="fw-semibold mb-2">
                      Weight <span className="text-danger">*</span>
                    </Form.Label>
                    <div className="input-group">
                      <Form.Control
                        type="number"
                        step="0.1"
                        placeholder="e.g., 70"
                        className={`py-3 ${errors.weight_kg ? 'is-invalid' : ''}`}
                        {...register('weight_kg', {
                          required: 'Weight is required',
                          min: {
                            value: 1,
                            message: 'Weight must be greater than 0',
                          },
                          max: {
                            value: 500,
                            message: 'Please enter a valid weight',
                          },
                          pattern: {
                            value: /^[0-9]+\.?[0-9]*$/,
                            message: 'Weight must be a valid number',
                          },
                        })}
                      />
                      <span className="input-group-text bg-light fw-semibold">
                        kg
                      </span>
                    </div>
                    {errors.weight_kg && (
                      <Form.Control.Feedback type="invalid" className="d-block mt-2">
                        {errors.weight_kg.message}
                      </Form.Control.Feedback>
                    )}
                    {watchWeight && !errors.weight_kg && (
                      <small className="text-success d-block mt-2">
                        ✓ {watchWeight} kg
                      </small>
                    )}
                  </Form.Group>
                </Col>

                <Col md={6}>
                  <Form.Group>
                    <Form.Label className="fw-semibold mb-2">
                      Height <span className="text-danger">*</span>
                    </Form.Label>
                    <div className="input-group">
                      <Form.Control
                        type="number"
                        step="0.1"
                        placeholder="e.g., 175"
                        className={`py-3 ${errors.height_cm ? 'is-invalid' : ''}`}
                        {...register('height_cm', {
                          required: 'Height is required',
                          min: {
                            value: 1,
                            message: 'Height must be greater than 0',
                          },
                          max: {
                            value: 300,
                            message: 'Please enter a valid height',
                          },
                          pattern: {
                            value: /^[0-9]+\.?[0-9]*$/,
                            message: 'Height must be a valid number',
                          },
                        })}
                      />
                      <span className="input-group-text bg-light fw-semibold">
                        cm
                      </span>
                    </div>
                    {errors.height_cm && (
                      <Form.Control.Feedback type="invalid" className="d-block mt-2">
                        {errors.height_cm.message}
                      </Form.Control.Feedback>
                    )}
                    {watchHeight && !errors.height_cm && (
                      <small className="text-success d-block mt-2">
                        ✓ {watchHeight} cm
                      </small>
                    )}
                  </Form.Group>
                </Col>
              </Row>

              {/* BMI Calculator Card */}
              {allFieldsFilled && bmiValue && (
                <Card className="bg-light border-0 mb-4">
                  <Card.Body className="p-3">
                    <div className="d-flex align-items-center justify-content-between">
                      <div>
                        <p className="text-muted small mb-1">Body Mass Index (BMI)</p>
                        <h5 className="mb-0">
                          <strong>{bmiValue}</strong>
                        </h5>
                      </div>
                      <div className="text-end">
                        <p
                          className={`badge bg-${bmiCategory.color} text-white fs-6 mb-0`}
                        >
                          {bmiCategory.label}
                        </p>
                      </div>
                    </div>
                  </Card.Body>
                </Card>
              )}

              {/* Info Alert */}
              <Alert variant="info" className="border-0 bg-info bg-opacity-10 mb-4">
                <div className="d-flex align-items-start">
                  <span className="me-2" style={{ fontSize: '1.2rem' }}>
                    ℹ️
                  </span>
                  <small>
                    All information is kept confidential and used only for medical
                    assessment purposes. Your data is encrypted and securely stored.
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
                      <span>Next: Lifestyle</span>
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
            Questions? Check our <a href="#faq" className="text-decoration-none">FAQ</a> or contact
            support.
          </small>
        </div>
      </Container>
    </div>
  );
}
