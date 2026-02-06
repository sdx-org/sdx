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

  // BMI calculation helpers
  const calculateBMI = () => {
    if (watchHeight && watchWeight) {
      const h = watchHeight / 100;
      return (watchWeight / (h * h)).toFixed(1);
    }
    return null;
  };

  const getBMICategory = () => {
    const bmi = calculateBMI();
    if (!bmi) return null;

    if (bmi < 18.5) return { label: t("bmi.underweight"), color: 'warning' };
    if (bmi < 25) return { label: t("bmi.normal"), color: 'success' };
    if (bmi < 30) return { label: t("bmi.overweight"), color: 'warning' };
    return { label: t("bmi.obese"), color: 'danger' };
  };

  const onSubmit = async (data) => {
    try {
      setApiError(null);

      if (!state.patientId) {
        throw new Error(t("errors.patientIdMissing"));
      }

      dispatch(
        consultationActions.updateDemographics({
          age: parseInt(data.age),
          gender: data.gender,
          weight_kg: parseFloat(data.weight_kg),
          height_cm: parseFloat(data.height_cm),
        })
      );

      await consultationAPI.updateConsultationDemographics(state.patientId, {
        age: parseInt(data.age),
        gender: data.gender,
        weight_kg: parseFloat(data.weight_kg),
        height_cm: parseFloat(data.height_cm),
      });

      dispatch(consultationActions.setCurrentStep('lifestyle'));
      navigate('/lifestyle');
    } catch (err) {
      console.error('Error saving demographics:', err);
      setApiError(err.message || t('errors.saveDemographicsFailed'));
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
            <small className="text-muted fw-semibold">
              {t('steps.ofTen', { step: 1 })}
            </small>
            <small className="text-muted">
              {t('steps.percentComplete', { percent: 10 })}
            </small>
          </div>
          <ProgressBar now={10} style={{ height: '8px' }} />
        </div>

        {/* Error Alert */}
        {apiError && (
          <Alert variant="danger" dismissible onClose={() => setApiError(null)}>
            <Alert.Heading>{t("errors.title")}</Alert.Heading>
            <p className="mb-0">{apiError}</p>
          </Alert>
        )}

        {/* Main Card */}
        <Card className="border-0 shadow-sm">
          <Card.Body className="p-4 p-md-5">

            {/* Header */}
            <div className="mb-5">
              <h1 className="display-6 fw-bold text-primary mb-2">📋 {t('demographics.title')}</h1>
              <p className="text-muted lead mb-0">
                {t("demographics.subtitle")}
              </p>
            </div>

            {/* Form */}
            <Form onSubmit={handleSubmit(onSubmit)}>

              {/* Age & Gender */}
              <Row className="g-4 mb-4">

                <Col md={6}>
                  <Form.Group>
                    <Form.Label className="fw-semibold mb-2">
                      {t("demographics.age")} <span className="text-danger">*</span>
                    </Form.Label>

                    <Form.Control
                      type="number"
                      placeholder={t("placeholders.age")}
                      className={`py-3 ${errors.age ? "is-invalid" : ""}`}
                      {...register("age", {
                        required: t("validation.ageRequired"),
                        min: { value: 1, message: t("validation.ageRange") },
                        max: { value: 120, message: t("validation.ageRange") },
                      })}
                    />

                    {errors.age && (
                      <Form.Control.Feedback type="invalid" className="d-block mt-2">
                        {errors.age.message}
                      </Form.Control.Feedback>
                    )}

                    {watchAge && !errors.age && (
                      <small className="text-success d-block mt-2">
                        ✓ {t('demographics.ageValue', { age: watchAge })}
                      </small>
                    )}
                  </Form.Group>
                </Col>

                <Col md={6}>
                  <Form.Group>
                    <Form.Label className="fw-semibold mb-2">
                      {t("demographics.gender")} <span className="text-danger">*</span>
                    </Form.Label>

                    <Form.Select
                      className={`py-3 ${errors.gender ? "is-invalid" : ""}`}
                      {...register("gender", {
                        required: t("validation.genderRequired"),
                      })}
                    >
                      <option value="">{t("placeholders.selectGender")}</option>
                      <option value="male">♂️ {t("gender.male")}</option>
                      <option value="female">♀️ {t("gender.female")}</option>
                      <option value="other">{t("gender.other")}</option>
                      <option value="prefer-not-to-say">{t("gender.nosay")}</option>
                    </Form.Select>

                    {errors.gender && (
                      <Form.Control.Feedback type="invalid" className="d-block mt-2">
                        {errors.gender.message}
                      </Form.Control.Feedback>
                    )}
                  </Form.Group>
                </Col>

              </Row>

              {/* Weight & Height */}
              <Row className="g-4 mb-4">

                <Col md={6}>
                  <Form.Group>
                    <Form.Label className="fw-semibold mb-2">
                      {t("demographics.weight")} <span className="text-danger">*</span>
                    </Form.Label>

                    <div className="input-group">
                      <Form.Control
                        type="number"
                        step="0.1"
                        placeholder={t("placeholders.weight")}
                        className={`py-3 ${errors.weight_kg ? 'is-invalid' : ''}`}
                        {...register("weight_kg", {
                          required: t("validation.weightRequired"),
                        })}
                      />
                      <span className="input-group-text bg-light fw-semibold">
                        {t('units.kg')}
                      </span>
                    </div>
                  </Form.Group>
                </Col>

                <Col md={6}>
                  <Form.Group>
                    <Form.Label className="fw-semibold mb-2">
                      {t("demographics.height")} <span className="text-danger">*</span>
                    </Form.Label>

                    <div className="input-group">
                      <Form.Control
                        type="number"
                        step="0.1"
                        placeholder={t("placeholders.height")}
                        className={`py-3 ${errors.height_cm ? 'is-invalid' : ''}`}
                        {...register("height_cm", {
                          required: t("validation.heightRequired"),
                        })}
                      />
                      <span className="input-group-text bg-light fw-semibold">
                        {t('units.cm')}
                      </span>
                    </div>
                  </Form.Group>
                </Col>

              </Row>

              {/* BMI Card */}
              {allFieldsFilled && bmiValue && (
                <Card className="bg-light border-0 mb-4">
                  <Card.Body className="p-3 d-flex justify-content-between align-items-center">
                    <div>
                      <p className="text-muted small mb-1">{t("bmi.title")}</p>
                      <h5 className="mb-0"><strong>{bmiValue}</strong></h5>
                    </div>

                    <p className={`badge bg-${bmiCategory.color} fs-6 mb-0`}>
                      {bmiCategory.label}
                    </p>
                  </Card.Body>
                </Card>
              )}

              {/* Info */}
              <Alert variant="info" className="border-0 bg-info bg-opacity-10 mb-4">
                <small>{t("info.confidentiality")}</small>
              </Alert>

              {/* Buttons */}
              <div className="d-flex justify-content-between align-items-center pt-4 border-top">
                <Button variant="outline-secondary" onClick={() => navigate(-1)} size="lg">
                  ← {t("common.back")}
                </Button>

                <Button type="submit" variant="primary" size="lg" disabled={isSubmitting || !allFieldsFilled}>
                  {isSubmitting ? (
                    <>
                      <Spinner animation="border" size="sm" /> {t("common.saving")}
                    </>
                  ) : (
                    <>
                      {t('actions.nextTo', { step: t('steps.lifestyle') })} →
                    </>
                  )}
                </Button>
              </div>

            </Form>
          </Card.Body>
        </Card>

        <div className="text-center mt-4">
          <small className="text-muted">
            {t('footer.faq')} <a href="#faq">{t('footer.faqLink')}</a>
          </small>
        </div>
      </Container>
    </div>
  );
}
