import consultationAPI from '../services/api';
import { consultationActions } from '../context/ConsultationContext';

/**
 * getSavedConsultationState
 */
function getSavedConsultationState(patientId) {
  try {
    if (typeof window === 'undefined' || !window.localStorage) return null;
    const raw = window.localStorage.getItem(`consultationState_${patientId}`);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : null;
  } catch {
    return null;
  }
}

/**
 * safeStepFromServer
 */
function safeStepFromServer(step) {
  return /^[a-z0-9-]+$/i.test(step) ? step : 'demographics';
}

export async function resumeConsultationForPatient({ patientId, dispatch, navigate }) {
  if (!patientId) {
    throw new Error('resumeConsultationForPatient: patientId is required.');
  }

  const consultationData = await consultationAPI.getConsultationStatus(patientId);

  if (!consultationData) {
    throw new Error('No consultation data found for this patient.');
  }

  const currentStep = consultationData.current_step || 'demographics';
  const language = consultationData.lang || 'en';

  dispatch(
    consultationActions.initConsultation(
      patientId,
      language,
      currentStep
    )
  );

  const shouldPrefillFromLocalStorage = [
    'demographics',
    'lifestyle',
    'symptoms',
    'mental',
  ].includes(currentStep);

  if (shouldPrefillFromLocalStorage) {
    const parsedState = getSavedConsultationState(patientId);
    const fd = parsedState?.formData;

    if (fd?.demographics) {
      dispatch(consultationActions.updateDemographics(fd.demographics));
    }
    if (fd?.lifestyle) {
      dispatch(consultationActions.updateLifestyle(fd.lifestyle));
    }
    if (fd?.symptoms) {
      dispatch(consultationActions.updateSymptoms(fd.symptoms));
    }
    if (fd?.mental) {
      dispatch(consultationActions.updateMentalHealth(fd.mental));
    }
  }

  const targetStep = safeStepFromServer(currentStep);
  navigate(`/${targetStep}`);
  return consultationData;
}
