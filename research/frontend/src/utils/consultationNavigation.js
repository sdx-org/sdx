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
  const s = String(step || '').toLowerCase();
  const ALLOWED_STEPS = new Set(['demographics', 'lifestyle', 'symptoms', 'mental']);
  return ALLOWED_STEPS.has(s) ? s : 'demographics';
}

export async function resumeConsultationForPatient({ patientId, dispatch, navigate }) {
  if (!patientId) {
    throw new Error('resumeConsultationForPatient: patientId is required.');
  }

  const consultationData = await consultationAPI.getConsultationStatus(patientId);

  if (!consultationData) {
    throw new Error('No consultation data found for this patient.');
  }

  const rawStep = consultationData.current_step ?? 'demographics';
  const targetStep = safeStepFromServer(rawStep);
  const language = consultationData.lang || 'en';

  dispatch(
    consultationActions.initConsultation(
      patientId,
      language,
      targetStep
    )
  );

  const shouldPrefillFromLocalStorage = [
    'demographics',
    'lifestyle',
    'symptoms',
    'mental',
  ].includes(targetStep);

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

  navigate(`/${targetStep}`);
  return consultationData;
}
