import consultationAPI from '../services/api';
import { consultationActions } from '../context/ConsultationContext';

/**
 * ALLOWED_STEPS
 */
const ALLOWED_STEPS = new Set(['demographics', 'lifestyle', 'symptoms', 'mental', 'medical-reports', 'wearable-data', 'confirmation', 'exams', 'diagnosis']);

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
  return ALLOWED_STEPS.has(s) ? s : 'demographics';
}

/**
 * safeLangFromServer
 */
function safeLangFromServer(lang, allowed = ['en']) {
  const s = String(lang || '').toLowerCase();
  return allowed.includes(s) ? s : 'en';
}

/**
 * shouldUseLocalState
 */
function shouldUseLocalState(localUpdatedAt, serverUpdatedAt) {
  const localMs = Date.parse(String(localUpdatedAt));
  const serverMs = Date.parse(String(serverUpdatedAt));
  if (!Number.isFinite(localMs) || !Number.isFinite(serverMs)) return false;
  return localMs > serverMs;
}

/**
 * isPlainObject
 */
function isPlainObject(v) {
  return v !== null && typeof v === 'object' && !Array.isArray(v);
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
  const language = safeLangFromServer(consultationData.lang, ['en', 'es']);

  dispatch(
    consultationActions.initConsultation(
      patientId,
      language,
      targetStep
    )
  );

  const shouldPrefillFromLocalStorage = ALLOWED_STEPS.has(targetStep);

  if (shouldPrefillFromLocalStorage) {
    const parsedState = getSavedConsultationState(patientId);
    const fd = parsedState?.formData;
    const useLocal = shouldUseLocalState(
      parsedState?.updatedAt,
      consultationData?.updated_at
    );

    if (useLocal && isPlainObject(fd?.demographics)) {
      dispatch(consultationActions.updateDemographics(fd.demographics));
    }
    if (useLocal && isPlainObject(fd?.lifestyle)) {
      dispatch(consultationActions.updateLifestyle(fd.lifestyle));
    }
    if (useLocal && isPlainObject(fd?.symptoms)) {
      dispatch(consultationActions.updateSymptoms(fd.symptoms));
    }
    if (useLocal && isPlainObject(fd?.mental)) {
      dispatch(consultationActions.updateMentalHealth(fd.mental));
    }
  }

  const STEP_ROUTES = {
    demographics: '/demographics',
    lifestyle: '/lifestyle',
    symptoms: '/symptoms',
    mental: '/mental',
    'medical-reports': '/medical-reports',
    'wearable-data': '/wearable-data',
    diagnosis: '/diagnosis',
    exams: '/exams',
    confirmation: '/confirmation'
  };

  navigate(STEP_ROUTES[targetStep] || '/demographics');
  return consultationData;
}
