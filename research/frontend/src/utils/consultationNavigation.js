import consultationAPI from '../services/api';
import { consultationActions } from '../context/ConsultationContext';

export async function resumeConsultationForPatient({ patientId, dispatch, navigate }) {
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
    const savedState = localStorage.getItem(`consultationState_${patientId}`);
    if (savedState) {
      const parsedState = JSON.parse(savedState);
      if (parsedState.formData.demographics) {
        dispatch(
          consultationActions.updateDemographics(parsedState.formData.demographics)
        );
      }
      if (parsedState.formData.lifestyle) {
        dispatch(
          consultationActions.updateLifestyle(parsedState.formData.lifestyle)
        );
      }
      if (parsedState.formData.symptoms) {
        dispatch(
          consultationActions.updateSymptoms(parsedState.formData.symptoms)
        );
      }
      if (parsedState.formData.mental) {
        dispatch(
          consultationActions.updateMentalHealth(parsedState.formData.mental)
        );
      }
    }
  }

  navigate(`/${currentStep}`);
  return consultationData;
}
