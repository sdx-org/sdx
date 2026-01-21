/**
 * API client for HiperHealth Frontend
 * Maps to backend at http://localhost:8000/api/*
 */

export const API_BASE=
    import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export function api(path){
    return `${API_BASE}${path}`
}

export const consultationAPI={
    // Patient Management
    /**
     * POST /api/patients
     * @param {string} language - 'en', 'es'
     * @returns {Promise<{patient_id: string, lang: string, created_at: string}>}
     */
    async createPatient(language='en'){
        const response=await fetch(api('/api/patients'),{
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body:JSON.stringify({lang:language})
        });

        if(!response.ok){
            throw new Error(`Create patient failed: ${response.status}`);
        }

        return response.json();
    },


    /**
     * Get all patients for dashboard
     * GET /api/patients
     */
    async getPatients(){
        const response= await fetch(api('/api/patients'));

        if(!response.ok){
            throw new Error(`Get patients failed: ${response.status}`);
        }

        return response.json();
    },


    /**
     * Delete a patient
     * DELETE /api/patients/{patient_id}
     */
    async deletePatient(patientId){
        const response= await fetch(api(`/api/patients/${patientId}`),{
            method: 'DELETE'
        });

        if(!response.ok){
            throw new Error(`Delete patient failed: ${response.status}`)
        }

        return response.json();
    },


    /**
     * Get consultation status and all patient data.
     * GET /api/consultations/{patient_id}/status
     */
    async getConsultationStatus(patientId){
        const response=await fetch(
            api(`/api/consultations/${patientId}/status`)
        );

        if(!response.ok){
            throw new Error(
                `Get consultation status Failed: ${response.status}`
            );
        }

        return response.json();
    },


    // Step 1: Demographics
    /**
     * Submit Demographics
     * POST /api/consultations/{patient_id}/demographics
     * Expects backend feilds: weight_kg, height_cm
     */
    async updateConsultationDemographics(patientId,data){
        const response= await fetch(
            api(`/api/consultations/${patientId}/demographics`),
            {
                method: 'POST',
                headers:{
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    age: data.age,
                    gender: data.gender,
                    weight_kg: data.weight_kg,
                    height_cm: data.height_cm,
                })
            }
        )

        if(!response.ok){
            throw new Error(`Submit demographics failed: ${response.status}`);
        }

        return response.json();
    },


    //Step 2: Lifestyle
    /**
     * Submit lifestyle
     * POST /api/consultations/{patient_id}/lifestyle
     */
    async updateConsultationLifestyle(patientId, data){
        const response= await fetch(
            api(`/api/consultations/${patientId}/lifestyle`),
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    diet: data.diet,
                    sleep_hours: data.sleep_hours,
                    physical_activity: data.physical_activity,
                    mental_exercises: data.mental_exercises,
                })
            }
        )

        if(!response.ok){
            throw new Error(`Submit lifestyle failed: ${response.status}`)
        }

        return response.json()
    },


    //Step 3: Symptoms
    /**
     * Submit symptoms
     * POST /api/consultations/{patient_id}/symptoms
     */
    async updateConsultationSymptoms(patientId,data){
        const response = await fetch(
            api(`/api/consultations/${patientId}/symptoms`),
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    symptoms:data.symptoms
                })
            }
        );

        if(!response.ok){
            throw new Error(`Submit symptoms failed: ${response.status}`)
        }

        return response.json();
    },


    //Step 4: Mental Health
    /**
     * Submit Mental Health
     * POST /api/consultations/{patient_id}/mental
     */
    async updateConsultationMentalHealth(patientId, data){
        const response =await fetch(
            api(`/api/consultations/${patientId}/mental`),
            {
                method: 'POST',
                headers:{
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    mental_health: data.mental_health
                })
            }
        )

        if(!response.ok){
            throw new Error(`Submit menatl health failed: ${response.status}`)
        }

        return response.json();
    },


    //Step 5: Medical Reports
    /**
     * Upload medical resports (multiple files).
     * POST /api/consultations/{patient_id}medical-reports
     */
    async uploadMedicalReports(patientId,files){
        const formData=new FormData();
        files.forEach((file)=>formData.append('files',file))

        const response =await fetch(
            api(`/api/consultations/${patientId}/medical-reports/upload`),
            {
                method:'POST',
                body: formData
            }
        );

        if(!response.ok){
            throw new Error(
                `Upload medical reports failed: ${response.status}`
            )
        }

        return response.json();
    },

    /**
     * Skip medical reports
     * POST /api/consultations/{patient_id}/medical-reports/skip
     */
    async skipMedicalReports(patientId){
        const response= await fetch(
            api(`/api/consultations/${patientId}/medical-reports/skip`),
            {
                method:'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            }
        )

        if(!response.ok){
            throw new Error('Skip medical reports failed: ${response.status}');
        }

        return response.json();
    },

    /**
     * List Medical Reports
     */
    async getMedicalReports(patientId){
        const response= await fetch(
            api(`/api/consultations/${patientId}/medical-reports`)
        )

        if(!response.ok){
            throw new Error(`Get medical reports failed: ${response.status}`)
        }

        return response.json()
    },

    //Step 6: Wearable Data
    /**
     * Upload wearable data
     * POST /api/consultations/{patient_id}/wearable-data/upload
     */
    async uploadWearableData(patientId,file){
        const formData=new FormData();
        formData.append('file',file);
        // TODO: add multiple file upload
        const response= await fetch(
            api(`/api/consultations/${patientId}/wearable-data/upload`),
            {
                method:'POST',
                body:formData
            }
        )

        if(!response.ok){
            throw new Error(`Upload Wearable data failed: ${response.status}`)
        }

        return response.json();
    },

    /**
     * Skip wearable data.
     * POST /api/consultations/{patient_id}/wearable-data/skip
     */
    async skipWearableData(patientId){
        const response= await fetch(
            api(`/api/consultations/${patientId}/wearable-data/skip`),
            {
                method:'POST',
                headers:{
                    'Content-Type': 'application/json'
                }
            }
        )

        if(!response.ok){
            throw new Error(`Skip wearable data failed: ${response.status}`);
        }

        return response.json();
    },

    //Step 7: AI Diagnosis
    /**
     * Get diagnosis suggestions
     * GET /api/consultations/{patient_id}/diagnosis
     */
    async getDiagnosisSuggestions(patientId){
        const response= await fetch(
            api(`/api/consultations/${patientId}/diagnosis`)
        )

        if(!response.ok){
            throw new Error(
                `Get diagnosis suggestions failed: ${response.status}`
            )
        }

        return response.json();
    },

    /**
     * Submit selected diagnosis+evaluations
     * POST /api/consultations/{patient_id}/diagnosis
     */
    async saveDiagnosisSelections(patientId,data){
        const response=await fetch(
            api(`/api/consultations/${patientId}/diagnosis`),
            {
                method: 'POST',
                headers:{
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    selected_diagnoses: data.selected_diagnoses,
                    evaluations: data.evaluations,
                })
            }
        )

        if(!response.ok){
            throw new Error(
                `Submit diagnosis failed: ${response.status}`
            );
        }

        return response.json();
    },

    //Step 8: AI Exams
    /**
     * Get exam suggestions
     * GET /api/consultations/{patient_id}/exams
     */
    async getExamSuggestions(patientId){
        const response= await fetch(
            api(`/api/consultations/${patientId}/exams`)
        );

        if(!response.ok)
        {
            throw new Error(
                `Get Exam suggestions failed: ${response.status}`
            )
        }

        return response.json();
    },

    /**
     * Submit selected exams+evaluations(finalizes consultation).
     * POST /api/consultations/{patient_id}/exams
    */
   async saveExamSelections(patientId,data){
       const response= await fetch(
           api(`/api/consultations/${patientId}/exams`),
           {
               method: 'POST',
               headers: {
                   'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    selected_exams: data.selected_exams,
                    evaluations: data.evaluations,
                }),
            }
        )

        if(!response.ok){
            throw new Error(
                `Submit Exam failed: ${response.status}`
            )
        }

        return response.json();
    },

    //Health Check
    /**
     * Backend health check
     * GET /api/health
    */
   async healthCheck(){
       const response= await fetch(api('/api/health'))

       if(!response.ok){
            throw new Error(
                `Health Check failed: ${response.status}`
            )
        }

        return response.json();
    }
};

export default consultationAPI;
