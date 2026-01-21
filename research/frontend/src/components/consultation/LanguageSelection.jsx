import React,{ useState } from 'react';
import {useNavigate} from 'react-router-dom';
import {Container, Row, Col, Button, Alert, Spinner} from 'react-bootstrap';
import {useConsultation, consultationActions} from '../../context/ConsultationContext';
import consultationAPI from '../../services/api';


export default function LanguageSelection(){
  const navigate= useNavigate();
  const {dispatch}= useConsultation();
  const [selectedLanguage, setSelectedLanguage]=useState(null);
  const [isLoading,setIsLoading]= useState(false);
  const [error, setError]=useState(null);
  const languages=[
    { code: 'en', name: 'English', flag: '🇬🇧' },
    { code: 'es', name: 'Spanish', flag: '🇪🇸' },
    { code: 'pt', name: 'Portuguese', flag: '🇧🇷' }
  ]

  const handleLanguageSelect= async (languageCode)=>{
    setSelectedLanguage(languageCode);
    setIsLoading(true);
    setError(null);
    try{
      const response= await consultationAPI.createPatient(languageCode);
      if(!response.patient_id){
        throw new Error('Failed to create patient: No patient ID returned');
      }
      dispatch(
        consultationActions.initConsultation(
          response.patient_id,
          languageCode,
          'demographics'
        )
      )
      localStorage.setItem('currentPatientId', response.patient_id);
      localStorage.setItem('currentLanguage', languageCode);
      navigate('/demographics')
    }catch(err){
      console.error('Error creating patient: err');
      setError(err.message || 'Failed to create patient. Please try again.');
      setSelectedLanguage(null);
    }finally{
      setIsLoading(false);
    }
  };

   return (
    <Container className="py-5">
      <Row className="justify-content-center">
        <Col lg={8} md={10} xs={12}>
          {/* Header */}
          <div className="text-center mb-5">
            <h1 className="display-5 fw-bold mb-2">Welcome</h1>
            <p className="lead text-muted">
              Select your preferred language to begin your consultation
            </p>
          </div>

          {/* Error Alert */}
          {error && (
            <Alert variant="danger" dismissible onClose={() => setError(null)}>
              <Alert.Heading>Error</Alert.Heading>
              <p>{error}</p>
            </Alert>
          )}

          {/* Language Selection Grid */}
          <div className="row g-3 mb-4">
            {languages.map((lang) => (
              <div key={lang.code} className="col-sm-6 col-lg-4">
                <Button
                  variant={selectedLanguage === lang.code ? 'primary' : 'outline-primary'}
                  className="w-100 py-3 d-flex align-items-center justify-content-center gap-2"
                  onClick={() => handleLanguageSelect(lang.code)}
                  disabled={isLoading}
                  size="lg"
                >
                  <span>{lang.flag}</span>
                  <span>{lang.name}</span>
                  {selectedLanguage === lang.code && isLoading && (
                    <Spinner
                      animation="border"
                      size="sm"
                      className="ms-2"
                      role="status"
                    >
                      <span className="visually-hidden">Loading...</span>
                    </Spinner>
                  )}
                </Button>
              </div>
            ))}
          </div>

          {/* Info Section */}
          <div className="alert alert-info">
            <h5>ℹ️ What happens next?</h5>
            <p className="mb-0">
              After selecting your language, you'll be guided through a series of health-related
              questions. Your responses will help us provide better insights.
            </p>
          </div>
        </Col>
      </Row>
    </Container>
  );
}
