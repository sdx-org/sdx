import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Dashboard from "./components/dashboard/Dashboard";
import PatientView from "./components/dashboard/PatientView";
import LanguageSelection from "./components/consultation/LanguageSelection";
import "./i18";
import Demographics from "./components/consultation/Demographics";
import Lifestyle from "./components/consultation/Lifestyle";
import Symptoms from "./components/consultation/Symptoms";
import Mental from "./components/consultation/Mental";
import MedicalReport from "./components/consultation/MedicalReport";
import Wearable from "./components/consultation/Wearable";
import Diagnosis from "./components/consultation/Diagnosis";
import Exam from "./components/consultation/Exam";
import Completion from "./components/consultation/Confirmation";
import { ConsultationProvider } from "./context/ConsultationContext";
import Confirmation from "./components/consultation/Confirmation";


export default function App() {
  return (
    <ConsultationProvider>
      <Router>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/patients/:id" element={<PatientView />} />
          <Route path="/language" element={<LanguageSelection />} />
          <Route path="/demographics" element={<Demographics />} />
          <Route path="/lifestyle" element={<Lifestyle />} />
          <Route path="/symptoms" element={<Symptoms />} />
          <Route path="/mental" element={<Mental />} />
          <Route path="/medical-reports" element={<MedicalReport />} />
          <Route path="/wearable-data" element={<Wearable />} />
          <Route path="/diagnosis" element={<Diagnosis />} />
          <Route path="/exams" element={<Exam />} />
          <Route path="/confirmation" element={<Confirmation />} />
          <Route path="*" element={<Dashboard />} />
        </Routes>
      </Router>
    </ConsultationProvider>
  );
}
