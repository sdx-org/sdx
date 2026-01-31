import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import 'bootstrap/dist/css/bootstrap.min.css';
import { ConsultationProvider } from './context/ConsultationContext';

import App from './App.jsx'
import './i18n.js';

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ConsultationProvider>
      <App />
    </ConsultationProvider>
  </StrictMode>,
)
