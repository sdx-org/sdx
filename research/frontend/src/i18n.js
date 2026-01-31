import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';
import en from './i18n/en.json';
import es from './i18n/es.json';
import pt from './i18n/pt.json'

i18n
// .use(LanguageDetector)
.use(initReactI18next).init({
  debug:true,
  resources: {
    en: { translation: en },
    es: { translation: es },
    pt: { translation: pt }
  },
  lng: 'en',
  fallbackLng: 'en',
  interpolation: { escapeValue: false },
});

export default i18n;
