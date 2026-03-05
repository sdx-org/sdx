import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react({
      babel: {
        plugins: [['babel-plugin-react-compiler']],
      },
    }),
  ],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) return;

          if (
            id.includes('/react/') ||
            id.includes('/react-dom/')
          ) {
            return 'framework';
          }

          if (
            id.includes('/i18next/') ||
            id.includes('/react-i18next/')
          ) {
            return 'i18n';
          }

          if (
            id.includes('/bootstrap/') ||
            id.includes('/react-bootstrap/')
          ) {
            return 'ui';
          }

          if (
            id.includes('/react-hook-form/') ||
            id.includes('/react-paginate/') ||
            id.includes('/react-icons/')
          ) {
            return 'forms-and-utils';
          }

          return 'vendor';
        },
      },
    },
  },
})
