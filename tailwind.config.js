/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './matcherapp/templates/**/*.html',
    './matcherapp/static/js/**/*.js',
  ],
  theme: {
    extend: {
      fontFamily: {
        inter: ['Inter', 'sans-serif'],
      },
      colors: {
        background: {
          default: 'var(--color-background-default)',
          dust: 'var(--color-background-dust)',
          paper: 'var(--color-background-paper)',
          contrast: 'var(--color-background-contrast)',
        },
        text: {
          light: 'var(--color-text-light)',
          dark: 'var(--color-text-dark)',
          body: 'var(--color-text-body)',
          disabled: 'var(--color-text-disabled)',
          contrast: 'var(--color-text-contrast)',
        },
        primary: {
          main: 'var(--color-primary-main)',
          light: 'var(--color-primary-light)',
          dark: 'var(--color-primary-dark)',
          contrast: 'var(--color-primary-contrast)',
        },
        secondary: {
          main: 'var(--color-secondary-main)',
          light: 'var(--color-secondary-light)',
          dark: 'var(--color-secondary-dark)',
          contrast: 'var(--color-secondary-contrast)',
        },
        error: {
          main: 'var(--color-error-main)',
          light: 'var(--color-error-light)',
          dark: 'var(--color-error-dark)',
        },
        warning: {
          main: 'var(--color-warning-main)',
          light: 'var(--color-warning-light)',
          dark: 'var(--color-warning-dark)',
        },
        success: {
          main: 'var(--color-success-main)',
          light: 'var(--color-success-light)',
          dark: 'var(--color-success-dark)',
        },
        info: {
          main: 'var(--color-info-main)',
          light: 'var(--color-info-light)',
          dark: 'var(--color-info-dark)',
        },
        divider: {
          main: 'var(--color-divider-main)',
          light: 'var(--color-divider-light)',
          body: 'var(--color-divider-body)',
        },
        hover: 'var(--color-hover)',
      },
    },
  },
  plugins: [],
}
