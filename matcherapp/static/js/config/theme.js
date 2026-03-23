/**
 * Theme management — dark / light toggle, persisted in localStorage.
 */

const callbacks = [];

const applyTheme = (theme) => {
    $('body').attr('data-theme', theme);
    localStorage.setItem('theme', theme);
    callbacks.forEach(cb => cb(theme));
};

export const setTheme = () => {
    const system = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    applyTheme(localStorage.getItem('theme') || system);
};

export const toggleTheme = () => {
    applyTheme(localStorage.getItem('theme') === 'dark' ? 'light' : 'dark');
};

export const getTheme     = ()   => localStorage.getItem('theme');
export const onThemeChange = (cb) => callbacks.push(cb);
