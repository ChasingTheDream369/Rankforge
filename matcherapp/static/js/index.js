/**
 * index.js — shared entry point loaded on every page via base.html.
 *
 * Sets up: AJAX defaults, theme, theme-toggle button, snackbar.
 * Page-specific modules import this and add their own logic.
 */

import './config/ajax.js';
import { setTheme, toggleTheme } from './config/theme.js';

// Apply saved / system theme on every page load
setTheme();

// Theme toggle button (rendered by navbar component)
$(document).on('click', '#theme-icon-btn', toggleTheme);
