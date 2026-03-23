/**
 * Global jQuery AJAX configuration — CSRF, loader, snackbar error handling.
 */

import { getCsrfToken } from '../utils/browser.js';

let activeRequests = 0;

$.ajaxSetup({
    headers: {
        'X-CSRFToken': getCsrfToken(),
        'X-Requested-With': 'XMLHttpRequest',
    },
});

$(document)
    .ajaxSend(() => {
        activeRequests++;
        $('#line-loader').removeClass('hidden');
    })
    .ajaxComplete(() => {
        activeRequests = Math.max(0, activeRequests - 1);
        if (activeRequests === 0) $('#line-loader').addClass('hidden');
    })
    .ajaxError((_evt, jqxhr) => {
        if (jqxhr.status === 0 || jqxhr.readyState === 0) return;
        const msg  = jqxhr.responseJSON?.message || 'An error occurred. Please try again.';
        const type = [403, 409].includes(jqxhr.status) ? 'warning' : 'error';
        window.showSnackbar(type, msg);
    })
    .ajaxSuccess((_evt, jqxhr) => {
        const json = jqxhr.responseJSON;
        if (json?.message && json?.showSnackbar !== false) {
            window.showSnackbar(json.status || 'success', json.message);
        }
    });
