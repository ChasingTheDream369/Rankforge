/**
 * Browser utilities — jQuery-based, no raw document access.
 */

export const getCsrfToken = () =>
    $('input[name=csrfmiddlewaretoken]').val() ||
    document.cookie
        .split(';')
        .map(c => c.trim())
        .find(c => c.startsWith('csrftoken='))
        ?.split('=')[1] || '';

export const showSnackbar = (type, message) => {
    const colours = {
        success: 'bg-success-main',
        error:   'bg-error-main',
        warning: 'bg-warning-main',
        info:    'bg-info-main',
    };
    const $item = $('<div>')
        .addClass(`snackbar-item ${colours[type] || colours.info}`)
        .text(message);
    $('#snackbar-container').append($item);
    setTimeout(() => $item.fadeOut(300, () => $item.remove()), 4000);
};

// Make globally accessible so inline template error handlers can call it
window.showSnackbar = showSnackbar;
