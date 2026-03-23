import { getCsrfToken } from '../utils/browser.js';

const POLL_MS = 1500;
let pollTimer = null;

const statusBar   = () => $('#test-status-bar');
const spinner     = () => $('#status-spinner');
const statusLabel = () => $('#status-label');
const summaryPills = () => $('#summary-pills');
const pillPassed  = () => $('#pill-passed');
const pillFailed  = () => $('#pill-failed');
const pillSkipped = () => $('#pill-skipped');
const resultsEl   = () => $('#results-container');
const rawSection  = () => $('#raw-log-section');
const rawLog      = () => $('#raw-log');
const runBtn      = () => $('#run-tests-btn');

const STATUS_ICON = {
    PASSED:  '<span class="inline-block w-2 h-2 rounded-full bg-success-main mr-1.5"></span>',
    FAILED:  '<span class="inline-block w-2 h-2 rounded-full bg-error-main mr-1.5"></span>',
    ERROR:   '<span class="inline-block w-2 h-2 rounded-full bg-error-main mr-1.5"></span>',
    SKIPPED: '<span class="inline-block w-2 h-2 rounded-full bg-background-contrast mr-1.5"></span>',
};

const badgeClass = status => {
    const map = {
        PASSED:  'bg-success-light text-success-dark',
        FAILED:  'bg-error-light text-error-dark',
        ERROR:   'bg-error-light text-error-dark',
        SKIPPED: 'bg-background-dust text-text-disabled',
    };
    return map[status] || 'bg-background-dust text-text-disabled';
};

const renderByClass = byClass =>
    Object.entries(byClass).map(([cls, tests]) => {
        const passCount = tests.filter(t => t.status === 'PASSED').length;
        const failCount = tests.filter(t => ['FAILED', 'ERROR'].includes(t.status)).length;
        const allPass   = failCount === 0;

        const rows = tests.map(t => `
            <tr class="border-b border-divider-light last:border-0">
                <td class="px-4 py-2.5">
                    <p class="text-xs font-mono text-text-body">${t.name}</p>
                    ${t.description ? `<p class="text-xs text-text-disabled mt-0.5 leading-snug">${t.description}</p>` : ''}
                </td>
                <td class="px-4 py-2.5 whitespace-nowrap">
                    <span class="inline-flex items-center text-xs px-2 py-0.5 rounded-full font-medium ${badgeClass(t.status)}">
                        ${STATUS_ICON[t.status] || ''}${t.status}
                    </span>
                </td>
                <td class="px-4 py-2.5 text-xs text-error-main font-mono max-w-xs">${t.detail || ''}</td>
            </tr>`).join('');

        return `
            <div class="bg-background-paper border border-divider-light rounded-xl overflow-hidden">
                <div class="flex items-center justify-between px-5 py-3 border-b border-divider-light">
                    <span class="text-sm font-semibold text-text-dark">${cls}</span>
                    <div class="flex items-center gap-2 text-xs">
                        <span class="text-success-dark font-medium">${passCount} passed</span>
                        ${failCount > 0 ? `<span class="text-error-dark font-medium">${failCount} failed</span>` : ''}
                    </div>
                </div>
                <table class="w-full text-sm">
                    <thead>
                        <tr class="text-xs text-text-disabled uppercase tracking-wider bg-background-dust">
                            <th class="text-left px-4 py-2">Test</th>
                            <th class="text-left px-4 py-2">Status</th>
                            <th class="text-left px-4 py-2">Detail</th>
                        </tr>
                    </thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>`;
    }).join('');

const applyState = state => {
    if (state.status === 'idle') return;

    statusBar().removeClass('hidden');

    if (state.status === 'running') {
        spinner().removeClass('hidden');
        statusLabel().text('Running tests…');
        summaryPills().addClass('hidden');
        runBtn().prop('disabled', true).addClass('opacity-50 cursor-not-allowed');
        return;
    }

    spinner().addClass('hidden');
    runBtn().prop('disabled', false).removeClass('opacity-50 cursor-not-allowed');

    if (state.status === 'error') {
        statusLabel().text(`Error: ${state.error}`);
        return;
    }

    if (state.status === 'done' && state.results) {
        const { summary, by_class, stdout } = state.results;
        const total = summary.total;

        statusLabel().text(`Done — ${total} test${total !== 1 ? 's' : ''} ran`);
        summaryPills().removeClass('hidden');
        pillPassed().text(`${summary.passed} passed`);

        if (summary.failed > 0) {
            pillFailed().text(`${summary.failed} failed`).removeClass('hidden');
        } else {
            pillFailed().addClass('hidden');
        }

        if (summary.skipped > 0) {
            pillSkipped().text(`${summary.skipped} skipped`).removeClass('hidden');
        } else {
            pillSkipped().addClass('hidden');
        }

        resultsEl().html(renderByClass(by_class));

        if (stdout) {
            rawLog().text(stdout);
            rawSection().removeClass('hidden');
        }
    }
};

const poll = () => {
    $.get('/api/tests/status/')
        .done(state => {
            applyState(state);
            if (state.status === 'running') {
                pollTimer = setTimeout(poll, POLL_MS);
            } else {
                pollTimer = null;
            }
        })
        .fail(() => { pollTimer = null; });
};

const startRun = () => {
    if (pollTimer) clearTimeout(pollTimer);
    resultsEl().empty();
    rawSection().addClass('hidden');
    rawLog().text('');

    $.post({ url: '/api/tests/run/', headers: { 'X-CSRFToken': getCsrfToken() } })
        .done(() => {
            statusBar().removeClass('hidden');
            spinner().removeClass('hidden');
            statusLabel().text('Starting…');
            summaryPills().addClass('hidden');
            poll();
        });
};

$(document).on('click', '#run-tests-btn', startRun);

$(document).on('click touchend', '#toggle-raw', e => {
    e.preventDefault();
    const isHidden = rawLog().hasClass('hidden');
    rawLog().toggleClass('hidden', !isHidden);
    const icon = $('#toggle-raw-icon');
    const label = $('#toggle-raw-label');
    if (isHidden) {
        // chevron-down: M19 9l-7 7-7-7
        icon.find('path').attr('d', 'M19 9l-7 7-7-7');
        label.text('Hide raw output');
    } else {
        // chevron-right: M9 5l7 7-7 7
        icon.find('path').attr('d', 'M9 5l7 7-7 7');
        label.text('Show raw output');
    }
});

// Resume any in-progress run on page load
$.get('/api/tests/status/').done(state => {
    if (state.status === 'running') {
        applyState(state);
        poll();
    } else if (state.status === 'done' || state.status === 'error') {
        applyState(state);
    }
});
