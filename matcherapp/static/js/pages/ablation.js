import { getCsrfToken } from '../utils/browser.js';

const POLL_MS = 2000;
let pollTimer = null;

const statusBar   = () => $('#ablation-status-bar');
const spinner     = () => $('#ablation-spinner');
const statusLabel = () => $('#ablation-status-label');
const resultsEl   = () => $('#ablation-results');
const tbody       = () => $('#ablation-tbody');
const jdLabel     = () => $('#ablation-jd-label');
const noLabels    = () => $('#ablation-no-labels');
const runBtn      = () => $('#run-ablation-btn');

// Metric keys to display in table order
const METRICS = [
    { key: 'ndcg@3',   label: 'nDCG@3'   },
    { key: 'ndcg@5',   label: 'nDCG@5'   },
    { key: 'ndcg@10',  label: 'nDCG@10'  },
    { key: 'mrr',      label: 'MRR'      },
    { key: 'p@3',      label: 'P@3'      },
    { key: 'p@5',      label: 'P@5'      },
    { key: 'spearman', label: 'Spearman' },
];

const fmt = v => (v == null ? '—' : v.toFixed(3));

// Find max value per metric column across all rows (for highlighting best)
const findBests = rows => {
    const bests = {};
    METRICS.forEach(({ key }) => {
        const vals = rows
            .filter(r => !r.error && r.metrics && r.metrics[key] != null)
            .map(r => r.metrics[key]);
        bests[key] = vals.length ? Math.max(...vals) : null;
    });
    // top-1 accuracy: best rank == 0 means top resume was #1
    const top1Vals = rows
        .filter(r => !r.error && r.top5 && r.top5.length > 0)
        .map(r => r.scores ? Object.values(r.scores)[0] : null)
        .filter(v => v != null);
    bests['top1'] = top1Vals.length ? Math.max(...top1Vals) : null;
    return bests;
};

const cellClass = (val, best) => {
    if (val == null || best == null) return 'px-4 py-3 text-center text-text-disabled';
    if (Math.abs(val - best) < 0.001) return 'px-4 py-3 text-center font-semibold text-success-dark';
    return 'px-4 py-3 text-center text-text-body';
};

const renderRows = (rows, hasLabels) => {
    const bests = findBests(rows);

    return rows.map((row, i) => {
        if (row.error) {
            return `
                <tr class="border-b border-divider-light">
                    <td class="px-4 py-3 text-sm font-medium text-text-dark">${row.approach}</td>
                    <td colspan="${METRICS.length + 1}" class="px-4 py-3 text-xs text-error-main">${row.error}</td>
                </tr>`;
        }

        const metricCells = METRICS.map(({ key }) => {
            const val = row.metrics[key];
            return `<td class="${cellClass(val, bests[key])}">${fmt(val)}</td>`;
        }).join('');

        const top1Score = row.top5 && row.top5.length > 0 && row.scores
            ? Object.values(row.scores)[0]
            : null;

        return `
            <tr class="border-b border-divider-light hover:bg-background-dust transition-colors">
                <td class="px-4 py-3 text-sm font-medium text-text-dark whitespace-nowrap">${row.approach}</td>
                ${hasLabels ? metricCells : `<td colspan="${METRICS.length}" class="px-4 py-3 text-xs text-text-disabled text-center">No labels</td>`}
                <td class="${cellClass(top1Score, bests['top1'])}">${fmt(top1Score)}</td>
            </tr>`;
    }).join('');
};

const applyState = state => {
    if (state.status === 'idle') return;

    statusBar().removeClass('hidden');

    if (state.status === 'running') {
        spinner().removeClass('hidden');
        runBtn().prop('disabled', true).addClass('opacity-50 cursor-not-allowed');
        const elapsed = typeof state.elapsed_sec === 'number' ? `${state.elapsed_sec}s` : '';
        const p = state.progress;
        if (p && p.name) {
            const step = `Step ${p.step}/${p.total}`;
            const slowHint = p.step === p.total
                ? ' — full LLM pipeline (many API calls); this step often takes several minutes.'
                : '';
            statusLabel().text(`${step}: ${p.name}${slowHint}${elapsed ? ` · ${elapsed} elapsed` : ''}`);
        } else {
            statusLabel().text(`Starting ablation…${elapsed ? ` (${elapsed} elapsed)` : ''}`);
        }
        return;
    }

    spinner().addClass('hidden');
    runBtn().prop('disabled', false).removeClass('opacity-50 cursor-not-allowed');

    if (state.status === 'error') {
        statusLabel().text(`Error: ${state.error}`);
        return;
    }

    if (state.status === 'done' && state.results) {
        const { rows, jd_id, has_labels } = state.results;
        statusLabel().text(`Done — ${rows.length} approach${rows.length !== 1 ? 'es' : ''} evaluated`);

        if (jd_id) jdLabel().text(`JD: ${jd_id}`);

        tbody().html(renderRows(rows, has_labels));
        resultsEl().removeClass('hidden');

        if (!has_labels) noLabels().removeClass('hidden');
        else noLabels().addClass('hidden');
    }
};

const poll = () => {
    $.get('/api/ablation/status/')
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
    resultsEl().addClass('hidden');
    noLabels().addClass('hidden');
    tbody().empty();

    $.post({ url: '/api/ablation/run/', headers: { 'X-CSRFToken': getCsrfToken() } })
        .done(() => {
            statusBar().removeClass('hidden');
            spinner().removeClass('hidden');
            statusLabel().text('Starting…');
            poll();
        });
};

$(document).on('click', '#run-ablation-btn', startRun);

// Resume in-progress run on page load
$.get('/api/ablation/status/').done(state => {
    if (state.status === 'running') {
        applyState(state);
        poll();
    } else if (state.status === 'done' || state.status === 'error') {
        applyState(state);
    }
});
