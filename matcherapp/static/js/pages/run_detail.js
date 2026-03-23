/**
 * run_detail.js — live status polling, table sort, rescore.
 * Preview is handled by components/resume_preview_modal.html.
 *
 * Django variables via #run-data data-attributes:
 *   data-run-id   — MatchRun pk
 *   data-status   — initial status string
 */

import '../index.js';

const $runData      = $('#run-data');
const runId         = $runData.data('run-id');
const initialStatus = $runData.data('status');

if (!runId) throw new Error('run-data element missing');

// ── Re-score entire run ───────────────────────────────────────────────────
$('#rescore-btn').on('click', function () {
    const $btn = $(this).prop('disabled', true).text('Re-scoring…');
    $.post(`/api/matching/run/${runId}/rescore/`)
        .done(() => window.location.reload())
        .fail(() => $btn.prop('disabled', false).text('Re-score'));
});

// ── Table sort ────────────────────────────────────────────────────────────
let sortCol = 'rank';
let sortAsc = true;

const sortTable = () => {
    const sorted = $('#results-tbody tr').toArray().sort((a, b) => {
        const av = $(a).data(sortCol) ?? '';
        const bv = $(b).data(sortCol) ?? '';
        const an = parseFloat(av);
        const bn = parseFloat(bv);
        if (!isNaN(an) && !isNaN(bn)) return sortAsc ? an - bn : bn - an;
        return sortAsc
            ? String(av).localeCompare(String(bv))
            : String(bv).localeCompare(String(av));
    });
    $('#results-tbody').empty().append(sorted);
};

$(document).on('click', '.sort-header', function () {
    const col = $(this).data('col');
    sortAsc   = sortCol === col ? !sortAsc : col === 'rank';
    sortCol   = col;
    sortTable();
});

// ── Status polling ────────────────────────────────────────────────────────
const renderScoreBadge = (score) => {
    const cls = score >= 0.70 ? 'bg-success-main'
              : score >= 0.40 ? 'bg-warning-main'
              :                 'bg-error-main';
    return `<span class="${cls} text-white text-xs font-bold px-2 py-1 rounded">${score.toFixed(2)}</span>`;
};

const renderResults = (results) => {
    if (!results?.length) {
        $('#results-tbody').html('<tr><td colspan="11" class="text-center py-8 text-text-disabled">No results</td></tr>');
        return;
    }
    const rows = results.map(r => `
        <tr class="hover:bg-hover cursor-pointer border-b border-divider-light transition-colors"
            data-rank="${r.rank}" data-final_score="${r.final_score}" data-name="${r.name}"
            onclick="if(!event.target.closest('.preview-btn')) window.location='${r.detail_url}'">
            <td class="px-4 py-3 text-text-body font-medium">#${r.rank}</td>
            <td class="px-4 py-3 text-text-dark font-medium">${r.name}</td>
            <td class="px-4 py-3">${renderScoreBadge(r.final_score)}</td>
            <td class="px-4 py-3 text-text-body text-sm">${r.d1_skills.toFixed(2)}</td>
            <td class="px-4 py-3 text-text-body text-sm">${r.d2_seniority.toFixed(2)}</td>
            <td class="px-4 py-3 text-text-body text-sm">${r.d3_domain.toFixed(2)}</td>
            <td class="px-4 py-3 text-text-body text-sm">${r.d4_constraints.toFixed(2)}</td>
            <td class="px-4 py-3"><span class="text-xs px-2 py-1 rounded bg-background-dust text-text-body">${r.confidence}</span></td>
            <td class="px-4 py-3 text-text-body text-xs">${r.recommendation.replace('_', ' ')}</td>
            <td class="px-4 py-3 text-xs ${r.threat_level !== 'NONE' ? 'text-warning-main font-medium' : 'text-text-disabled'}">${r.threat_level}</td>
            <td class="px-4 py-3">
                <button class="preview-btn text-xs px-2.5 py-1 rounded border border-divider-light text-text-body hover:bg-hover transition-colors"
                    data-resume-id="${r.resume_id}" data-resume-name="${r.name}">Preview</button>
            </td>
        </tr>
    `).join('');

    $('#results-tbody').html(rows);
    $('#results-section').removeClass('hidden');
};

const pollStatus = () => {
    const timer = setInterval(() => {
        $.get(`/api/matching/run/${runId}/status/`, (data) => {
            $('#progress-bar').css('width', `${data.progress_pct}%`);
            $('#progress-text').text(`${data.processed} / ${data.total} resumes processed`);

            if (data.status === 'complete') {
                clearInterval(timer);
                renderResults(data.results);
                $('#status-badge')
                    .attr('class', 'px-3 py-1.5 rounded-lg text-xs font-medium bg-success-main text-white')
                    .text('Complete');
                $('#progress-section').addClass('hidden');
            } else if (data.status === 'failed') {
                clearInterval(timer);
                window.showSnackbar('error', 'Processing failed. Please try again.');
                $('#status-badge')
                    .attr('class', 'px-3 py-1.5 rounded-lg text-xs font-medium bg-error-main text-white')
                    .text('Failed');
            }
        });
    }, 2000);
};

if (['processing', 'pending'].includes(initialStatus)) pollStatus();
