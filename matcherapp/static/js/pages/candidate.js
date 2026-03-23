/**
 * candidate.js — collapsible sections + single-resume rescore.
 *
 * Django variable via #candidate-data:
 *   data-result-id — MatchResult pk
 */

import '../index.js';

// ── Collapsible detail sections ───────────────────────────────────────────
$(document).on('click', '.collapsible-toggle', function () {
    $(`#${$(this).data('target')}`).slideToggle(200);
    $(this).find('.toggle-icon').toggleClass('rotate-180');
});

// ── Re-score single resume ────────────────────────────────────────────────
$('#rescore-single-btn').on('click', function () {
    const resultId = $(this).data('result-id');
    const $btn = $(this).prop('disabled', true).text('Re-scoring…');
    $.post(`/api/matching/result/${resultId}/rescore/`)
        .done(() => window.location.reload())
        .fail(() => $btn.prop('disabled', false).text('Re-score'));
});
