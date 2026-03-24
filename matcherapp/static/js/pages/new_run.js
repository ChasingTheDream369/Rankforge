/**
 * new_run.js — drag-drop file upload + run submission + weight profile selector.
 * Django variables: none (pure UI, API call returns redirect_url).
 */

import '../index.js';

const $dropZone  = $('#drop-zone');
const $fileInput = $('#resume-files');
const $fileList  = $('#file-list');
let   fileStore  = new DataTransfer();

// ── File list renderer ────────────────────────────────────────────────────
const renderFileList = () => {
    const files = Array.from(fileStore.files);
    $fileList.empty();

    if (files.length === 0) {
        $fileList.append('<p class="text-text-disabled text-sm">No files selected</p>');
        return;
    }

    const items = files.map((f, i) => `
        <div class="flex items-center justify-between py-1.5 px-3 rounded bg-background-dust text-sm">
            <span class="text-text-body truncate max-w-xs">${f.name}</span>
            <button class="remove-file text-error-main hover:text-error-dark ml-2 text-xs" data-index="${i}">✕</button>
        </div>
    `).join('');

    $fileList.append(items);
    $fileList.append(`<p class="text-text-disabled text-xs mt-1">${files.length} file(s) selected</p>`);
};

// ── Add files to store ────────────────────────────────────────────────────
const addFiles = (incoming) => {
    Array.from(incoming).forEach(f => fileStore.items.add(f));
    renderFileList();
};

// ── Drag-drop handlers ────────────────────────────────────────────────────
$dropZone
    .on('dragover',  (e) => { e.preventDefault(); $dropZone.addClass('border-primary-main'); })
    .on('dragleave', ()  => { $dropZone.removeClass('border-primary-main'); })
    .on('drop',      (e) => {
        e.preventDefault();
        $dropZone.removeClass('border-primary-main');
        addFiles(e.originalEvent.dataTransfer.files);
    })
    .on('click', () => $fileInput.trigger('click'));

$fileInput.on('change', function () { addFiles(this.files); });

// ── Weight profile presets (match ROLE_WEIGHTS in scorer.py) ─────────────
const PROFILES = {
    junior:    { d1: 50, d2: 25, d3: 15, d4: 10, hint: 'Skills-heavy — emphasizes technical skill match over experience level.' },
    mid:       { d1: 40, d2: 35, d3: 15, d4: 10, hint: 'Balanced — standard weight distribution for mid-level roles.' },
    senior:    { d1: 35, d2: 45, d3: 12, d4: 8,  hint: 'Experience-heavy — values seniority and depth of experience.' },
    staff:     { d1: 30, d2: 50, d3: 12, d4: 8,  hint: 'Leadership-focused — strongly favors seniority and architecture skills.' },
    executive: { d1: 25, d2: 55, d3: 12, d4: 8,  hint: 'Strategic — heavily weights leadership and executive experience.' },
    custom:    { d1: 40, d2: 35, d3: 15, d4: 10, hint: 'Set your own dimension weights below.' },
};

const PILL_ACTIVE   = 'border-primary-main text-primary-main';
const PILL_INACTIVE = 'border-divider-light text-text-body';
let activeProfile = 'auto';

function setActivePill(profile) {
    activeProfile = profile;
    $('.profile-pill').removeClass(PILL_ACTIVE).addClass(PILL_INACTIVE).removeClass('selected');
    $(`.profile-pill[data-profile="${profile}"]`).removeClass(PILL_INACTIVE).addClass(PILL_ACTIVE).addClass('selected');
}

function updateBars() {
    ['d1', 'd2', 'd3', 'd4'].forEach(k => {
        const v = Math.max(0, Math.min(100, parseInt($(`#${k}-pct`).val()) || 0));
        $(`#${k}-bar`).css('width', v + '%');
    });
}

function fillWeights(p) {
    $('#d1-pct').val(p.d1);
    $('#d2-pct').val(p.d2);
    $('#d3-pct').val(p.d3);
    $('#d4-pct').val(p.d4);
    updateBars();
}

// ── Profile pill click handler ───────────────────────────────────────────
$('#weight-profiles').on('click', '.profile-pill', function () {
    const profile = $(this).data('profile');
    setActivePill(profile);

    if (profile === 'auto') {
        $('#weights-panel').addClass('hidden');
        $('#profile-hint').text('Reads seniority from the JD and applies matching dimension bias automatically.');
        $('.dim-input').prop('disabled', false);
    } else {
        const p = PROFILES[profile];
        fillWeights(p);
        $('#weights-panel').removeClass('hidden');
        $('#profile-hint').text(p.hint);
        $('.dim-input').prop('disabled', false);
    }
});

// ── Live bar + auto-switch to Custom on manual edits ─────────────────────
$('#weights-panel').on('input', '.dim-input', function () {
    updateBars();
    if (activeProfile !== 'custom' && activeProfile !== 'auto') {
        const p = PROFILES[activeProfile];
        if (p && (
            parseInt($('#d1-pct').val()) !== p.d1 ||
            parseInt($('#d2-pct').val()) !== p.d2 ||
            parseInt($('#d3-pct').val()) !== p.d3 ||
            parseInt($('#d4-pct').val()) !== p.d4
        )) {
            setActivePill('custom');
            $('#profile-hint').text('Custom weights — modified from preset.');
        }
    }
});

// ── Remove individual file ────────────────────────────────────────────────
$fileList.on('click', '.remove-file', function () {
    const idx = parseInt($(this).data('index'), 10);
    const next = new DataTransfer();
    Array.from(fileStore.files)
        .filter((_, i) => i !== idx)
        .forEach(f => next.items.add(f));
    fileStore = next;
    renderFileList();
});

// ── Form submit ───────────────────────────────────────────────────────────
$('#run-form').on('submit', function (e) {
    e.preventDefault();

    const jdText  = $('#jd-text').val().trim();
    const jdTitle = $('#jd-title').val().trim() || 'Untitled Job';

    if (!jdText) {
        window.showSnackbar('warning', 'Please enter a job description.');
        return;
    }
    if (fileStore.files.length === 0) {
        window.showSnackbar('warning', 'Please add at least one resume.');
        return;
    }

    const $btn = $('#submit-btn').prop('disabled', true).text('Starting…');

    const formData = new FormData();
    formData.append('jd_text', jdText);
    formData.append('jd_title', jdTitle);
    formData.append('scoring_mode', $('#scoring-mode').val());
    formData.append('weight_profile', activeProfile);

    if (activeProfile !== 'auto') {
        formData.append('custom_dim_weights', '1');
        ['d1', 'd2', 'd3', 'd4'].forEach((k) => {
            const v = $(`#${k}-pct`).val();
            formData.append(`${k}_pct`, v === '' || v == null ? '0' : String(v));
        });
    }

    Array.from(fileStore.files).forEach(f => formData.append('resumes', f));

    $.ajax({
        url:         '/api/matching/start/',
        method:      'POST',
        data:        formData,
        processData: false,
        contentType: false,
        success: ({ redirect_url }) => { window.location.href = redirect_url; },
        error:   ()                 => { $btn.prop('disabled', false).text('Start Matching Run'); },
    });
});

renderFileList();
