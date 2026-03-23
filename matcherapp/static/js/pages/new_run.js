/**
 * new_run.js — drag-drop file upload + run submission.
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
