#!/usr/bin/env python3
"""
Gemma 4 Resume Parser — API Server
Flask API for parsing resumes using Google's Gemma 4 (via Google AI Studio).
Supports single file, bulk upload (up to 50), CSV import, and ATS formats.
"""

import os
import csv
import time
import tempfile
import uuid
import concurrent.futures
from io import StringIO
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from gemma_parser import (
    parse_resume, extract_text_from_file, is_google_configured, GOOGLE_MODEL,
)
from bulk_processor import init_bulk_processing, UPLOAD_DIR

# Active model (Gemma 4 via Google AI Studio).
ACTIVE_MODEL = GOOGLE_MODEL

app = Flask(__name__, static_folder='.')
CORS(app)

UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt', 'html', 'htm', 'jpg', 'jpeg', 'png', 'tiff', 'bmp'}
MAX_FILE_SIZE = 10 * 1024 * 1024       # 10MB per file
BULK_MAX_SIZE = 50 * 1024 * 1024       # 50MB total for bulk
BULK_MAX_FILES = 50

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = BULK_MAX_SIZE

# Initialize async bulk processing (SQLite + background thread)
bulk_store = init_bulk_processing(app)


# --- Security headers ---

@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Cache-Control'] = 'no-store'
    return response


# --- Helpers ---

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def parse_single_file(filepath, filename):
    """Parse a single file and return result dict."""
    start = time.time()
    try:
        resume_text = extract_text_from_file(filepath)
        if not resume_text or len(resume_text.strip()) < 50:
            return {'filename': filename, 'error': 'Could not extract text from file'}

        result = parse_resume(resume_text)
        elapsed = int((time.time() - start) * 1000)

        if 'error' in result:
            err = {'filename': filename, 'error': result['error'], 'processing_time_ms': elapsed}
            if 'finish_reason' in result:
                err['finish_reason'] = result['finish_reason']
            if 'raw_response' in result:
                err['raw_response'] = result['raw_response']
            return err

        return {'filename': filename, 'processing_time_ms': elapsed, 'result': result}
    except Exception as e:
        return {'filename': filename, 'error': str(e)}
    finally:
        try:
            os.remove(filepath)
        except OSError:
            pass


# --- Routes ---

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'provider': 'google',
        'model': ACTIVE_MODEL,
        'configured': is_google_configured(),
        'supported_formats': sorted(ALLOWED_EXTENSIONS),
        'max_bulk_files': BULK_MAX_FILES,
        'timestamp': time.time(),
    })


@app.route('/parse', methods=['POST'])
def parse():
    """Parse a single uploaded resume file."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided. Send a file with key "file".'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'Unsupported file type. Allowed: {", ".join(sorted(ALLOWED_EXTENSIONS))}'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result = parse_single_file(filepath, filename)

    if 'error' in result and 'result' not in result:
        return jsonify(result), 502

    return jsonify(result)


@app.route('/parse/text', methods=['POST'])
def parse_text():
    """Parse raw resume text (no file upload)."""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Send JSON with "text" field containing resume text.'}), 400

    resume_text = data['text']
    if len(resume_text.strip()) < 50:
        return jsonify({'error': 'Resume text is too short.'}), 400

    result = parse_resume(resume_text)

    if 'error' in result:
        return jsonify(result), 502

    return jsonify({'result': result})


@app.route('/parse/bulk', methods=['POST'])
def parse_bulk():
    """Parse multiple resume files (up to 50). Send files with key "files"."""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided. Send files with key "files".'}), 400

    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files selected'}), 400

    if len(files) > BULK_MAX_FILES:
        return jsonify({'error': f'Too many files. Maximum {BULK_MAX_FILES} files per request.'}), 400

    start = time.time()

    # Save all files first
    tasks = []
    for file in files:
        if file.filename == '' or not allowed_file(file.filename):
            continue
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"bulk_{int(time.time()*1000)}_{filename}")
        file.save(filepath)
        tasks.append((filepath, filename))

    if not tasks:
        return jsonify({'error': 'No valid files found in upload.'}), 400

    # Parse concurrently (max 5 at a time to respect API rate limits)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(parse_single_file, fp, fn): fn for fp, fn in tasks}
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    elapsed = int((time.time() - start) * 1000)
    successful = sum(1 for r in results if 'result' in r)

    return jsonify({
        'total_files': len(tasks),
        'successful': successful,
        'failed': len(tasks) - successful,
        'total_processing_time_ms': elapsed,
        'results': results,
    })


# --- Async Bulk Processing ---

@app.route('/jobs/bulk', methods=['POST'])
def submit_bulk_job():
    """Submit resumes for async bulk parsing. Returns job_id immediately."""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided. Send files with key "files".'}), 400

    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files selected'}), 400

    if len(files) > BULK_MAX_FILES:
        return jsonify({'error': f'Too many files. Maximum {BULK_MAX_FILES} per request.'}), 400

    # Save files to job-specific upload directory
    job_id = uuid.uuid4().hex
    job_upload_dir = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(job_upload_dir, exist_ok=True)

    files_info = []
    for file in files:
        if file.filename == '' or not allowed_file(file.filename):
            continue
        filename = secure_filename(file.filename)
        stored_name = f"{int(time.time()*1000)}_{filename}"
        stored_path = os.path.join(job_upload_dir, stored_name)
        file.save(stored_path)
        files_info.append((filename, stored_path))

    if not files_info:
        os.rmdir(job_upload_dir)
        return jsonify({'error': 'No valid files found in upload.'}), 400

    actual_job_id = bulk_store.create_job(files_info)

    return jsonify({
        'job_id': actual_job_id,
        'status': 'processing',
        'total_files': len(files_info),
        'message': f'Job submitted. Poll GET /jobs/{actual_job_id} for progress.',
    }), 202


@app.route('/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get current status and progress of a bulk parse job."""
    status = bulk_store.get_job_status(job_id)
    if not status:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(status)


@app.route('/jobs/<job_id>/results', methods=['GET'])
def get_job_results(job_id):
    """Get completed results for a bulk parse job."""
    status = bulk_store.get_job_status(job_id)
    if not status:
        return jsonify({'error': 'Job not found'}), 404

    if status['status'] != 'completed':
        return jsonify({
            'error': 'Job not yet completed',
            'status': status['status'],
            'progress_pct': status['progress_pct'],
        }), 409

    results = bulk_store.get_results(job_id)
    if not results:
        return jsonify({'error': 'Results file not found'}), 404

    return jsonify(results)


@app.route('/import/csv', methods=['POST'])
def import_csv():
    """
    Import candidate data from CSV file.
    CSV should have columns that map to resume fields.
    Each row is parsed as a candidate record.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided. Send a CSV with key "file".'}), 400

    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'File must be a .csv'}), 400

    try:
        content = file.read().decode('utf-8-sig')
        reader = csv.DictReader(StringIO(content))
        rows = list(reader)
    except Exception as e:
        return jsonify({'error': f'Failed to read CSV: {str(e)}'}), 400

    if not rows:
        return jsonify({'error': 'CSV is empty'}), 400

    if len(rows) > BULK_MAX_FILES:
        return jsonify({'error': f'Too many rows. Maximum {BULK_MAX_FILES} per import.'}), 400

    results = []
    for i, row in enumerate(rows):
        # Build resume text from CSV columns
        text_parts = []
        for key, value in row.items():
            if value and value.strip():
                text_parts.append(f"{key}: {value.strip()}")

        resume_text = '\n'.join(text_parts)

        if len(resume_text.strip()) < 30:
            results.append({'row': i + 1, 'error': 'Insufficient data in row'})
            continue

        result = parse_resume(resume_text)
        if 'error' in result:
            results.append({'row': i + 1, 'error': result['error']})
        else:
            results.append({'row': i + 1, 'result': result})

    successful = sum(1 for r in results if 'result' in r)

    return jsonify({
        'total_rows': len(rows),
        'successful': successful,
        'failed': len(rows) - successful,
        'results': results,
    })


# --- ATS Integration Endpoints ---

ATS_FIELD_MAPS = {
    'bullhorn': {
        'firstName': 'PersonalDetails.FirstName',
        'lastName': 'PersonalDetails.LastName',
        'email': 'PersonalDetails.EmailID',
        'phone': 'PersonalDetails.PhoneNumber',
        'address': 'PersonalDetails.Location',
        'occupation': 'OverallSummary.CurrentJobRole',
        'description': 'OverallSummary.Summary',
        'skillList': 'PrimarySkills',
        'educationDegree': 'ListOfEducation[0].Degree',
        'certifications': 'Certifications',
    },
    'dice': {
        'full_name': 'PersonalDetails.FullName',
        'email_address': 'PersonalDetails.EmailID',
        'phone_number': 'PersonalDetails.PhoneNumber',
        'location': 'PersonalDetails.Location',
        'current_title': 'OverallSummary.CurrentJobRole',
        'total_experience': 'OverallSummary.TotalExperience',
        'skills': 'ListOfSkills',
        'work_history': 'ListOfExperiences',
        'education': 'ListOfEducation',
    },
    'ceipal': {
        'CandidateName': 'PersonalDetails.FullName',
        'FirstName': 'PersonalDetails.FirstName',
        'LastName': 'PersonalDetails.LastName',
        'Email': 'PersonalDetails.EmailID',
        'Phone': 'PersonalDetails.PhoneNumber',
        'City': 'PersonalDetails.Location',
        'JobTitle': 'OverallSummary.CurrentJobRole',
        'Experience': 'OverallSummary.TotalExperience',
        'Skills': 'PrimarySkills',
        'Education': 'ListOfEducation',
        'Certifications': 'Certifications',
    },
}


def _resolve_field(data, path):
    """Resolve a dot-notation path like 'PersonalDetails.FirstName' from parsed result."""
    parts = path.split('.')
    current = data
    for part in parts:
        if '[' in part:
            key, idx = part.split('[')
            idx = int(idx.rstrip(']'))
            current = current.get(key, [])
            if isinstance(current, list) and len(current) > idx:
                current = current[idx]
            else:
                return None
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return None
        if current is None:
            return None
    return current


def _transform_to_ats(parsed_result, ats_name):
    """Transform parsed resume to ATS-specific format."""
    field_map = ATS_FIELD_MAPS.get(ats_name)
    if not field_map:
        return None

    ats_data = {}
    for ats_field, source_path in field_map.items():
        value = _resolve_field(parsed_result, source_path)
        ats_data[ats_field] = value

    return ats_data


@app.route('/parse/ats/<ats_name>', methods=['POST'])
def parse_ats(ats_name):
    """Parse resume and return in ATS-specific format (bullhorn, dice, ceipal)."""
    ats_name = ats_name.lower()
    if ats_name not in ATS_FIELD_MAPS:
        return jsonify({
            'error': f'Unsupported ATS: {ats_name}. Supported: {", ".join(ATS_FIELD_MAPS.keys())}'
        }), 400

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided.'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file.'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    start = time.time()
    try:
        resume_text = extract_text_from_file(filepath)
        if not resume_text or len(resume_text.strip()) < 50:
            return jsonify({'error': 'Could not extract text from file.'}), 400

        result = parse_resume(resume_text)
        elapsed = int((time.time() - start) * 1000)

        if 'error' in result:
            return jsonify(result), 502

        ats_data = _transform_to_ats(result, ats_name)

        return jsonify({
            'filename': filename,
            'ats': ats_name,
            'processing_time_ms': elapsed,
            'data': ats_data,
            'full_result': result,
        })
    finally:
        try:
            os.remove(filepath)
        except OSError:
            pass


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print(f"Starting Gemma 4 Resume Parser on port {port}")
    print(f"Model: {ACTIVE_MODEL}")
    print(f"Google AI Studio: {'configured' if is_google_configured() else 'NOT SET — set GOOGLE_API_KEY'}")
    print(f"Formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    print(f"ATS: {', '.join(ATS_FIELD_MAPS.keys())}")
    app.run(host='0.0.0.0', port=port, debug=True)
