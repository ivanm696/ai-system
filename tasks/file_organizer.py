if not source_dir.startswith('/safe/directory/'):
    return {'status': 'error', 'message': 'Invalid source directory'}