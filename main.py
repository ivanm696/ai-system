if 'tasks/doc_generator.py' in sys.modules or 'tasks/doc_generator' in sys.modules:
    # Module already loaded, skip
    pass
else:
    try:
        spec = importlib.util.spec_from_file_location('tasks.doc_generator', 'tasks/doc_generator.py')
        tasks.doc_generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tasks.doc_generator)
    except ImportError as e:
        # Handle import error
        print(f'Error importing module: {e}')