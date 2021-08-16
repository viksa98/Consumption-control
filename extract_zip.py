import zipfile

def extract_zip(filepath):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(os.getcwd())