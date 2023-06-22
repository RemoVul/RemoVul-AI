import os
import requests
from flask import Flask, jsonify, request
from modelinference import Inferance
import logging

app = Flask(__name__)

logging.basicConfig(filename='linelevel.log', level=logging.DEBUG)

def process_files(directory,headers):
    # Get the repository contents
    response = requests.get(directory, headers=headers)

    # Check if the repository exists
    if response.status_code != 200:
        logging.error("response message: %s", response.json()['message'])
        return

    logging.info("Processing files in %s", directory)
    all_vul_lines={}
    # Loop over the repository files and count rows for C/C++ files
    for file_info in response.json():
        if file_info['type'] == 'file' and (file_info['name'].endswith('.c') or file_info['name'].endswith('.cpp')):
            file_url = file_info['download_url']
            file_response = requests.get(file_url, headers=headers)
            if file_response.status_code == 200:
                vul_lines = Inferance(file_response.text, file_info['name'])
                all_vul_lines[file_info['name']] = {"vul_lines":vul_lines,"file_url":file_url}
                logging.info("Vulnerable lines in %s file: %s", file_info['name'], vul_lines)

        elif file_info['type'] == 'dir':
            subdir = os.path.join(directory, file_info['name'])
            vul_lines=process_files(subdir,headers)
            all_vul_lines[file_info['name']]=vul_lines
    return all_vul_lines     

@app.route('/api/vul_lines', methods=['GET'])
def vul_lines():

    os.environ['GITHUB_ACCESS_TOKEN'] = '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'

    github_link = request.args.get('github_link')

    # Retrieve the access token from the environment variable
    access_token = os.environ['GITHUB_ACCESS_TOKEN']

    # Validate the GitHub link
    if not github_link.startswith('https://github.com/'):
        return jsonify({'error': 'Invalid GitHub link'})

    # dict to store the name of the file and the vul lines
    #all_vul_lines = {}

    headers = {
        'Authorization': f'token {access_token}'  # Replace YOUR_TOKEN with your personal access token
    }

    # Process files recursively
    api_url = github_link.replace('https://github.com/', 'https://api.github.com/repos/') + '/contents'
    all_vul_lines=process_files(api_url, headers)

    return jsonify({'vul_lines': all_vul_lines})


# api take file url and return content of file
@app.route('/api/file_content', methods=['GET'])
def file_content():
    file_url = request.args.get('file_url')
    file_response = requests.get(file_url)
    if file_response.status_code == 200:
        return jsonify({'file_content': file_response.text})
    else:
        return jsonify({'error': 'File not found'})
    

if __name__ == '__main__':
    app.run()

# http://localhost:5000/api/vul_lines?github_link=https://github.com/nixrajput/char-counter-cpp
# http://localhost:5000/api/vul_lines?github_link=https://github.com/naemazam/Hotel-Management-System
# http://localhost:5000/api/vul_lines?github_link=https://github.com/dev-aniketj/Learn-CPlusPlus
# http://localhost:5000/api/vul_lines?github_link=https://github.com/nragland37/cpp-projects4
# http://localhost:5000/api/vul_lines?github_link=https://github.com/conikeec/seeve
# http://localhost:5000/api/file_content?file_url=https://raw.githubusercontent.com/conikeec/seeve/master/CWE-119/src/test4.c