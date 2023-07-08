import os
import requests
from flask import Flask, jsonify, request
from modelinference import Inferance
from flask_cors import cross_origin
import logging
from flask_cors import CORS
from dotenv import load_dotenv
import subprocess
import json
import sys
load_dotenv()

app = Flask(__name__)
cors = CORS(app)

logging.basicConfig(filename='linelevel.log', level=logging.DEBUG)

def process_files(directory,headers,static_analysis,ai_analysis,path):
    # Get the repository contents
    response = requests.get(directory, headers=headers)

    # Check if the repository exists
    if response.status_code != 200:
        logging.error("response message: %s", response.json()['message'])
        return False

    logging.info("Processing files in %s", directory)
    # Loop over the repository files and count rows for C/C++ files
    for file_info in response.json():
        if file_info['type'] == 'file' and (file_info['name'].endswith('.c') or file_info['name'].endswith('.cpp')):
            file_url = file_info['download_url']
            file_response = requests.get(file_url, headers=headers)
            if file_response.status_code == 200:
                vul_lines = Inferance(file_response.text, file_info['name'])
                file_path = path + "/" + file_info['name']
                #vul_lines = [item for sublist in vul_lines for item in sublist]
                if vul_lines:
                    ai_analysis[file_path] = {"vul_lines":vul_lines,"file_url":file_url,"type_file":file_url.split('.')[-1]}
                    logging.info("Vulnerable lines in %s file: %s", file_info['name'], ai_analysis[file_path])
        
        if file_info['type'] == 'file' and (file_info['name'].endswith('.py')):
            file_url = file_info['download_url']
            file_response = requests.get(file_url, headers=headers)
            if file_response.status_code == 200:
                #vul_lines = Inferance(file_response.text, file_info['name'])
                result = subprocess.run(["./RemoVul",file_response.text], stdout=subprocess.PIPE)
                output = result.stdout.decode('utf-8')
                output = json.loads(output)
                # loop over all the keys in output and append the vul lines to the static_analysis dictionarys
                for key in output.keys():
                    if output[key]:
                        if key not in static_analysis:
                            static_analysis[key] = []
                        file_path = path + "/" + file_info['name']
                        static_analysis[key].append({"path": file_path, "vul_lines": output[key],"file_url":file_url,"type_file":file_url.split('.')[-1]})
                # check if the file_url c or cpp
            
                #all_vul_lines[file_info['name']] = {"vul_lines":[item for sublist in vul_lines for item in sublist],"file_url":file_url,"type_file":file_url.split('.')[-1]}
                #logging.info("Vulnerable lines in %s file: %s", file_info['name'], vul_lines)

        elif file_info['type'] == 'dir':
            subdir = os.path.join(directory, file_info['name'])
            process_files(subdir,headers,static_analysis,ai_analysis,path + "/" + file_info['name'])

    return True        



@app.route('/api/vul_lines', methods=['GET'])
@cross_origin()
def vul_lines():


    github_link = request.args.get('github_link')

    # Retrieve the access token from the environment variable
    access_token = os.environ['GITHUB_ACCESS_TOKEN']

    # Validate the GitHub link
    if not github_link.startswith('https://github.com/'):
        return jsonify({'error': 'Invalid GitHub link'}), 400

    # dict to store the name of the file and the vul lines
    #all_vul_lines = {}

    headers = {
        'Authorization': f'token {access_token}'  # Replace YOUR_TOKEN with your personal access token
    }

    # Process files recursively
    api_url = github_link.replace('https://github.com/', 'https://api.github.com/repos/') + '/contents'
    static_analysis = {}
    ai_analysis = {}

    if process_files(api_url, headers,static_analysis,ai_analysis,""):
        return jsonify({'static_analysis': static_analysis,"ai_analysis":ai_analysis}), 200
    else:
        return jsonify({'error': 'Invalid GitHub link'}), 400
    # if all_vul_lines id null or empty
    # if not static_analysis:
    #     return jsonify({'error': 'Invalid GitHub link'}), 400
    
    # if not static_analysis and not ai_analysis:
    #     return jsonify({'error': 'No cpp or c or py vul found'}), 404
        
    

@app.route('/api/custom_rule', methods=['GET'])
@cross_origin()
def custom_rule():

    # Get the custom rule string from the body of the request
    custom_rule = request.args.get('custom_rule')

    # Create a temporary file to store the custom rule in the tested directory

    # Create the file first
    file = open('./tested/custom_rule.yaml', 'w')

    # Write the custom rule to the file
    file.write(custom_rule)

    # Close the file
    file.close()

    github_link = request.args.get('github_link')

    # Retrieve the access token from the environment variable
    access_token = os.environ['GITHUB_ACCESS_TOKEN']



    # Validate the GitHub link
    if not github_link.startswith('https://github.com/'):
        return jsonify({'error': 'Invalid GitHub link'}), 400

    # dict to store the name of the file and the vul lines
    #all_vul_lines = {}

    headers = {
        'Authorization': f'token {access_token}'  # Replace YOUR_TOKEN with your personal access token
    }

    # Process files recursively
    api_url = github_link.replace('https://github.com/', 'https://api.github.com/repos/') + '/contents'
    static_analysis = {}
    ai_analysis = {}
    process_files(api_url, headers,static_analysis,ai_analysis,"")
    # if all_vul_lines id null or empty
    # if not static_analysis:
    #     return jsonify({'error': 'Invalid GitHub link'}), 400
    
    # if not static_analysis and not ai_analysis:
    #     return jsonify({'error': 'No cpp or c or py vul found'}), 404
        
    # Delete the temporary file
    os.remove('./tested/custom_rule.yaml')
    return jsonify({'static_analysis': static_analysis,"ai_analysis":ai_analysis}), 200


# api take file url and return content of file
@app.route('/api/file_content', methods=['GET'])
@cross_origin()
def file_content():
    file_url = request.args.get('file_url')
    file_response = requests.get(file_url)
    if file_response.status_code == 200:
        return jsonify({'file_content': file_response.text}), 200
    else:
        return jsonify({'error': 'File not found'}), 404
    



if __name__ == '__main__':
    app.run(host="0.0.0.0")

# http://localhost:5000/api/vul_lines?github_link=https://github.com/nixrajput/char-counter-cpp
# http://localhost:5000/api/vul_lines?github_link=https://github.com/naemazam/Hotel-Management-System
# http://localhost:5000/api/vul_lines?github_link=https://github.com/dev-aniketj/Learn-CPlusPlus
# http://localhost:5000/api/vul_lines?github_link=https://github.com/nragland37/cpp-projects4
# http://localhost:5000/api/vul_lines?github_link=https://github.com/conikeec/seeve
# http://localhost:5000/api/vul_lines?github_link=https://github.com/RemoVul/classical-tests
# http://localhost:5000/api/file_content?file_url=https://raw.githubusercontent.com/conikeec/seeve/master/CWE-119/src/test4.c