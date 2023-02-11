from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import subprocess
from werkzeug.utils import secure_filename
import image_processing
import os
import traceback
import logging
logging.basicConfig(filename='output.log', encoding='utf-8', level=logging.INFO)

flask = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
BACKGROUNDS_FOLDER = 'backgrounds'


@flask.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        instance_name = request.form['session']

        if 'images' not in request.files:
            return jsonify(message='No file uploaded', category="error", status=500)
        file = request.files['images']
        if file.filename == '':
            return jsonify(message='No file uploaded', category="error", status=500)

        filename = secure_filename(file.filename)
        try:
            file.save(os.path.join(UPLOAD_FOLDER, filename))
        except:
            return jsonify(message='The zip file could not be saved', category="error", status=500)

        try:
            subprocess.run(["unzip", "-j", UPLOAD_FOLDER + '/' + filename, '-d', UPLOAD_FOLDER + '/' + instance_name], check=True)

            for i, file in enumerate(os.listdir(UPLOAD_FOLDER + '/' + instance_name)):
                file_path = UPLOAD_FOLDER + '/' + instance_name + '/' + file
                if os.path.isdir(file_path):
                    subprocess.run(["rm", "-rf", file_path])
                    continue

                # check if hidden file
                if file[0] == '.':
                    os.remove(file_path)
                    continue

                if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    traceback.print_exc()
                    return jsonify(message='Zip contains other files than, png, jpg or jpeg', category="error", status=500)

        except Exception as e:
            return jsonify(message='An error occurred processing the zip file', category="error", status=500)

        subprocess.run(["rm", "-rf", UPLOAD_FOLDER + '/' + filename])

        subprocess.getoutput("mkdir -p " + PROCESSED_FOLDER + '/' + instance_name)

        for i, file in enumerate(os.listdir(UPLOAD_FOLDER + '/' + instance_name)):
            file_path = UPLOAD_FOLDER + '/' + instance_name + '/' + file
            try:
                result = image_processing.process_image(file_path, BACKGROUNDS_FOLDER + '/bck' + str(i % 20) + '.png')
                result.save(PROCESSED_FOLDER + '/' + instance_name + '/' + str(i) + '.png')
            except Exception as e:
                continue

        zip_file = PROCESSED_FOLDER + '/' + instance_name + '/' + instance_name + '.zip'
        zip_files = PROCESSED_FOLDER + '/' + instance_name + '/*'
        subprocess.getoutput("zip -j {ZIP_FILE} {ZIP_FILES}".format(ZIP_FILE=zip_file , ZIP_FILES=zip_files))

        return send_file(zip_file)

    return jsonify(message='Did not receive a POST request', category="error", status=500)


if __name__ == '__main__':
    flask.run(host='0.0.0.0', port=4000)
