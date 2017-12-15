from flask import Flask, request, render_template
import os

UPLOAD_FOLDER = "./uploads/"
ALLOWED_EXTENSIONS = ["jpg","jpeg"]
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/',methods=['GET','POST'])
def main():
    # print request.method
    if request.method == "POST":
        print request.method
        print request.files
        if 'file' not in request.files:
            return render_template('index.html',error="File not in request")
        file = request.files['file']
        if file.filename == '':
            print 'No selected file'
        if file:
            file_extension =  file.filename.split(".")[-1]
            if file_extension not in ALLOWED_EXTENSIONS:
                return render_template('index.html',error="File extension not supported")
            filenames_list = os.listdir(UPLOAD_FOLDER)
            if filenames_list:
                filename = "{}.{}".format(str(max([int(a.split(".")[0]) for a in filenames_list])+1),file_extension)
            else:
                filename = "1."+file_extension

            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    return render_template('index.html')