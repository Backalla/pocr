from flask import Flask, request, render_template
import os
import ocr

UPLOAD_FOLDER = "static/"
ALLOWED_EXTENSIONS = ["jpg","jpeg","png"]
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/',methods=['GET','POST'])
def main():
    '''
    Main server entrypoint
    This function will be executed for GET or POST request at localhost
    If it is a post request, ie form is submitted, then the form will be
    checked for files. If there is a file, it is saved with a name that
    is generated to serially save the uploaded images in uploads directory
    after checking for its extension.
    :return: Returns the rendered page depending on the situations.
    '''
    if request.method == "POST":
        # print request.method
        # print request.files
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
            result = ocr.do_ocr(os.path.join(app.config['UPLOAD_FOLDER'], filename),console=False)
            return render_template('results.html', result=result)


    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)