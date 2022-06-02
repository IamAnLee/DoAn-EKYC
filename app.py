from flask import Flask, render_template ,request, send_from_directory,Response,redirect,url_for
from flask import jsonify
import json
import re
import os
import shutil
import scipy.misc
import warnings
import sys
import compare_image
import time
import detect_face
from werkzeug.utils import secure_filename
import cv2
import datetime
import requests
import fitz
import webbrowser
import matplotlib
matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input



# =================facematching=============
def extract_face(filename, required_size=(224, 224)):
    pixels = plt.imread(filename)
    detector = MTCNN()
    results = detector.detect_faces(pixels)

    if results:
        x1, y1, width, height = results[0]['box']
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        print("x1,width,y1,height", x1, width, y1, height)
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        return face_array
    return None




# Trích xuất khuôn mặt và tính toán số lần nhúng khuôn mặt cho danh sách các tệp ảnh

def get_embeddings(filenames):

    faces = [ extract_face(f)  for f in filenames]

    if  faces[0] is None or faces[1] is None:

        return None
    else:

        samples = asarray(faces, 'float32')
        samples = preprocess_input(samples, version=2)
        model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        return model.predict(samples)


# xác định xem hai khuôn mặt có khớp nhau không
# lấy khoảng cách cosine giữa chúng


def match_embeddings(known_embedding, candidate_embedding, thresh=0.5):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    if score <= thresh:
        return True
    else:
        return False


app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = r"D:\\DL&WebNC\\ekyc2\\EKYC-Verification1"

#=======================ROUTES=================================================================


#-------------Home Page---------------------
@app.route('/homepage')
def index():
    return render_template('index.html')

#-----------Steps Routes-------------------
# @app.route('/ekyc')
# def home():
#     return render_template('home.html')
@app.route('/ekyc-stp1')
def stp1():
    return render_template('stp1.html')

@app.route('/ekyc-stp2')
def stp2():
    return render_template('stp2.html')

@app.route('/ekyc-stp3')
def stp3():
    f=open('result.txt','r')
    x=f.read()
    if x=='0':
        return render_template('stp3.html',result=False)
    else:
        return render_template('stp3.html',result=True)
    
    
#------------Make New Dir DatTime.Now ----------------------------------   
@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():
    dirname=''
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            image.save(os.path.join(
                app.config["IMAGE_UPLOADS"], image.filename))
            dirname=str(datetime.datetime.now())
            dirname=dirname.replace(':','')
            dirname=dirname.replace('-','')
            dirname=dirname.replace(' ','')
            newpath = r'D:\\DL&WebNC\\ekyc2\\EKYC-Verification1\\images'+str(dirname) +'\\dataset'

            if not os.path.exists(newpath):
                os.makedirs(newpath)

            formImg(image.filename,dirname)
    return render_template('stp2.html',dirname=dirname)



#-------------- Get Images from PDF---------------------------------------
def formImg(fileName,dirname):
    print(fileName)
    doc = fitz.open(fileName)
    counter = 0
    newpath = r'D:\\DL&WebNC\\ekyc2\\EKYC-Verification1\\images' + str(dirname) + '\\dataset'
    source_folder = r"D:\\DL&WebNC\\ekyc2\\EKYC-Verification1\\" + fileName

    oldpath = source_folder
    newpath2 = r'D:\\DL&WebNC\\ekyc2\\EKYC-Verification1\\\\face.jpg'
    os.rename(oldpath, newpath2)
    destination_folder = newpath
    shutil.move(newpath2, newpath)
    for i in range(len(doc)):
        for img in doc.getPageImageList(i):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n < 5:       # this is GRAY or RGB
                pix.writePNG("p%s.png" % (i))
                counter += 1
            else:               # CMYK: convert to RGB first
                pix1 = fitz.Pixmap(fitz.csRGB, pix)
                pix1.writePNG("p%s.png" % (i))
                pix1 = None
                counter += 1
            pix = None
    count= 0
    for i in range(0, counter):

        imagePath = r"D:\\DL&WebNC\\ekyc2\\EKYC-Verification1\\p" + \
            str(i)+".png"

        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faceCascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
        )
        
        print("[INFO] Found {0} Faces.".format(len(faces)))
        padding = 30
        for (x, y, w, h) in faces:
            image = cv2.rectangle(image, (x-padding, y-padding),
                                  (x + w+padding, y + h+padding), (0, 255, 0), 2)
            roi_color = image[y-30:y + h+30, x-30:x + w+30]
            print("[INFO] Object found. Saving locally.")
            if(count==0):
                cv2.imwrite(f'D:\\DL&WebNC\\ekyc2\\EKYC-Verification1\\images{dirname}\\dataset\\face.jpg', roi_color)
                count=count+1
        status = cv2.imwrite('faces_detected.jpg', image)
    return ''



#-----------Live Video Image Picker ----------------------------------------------------
@app.route('/opencamera',methods=['GET','POST'])    
def camera():
    dirname=request.form['dirname']
    vid = cv2.VideoCapture(0) 
    i=0
    while( i < 200): 
        ret, frame = vid.read() 
        cv2.imshow('frame', frame) 
        i=i+1
        if i==101:
            cv2.imwrite(f'D:\\DL&WebNC\\ekyc2\\EKYC-Verification1\\images{dirname}\\dataset\\trial{str(i)}.jpeg',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    compare(dirname)
    return redirect(url_for('stp3'))
    
#------------- Compare Images ------------------------
def compare(dirname):
    url="http://localhost:8000/api/v1/compare_faces"
    target=f'D:\\DL&WebNC\\ekyc2\\EKYC-Verification1\\images{dirname}\\dataset\\trial101.jpeg'
    directory=f'D:\\DL&WebNC\\ekyc2\\EKYC-Verification1\\images{dirname}\\dataset\\face.jpg'
    files = {"target": open(target,'rb'),"faces":open(directory,'rb')}
    x=requests.post(url,files= files) 
    return ''


@app.route('/api/v1/compare_faces', methods=['POST'])
def compare_faces():

    target = request.files['target']
    faces =  request.files.getlist("faces")
    target_filename=secure_filename(target.filename)
    response=[]
    for face in faces:
        distance,result = compare_image.main(target,face)    
    print(result)
    f=open('result.txt','w+')
    if result:
        f.write('1')
    else:

        f.write('0')
    return ''
    

@app.route('/api/v1/detect_faces', methods=['POST'])
def detect_faces():
    faces =  request.files.getlist("faces")
    # target_filename=secure_filename(target.filename)
    response=[]
    for face in faces:
        start = time.time()
        _,result = detect_face.get_coordinates(face)
        end=time.time()
        json_contect={
                'coordinates':result,
                'time_taken':round(end-start,3),
                'image_name':secure_filename(face.filename)
            }
        response.append(json_contect)
    python2json = json.dumps(response)
    return app.response_class(python2json, content_type='application/json')




# được hiển thị dưới dạng trang đầu của ứng dụng
# @app.route('/')
# def load_front_page():
#     return render_template('index.html')


# được gọi khi  tải hình ảnh lên
@app.route('/index', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        first_file = request.files['original_image']

        print(first_file)
        print(type(first_file))
        second_file = request.files['to_compare_image']
        embeddings_array = [first_file, second_file]
        embeddings = get_embeddings(embeddings_array)
        if embeddings  is None:
            message = "Không tìm thấy khuôn mặt ! hãy thử lại."
            return render_template('index.html', message=message)
        else:
            if match_embeddings(embeddings[0], embeddings[1]):
                message = "Hai ảnh khuôn mặt  cùng một người."


                return render_template('index.html', message=message)
            else:
                message="Hai ảnh khuôn mặt không cùng một người."

                return render_template('index.html', message=message)


@app.errorhandler(500)
def internal_error(error):
    return render_template('index.html', message="Please upload valid photos having a face")
@app.errorhandler(404)
def not_found(error):
    return render_template('index.html', message="")

#================RUN===============================================================================    
  

if __name__ == "__main__":
    app.run(debug=True,port=8000)