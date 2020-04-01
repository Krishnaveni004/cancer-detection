from django.shortcuts import render
from django.views import View
from django.http import HttpResponse
import tensorflow as tf
from .forms import FileForm
from pydicom import dcmread
import keras, dicom, os
import numpy as np
from pydicom.filebase import DicomBytesIO
from PIL import Image
from django.core.files.storage import FileSystemStorage


# Create your views here.
class DiagnoseDICOMImage(View):
    
    def readScan(self, dirName):
        result=[]
        files=[]
        for dirName, _, fileList in os.walk(dirName):
          for filename in fileList:
            if ".dcm" in filename.lower():
                files.append(os.path.join(dirName,filename))

        total_files = len(files)    
        RefDs = dicom.read_file(files[0])
        ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(files))
        
        ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
        locations = list()
        for i in range(len(files)):
            location = dicom.read_file(files[i])
            locations.append(int(location.SliceLocation))

        locations = np.array(locations)
        minimum = locations.min()

        for index in range(total_files):
            ds = dicom.read_file(files[index])
            ArrayDicom[:, :, (int(ds.SliceLocation)-int(minimum))] = ds.pixel_array
            
        result.append(ArrayDicom[:,:,:250].reshape(512,512,250,1))
        return np.stack(result).astype(float)

    def post(self, request):
        print(request.FILES.getlist('dicom_images'))
        files = request.FILES.getlist('dicom_images')

        filenames=[]
        fs = FileSystemStorage()
        for immemoryfile in files:
            filenames.append(fs.save(immemoryfile.name, immemoryfile))
        loaded_dicom_image = self.readScan('media')
        dicom_model = keras.models.load_model("./Generic_spie.save")

        with tf.Session() as sess:
            print('global_variables_initializer...')
            sess.run(tf.global_variables_initializer())
            prediction = dicom_model.predict(loaded_dicom_image, batch_size=1)

        predicted_probability = max(prediction[0])
        predicted_class = list(prediction[0]).index(predicted_probability)

        return render(request, 'Results.html', context={'predicted_probability': predicted_probability,
                                                        'predicted_class': predicted_class})


class DiagnoseMammogramImage(View):
    def post(self, request):
        mammogram_image = request.FILES['mammogram-image']

        fs = FileSystemStorage()
        filename=fs.save(mammogram_image.name, mammogram_image)
        mammogram_model = keras.models.load_model("./AlexNet_ddsm_new.save")
        classes = {0:'Negative', 1:'Benign Calcification', 2:'Benign Mass', 3:'Malignant Calcification', 4:'Malignant Mass'}

        with tf.Session() as sess:
            print('global_variables_initializer...')
            sess.run(tf.global_variables_initializer())
            np.random.seed(1)
            tf.set_random_seed(2)
            data = Image.open('media/'+filename).getdata()
            image_numpy = np.array(data)
            prediction = mammogram_model.predict(image_numpy.reshape([1,299,299,1]), batch_size=1)

        predicted_probability = max(prediction[0])
        predicted_class = list(prediction[0]).index(predicted_probability)

        return render(request, 'Results.html', context={'source_image': 'media/'+filename,
                                                        'predicted_probability': predicted_probability,
                                                        'predicted_class': classes[predicted_class]})

        
