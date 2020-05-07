from django.shortcuts import render
from django.views import View
from django.http import HttpResponse
import tensorflow as tf
import keras, pydicom, os
import numpy as np
from skimage import exposure
from PIL import Image
from django.core.files.storage import FileSystemStorage


# Create your views here.
class DiagnoseDICOMImage(View):

    def process_input_1(self,file_name):
        loaded_dicom = exposure.equalize_adapthist(pydicom.read_file(file_name).pixel_array).reshape(512,512,1)
        tf_tensor = tf.image.grayscale_to_rgb(tf.convert_to_tensor(loaded_dicom))
        return np.array(tf_tensor.eval())
        
    def process_input_2(self, file_name):
        loaded_dicom = exposure.equalize_adapthist(pydicom.read_file(file_name).pixel_array).reshape(512,512,1)
        return loaded_dicom

    def post(self, request):
        for filename in os.listdir('media'):
            file_path = 'media/'+filename
            if os.path.exists(file_path):
                os.remove(file_path)

        immemoryfile = request.FILES['dicom_image']
        fs = FileSystemStorage()
        filename = fs.save(immemoryfile.name, immemoryfile)
        dicom_model_1 = keras.models.load_model("./model_vgg_dicom.h5")
        dicom_model_2 = keras.models.load_model("./model_resnet(1).h5")
        classes = {0:'Negative', 1:'Benign', 2:'Malign'}
        
        with tf.Session() as sess:
            print('global_variables_initializer...')
            sess.run(tf.global_variables_initializer())
            test_image = Image.fromarray(pydicom.read_file('media/'+filename).pixel_array)
            test_image.convert('L').save('media/test.png')
            data_1 = self.process_input_1('media/'+filename)
            data_2 = self.process_input_2('media/'+filename)
        
            prediction_1 = dicom_model_1.predict(data_1.reshape([1,512,512,3]), batch_size=1)
            prediction_2 = dicom_model_2.predict(data_2.reshape([1,512,512,1]), batch_size=1)

        final_prediction = prediction_1[0] + prediction_2[0]
        predicted_probability = max(final_prediction)
        predicted_class = list(final_prediction).index(predicted_probability)

        return render(request, 'Results.html', context={'source_image': 'media/test.png',
                                                        'predicted_probability': round(predicted_probability/2, 4),
                                                        'predicted_class': classes[predicted_class]})



class DiagnoseMammogramImage(View):

    def post(self, request):
        for filename in os.listdir('media'):
            file_path = 'media/'+filename
            if os.path.exists(file_path):
                os.remove(file_path)
        
        mammogram_image = request.FILES['mammogram-image']

        fs = FileSystemStorage()
        filename=fs.save(mammogram_image.name, mammogram_image)
        mammogram_model = keras.models.load_model("./DDSM.save")
        classes = {0:'Negative', 1:'Benign Calcification', 2:'Benign Mass', 3:'Malignant Calcification', 4:'Malignant Mass'}

        with tf.Session() as sess:
            
            print('global_variables_initializer...')
            sess.run(tf.global_variables_initializer())
            np.random.seed(1)
            tf.set_random_seed(2)
            data = Image.open('media/'+filename).convert('L').getdata()
            image_numpy = np.array(data).reshape([1,299,299,1])
            prediction = mammogram_model.predict(image_numpy)

        predicted_probability = max(prediction[0])
        predicted_class = list(prediction[0]).index(predicted_probability)

        return render(request, 'Results.html', context={'source_image': 'media/'+ filename,
                                                        'predicted_probability': predicted_probability,
                                                        'predicted_class': classes[predicted_class]})

        
class DiagnoseLungXRay(View):

    def post(self, request):
        for filename in os.listdir('media'):
            file_path = 'media/'+filename
            if os.path.exists(file_path):
                os.remove(file_path)
        
        xray = request.FILES['x-ray-image']

        fs = FileSystemStorage()
        filename=fs.save(xray.name, xray)
        xray_model = keras.models.load_model("./model_vgg16.h5")
        classes = {0:'Negative', 1:'Positive'}

        with tf.Session() as sess:
            print('global_variables_initializer...')
            sess.run(tf.global_variables_initializer())
            np.random.seed(1)
            tf.set_random_seed(2)
            img = np.array(Image.open('media/'+filename).resize((224, 224)).convert('RGB').getdata()).reshape(224,224,3)
            data = (1./255) * img   

            image_numpy = np.array(data).reshape([1,224,224,3])
            prediction = xray_model.predict(image_numpy)

        predicted_probability = max(prediction[0])
        predicted_class = list(prediction[0]).index(predicted_probability)

        return render(request, 'Results.html', context={'source_image': 'media/'+ filename,
                                                        'predicted_probability': predicted_probability,
                                                        'predicted_class': classes[predicted_class]})