from django.shortcuts import render
import os
import pickle
from django.conf import settings
from django.shortcuts import render

#Paths to saved model and vectorizer..
MODEL_PATH = os.path.join(settings.BASE_DIR, 'detector/models/spam_model.pkl')
VECTORIZER_PATH = os.path.join(settings.BASE_DIR, 'detector/models/vectorizer.pkl')

#load once when the server starts
with open(MODEL_PATH, 'rb') as f:
    nb_model = pickle.load(f)
    
with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)
    
def classify_message(request):
    if request.method == 'POST':
        message = request.POST.get('message')
        #vectorize the message
        message_vector = vectorizer.transform([message])
        #predict using the model
        prediction = nb_model.predict(message_vector)[0]
        result = 'Spam' if prediction == 1 else 'Not Spam'
        return render(request, 'index.html', {'result': result})
    return render(request, 'index.html')    


