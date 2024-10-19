import gradio as gr
import numpy as np
from keras.models import load_model
import cv2
loaded_model = load_model('fruitgrader.h5')


def classify_fruit(image):
    img_array=cv2.resize(image, (150,150)).astype(np.float64)
    img_array = np.expand_dims(img_array, axis=0)  # Reshape (add batch dimension)

    # Preprocess the image (rescale pixel values)
    img_array /= 255

    # Predict the class probabilities
    predictions = loaded_model.predict(img_array)
    
    # Get the class with the highest probability
    class_index = np.argmax(predictions)
    # Define classes
    classes = ['Fresh Apple', 'Fresh Banana', 'Fresh Orange','Rotten Apple','Rotten Banana','Rotten Orange']
    print(predictions)
    # Return the predicted class
    if np.max(predictions)>0.8:
        text1 = f"This fruit belongs to {classes[class_index]} class"
        text2 = f"Probability of prediction class is {round(np.max(predictions)*100, 2)}%"
    else:
        text1="It doesn't look like specified class image!"
        text2="Probability values are low!"
    return text1, text2

demo = gr.Interface(fn=classify_fruit, inputs=gr.Image(), outputs=[gr.Text(),gr.Text()],title="FRUIT GRADER",description="Fruit quality analysing system")
demo.launch(debug=True,share=True)


#run with command ```gradio app.py```