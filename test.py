import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plts
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA, IncrementalPCA
import tkinter as tk 
from tkinter.ttk import *
from tkinter import filedialog as fd
from tkinter import messagebox
import PIL.Image
import PIL.ImageTk
import os
import time
import matplotlib.pyplot as plt
import pickle
#from skimage.transform import resizedef prediction(model):
def open_file():
    from skimage.transform import resize
    # file type
    filetypes = (
        ('Image files', '*.png'),
        ('image2 files','*.jpg'),
        ('All files', '*.*')
    )
    # show the open file dialog
    f = fd.askopenfile(filetypes=filetypes)
    name = os.path.basename(f.name)
    from PIL import Image
    import numpy as np
    import pandas as pd
    import matplotlib.image as mpimg
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA,IncrementalPCA
    pca = PCA()
    img=name
    img = Image.open(img).convert('L')  # convert image to 8-bit grayscale
    newsize=(400,400)
    img = img.resize(newsize)
    WIDTH, HEIGHT = img.size

    data = list(img.getdata()) # convert image data to a list of integers
    # convert that to 2D list (list of lists of integers)
    img = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]
    img=scale(img)
    pca=IncrementalPCA(n_components=400)
    img=scale(img)
    X=pca.fit_transform(img)
    loaded_model = pickle.load(open('digits_pickle_model.pkl', 'rb'))
    result = loaded_model.predict(X)
    counter=0
    num = result[0]
    result=list(result)

    for i in result:
        curr_frequency = result.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
    messagebox.showinfo("Prediction", "Predicted value is:"+str(num))

 

window = tk.Tk()
window.title("Digit_Recogniser")
#answer = messagebox.askquestion(dialog_title, dialog_text)
window.geometry('1000x600')
image = PIL.Image.open('bg.jpg')
i=(1000,600)
image=image.resize(i)
photo_image = PIL.ImageTk.PhotoImage(image)

Label(window, image = photo_image).pack()
window.configure(background='blue')
trainImg = tk.Button(window, text="Select Image", command=open_file  ,fg="purple"  ,bg="light blue"  ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
trainImg.place(x=370, y=235)
window.mainloop()
