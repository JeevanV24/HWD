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
def prediction(model):
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

def test(model,Xts,y_test):
    
    def testing():
        print("Testing the model")
        y_pred = model.predict(Xts)
        acc=metrics.accuracy_score(y_test, y_pred)*100
        messagebox.showinfo("Tested", "Achieved "+str(acc)+" Accuracy")
        pkl_filename = "digits_pickle_model.pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(model, file)
        window.destroy()
        prediction(model)

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
    trainImg = tk.Button(window, text="Test Model", command=testing  ,fg="purple"  ,bg="light blue"  ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
    trainImg.place(x=370, y=235)
    window.mainloop() 
    
def start_train(Xtr,Xts,y_train,y_test):
    def train():
        print("Selecting the model")
        model = SVC(C=10, gamma = 0.001, kernel="rbf")
        print("Training Started")
        model.fit(Xtr, y_train)
        messagebox.showinfo("Trained", "Model Trained Successfully")
        window.destroy()
        test(model,Xts,y_test)
    window = tk.Tk()
    window.title("Digit_Recogniser")
    #answer = messagebox.askquestion(dialog_title, dialog_text)
    window.geometry('1000x600')
    image = PIL.Image.open('b.jpg')
    i=(1000,600)
    image=image.resize(i)
    photo_image = PIL.ImageTk.PhotoImage(image)

    Label(window, image = photo_image).pack()
    window.configure(background='blue')
    trainImg = tk.Button(window, text="Start Training", command=train  ,fg="purple"  ,bg="light blue"  ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
    trainImg.place(x=370, y=235)
    window.mainloop()
def extract_pca(X_train, X_test, y_train, y_test):
    def extract():
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.decomposition import PCA,IncrementalPCA
        pca = PCA()
        pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
        Xtr = pipe.fit_transform(X_train)
        Xts=pipe.fit_transform(X_test)
        print(X_train[0][0].shape)
        print(X_test[0][0].shape)
        pca=IncrementalPCA(n_components=400)
        x_train=pca.fit_transform(X_train)
        x_test=pca.transform(X_test)
        messagebox.showinfo("PCA Extracted", "PCA extracted Successfully")
        pca_graph(Xtr,Xts,x_train,x_test)
    def pca_graph(Xtr,Xts,x_train,x_test):
        window.destroy()
        plot = plt.scatter(Xtr[:,0], Xtr[:,1], c=y_train)
        plt.legend(handles=plot.legend_elements()[0], labels=list([0,1,2,3,4,5,6,7,8,9]))
        plt.show()

        start_train(x_train,x_test,y_train,y_test)

    window = tk.Tk()
    window.title("Digit_Recogniser")
    #answer = messagebox.askquestion(dialog_title, dialog_text)
    window.geometry('1000x600')
    image = PIL.Image.open('b.jpg')
    i=(1000,600)
    image=image.resize(i)
    photo_image = PIL.ImageTk.PhotoImage(image)

    Label(window, image = photo_image).pack()
    window.configure(background='blue')
    trainImg = tk.Button(window, text="Extract PCA", command=extract  ,fg="purple"  ,bg="light blue"  ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
    trainImg.place(x=370, y=235)
    window.mainloop()
    
def plot_graph(count_table,training_data):
    def draw_graph():
        plts.figure(figsize=(10, 5))
        sns.barplot(x='index', y='label', data=count_table)
        plt.show()
        print("Loading heatmaap graph......")
        digit_means = training_data.groupby('label').mean()
        digit_means.head()
        plts.figure(figsize=(18, 10))
        sns.heatmap(digit_means)
        plt.show()
        window.destroy()
        extract_pca(X_train, X_test, y_train, y_test)
    global window
    window.destroy()
    print("Shaping the data for Training and Testing")
    round(training_data.drop('label', axis=1).mean(), 2).sort_values()
    X = training_data.drop("label", axis = 1)
    y = training_data['label']
    # scaling the features
    X_scaled = scale(X)
    # train test split
    print("Spliting the data for training and testing")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, random_state = 101)
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
    trainImg = tk.Button(window, text="Draw Graph", command=draw_graph  ,fg="purple"  ,bg="light blue"  ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
    trainImg.place(x=370, y=235)
    window.mainloop()

def LoadData():
    for dirname, _, filenames in os.walk(r'./Data'):
        for filename in filenames:
            print(os.path.join(dirname, filename))
    test_data = pd.read_csv(r"./Data/mnist_test.csv")
    training_data = pd.read_csv(r"./Data/mnist_train.csv")
    training_data.max().sort_values()
    training_data.isna().sum().sort_values(ascending=False)
    training_data.duplicated().sum()
    count_table = training_data.label.value_counts()
    count_table = count_table.reset_index().sort_values(by='index')
    messagebox.showinfo("Data Load", "Data Loaded Successfully")
    plot_graph(count_table,training_data)
    
    
window = tk.Tk()
#helv36 = tk.Font(family='Helvetica', size=36, weight='bold')
window.title("Digit_Recogniser")
#answer = messagebox.askquestion(dialog_title, dialog_text)
window.geometry('1000x600')
image = PIL.Image.open('bg.jpg')
i=(1000,600)
image=image.resize(i)
photo_image = PIL.ImageTk.PhotoImage(image)

Label(window, image = photo_image).pack()
window.configure(background='blue')
trainImg = tk.Button(window, text="Load Data", command=LoadData  ,fg="purple"  ,bg="light blue"  ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
trainImg.place(x=370, y=235)
window.mainloop()
