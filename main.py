import tkinter as tk
#import ansyswrapper as aw
from tkinter.filedialog import askopenfilename
import processing as pc
from PIL import Image, ImageTk
import numpy as np


image_file = None
originimage = None
proceimage = None

def resize(w, h, w_box, h_box, pil_image):
    '''
    resize a pil_image object so it will fit into
    a box of size w_box times h_box, but retain aspect ratio
    '''
    f1 = 1.0 * w_box / w  # 1.0 forces float division in Python2
    f2 = 1.0 * h_box / h
    factor = min([f1, f2])
    # print(f1, f2, factor) # test
    # use best down-sizing filter
    width = int(w * factor)
    height = int(h * factor)
    # Use resize to zoom
    return pil_image.resize((width, height), Image.ANTIALIAS)


def open_image():
    global image_file
    # У змінну filepath запишемо результат виконання askopenfile
    filepath = askopenfilename()
    # open the picure with Image
    image_file = Image.open(filepath)
    w_box = 500
    h_box = 350
    showimg(image_file, imgleft, w_box, h_box)
    showimg(image_file, imgright, w_box, h_box)


def save_image():
    global proceimage
    proceimage.save('images/processed.jpg')


def hst_eql():
    global proceimage
    PIL_eq, PIL_gary = pc.hist_eql(image_file)
    proceimage = PIL_eq
    # expected image display size
    w_box = 500
    h_box = 350
    showimg(PIL_gary, imgleft, w_box, h_box)
    histO = Image.open('images/templeft.png')
    showimg(histO, histleft, w_box, h_box)

    showimg(PIL_eq, imgright, w_box, h_box)
    histE = Image.open('images/tempright.png')
    showimg(histE, histright, w_box, h_box)
    # pc.drawHist(PIL_img,'right')
    # draw_hist(PILimg, 'right')


def showimg(PIL_img, master, width, height):
    """
    :param PIL_img: picture to display
    :param master: Need to be displayed in this element
    :param width: desired maximum width
    :param height: expected maximum height
    :return: nothing
    """
    # Get image parameters
    w, h = PIL_img.size
    # Zoom image
    img_resize = resize(w, h, width, height, PIL_img)
    # Image 2 ImageTk
    Tk_img = ImageTk.PhotoImage(image=img_resize)
    # master display image
    master.config(image=Tk_img)
    master.image = Tk_img

def edge():
    PIL_detect = pc.edge_detect(image_file)
    global proceimage 
    proceimage = PIL_detect
    # Expected image display size
    w_box = 500
    h_box = 350
    showimg(image_file, imgleft, w_box, h_box)
    showimg(PIL_detect, imgright, w_box, h_box)
    histleft.config(image=None)
    histleft.image = None
    histright.config(image=None)
    histright.image = None

def otsu():
    PIL_gary,PIL_Otsu = pc.otsu_hold(image_file)
    w_box = 500
    h_box = 350
    showimg(PIL_gary, imgleft, w_box, h_box)
    showimg(PIL_Otsu, imgright, w_box, h_box)
    histleft.config(image=None)
    histleft.image = None
    histright.config(image=None)
    histright.image = None
    global proceimage
    proceimage = PIL_Otsu

def fe_calc():
    pass

root = tk.Tk()
root.title('Image processing')
root.geometry('1100x700')
root.config(bg='white')

#menubar
menubar = tk.Menu(root)
filemenu = tk.Menu(menubar, tearoff = 0)
filemenu.add_command(label='open', command = open_image)
filemenu.add_command(label='save', command = save_image)
operate = tk.Menu(menubar, tearoff = 0)
operate.add_command(label='Histogram equalization', command=hst_eql)
operate.add_command(label='Edge recognition by Canny', command=edge)
operate.add_command(label='Otsu treshold method',command = otsu)
ansys_calc = tk.Menu(menubar, tearoff=0)
ansys_calc.add_command(label='FE calculaction by Ansys' command = fe_calc)
menubar.add_cascade(label='file', menu = filemenu)
menubar.add_cascade(label='operate', menu = operate)
menubar.add_cascade(label='FE calc', menu = ansys_calc)


# 
frm = tk.Frame(root, bg='white')
frm.pack()

frm_left = tk.Frame(frm, bg='white')
frm_right = tk.Frame(frm, bg='white')
# 
frm_left.pack(side='left')
frm_right.pack(side='right')

imgleft = tk.Label(frm_left, bg='white')
histleft = tk.Label(frm_left, bg='white')

imgright = tk.Label(frm_right, bg='white')
histright = tk.Label(frm_right, bg='white')
imgleft.pack()
histleft.pack()
imgright.pack()
histright.pack()
# canvasl = tk.Canvas(frm_left, bg='white').pack()
# canvasr = tk.Canvas(frm_right, bg='white').pack()


root.config(menu=menubar)
root.mainloop()