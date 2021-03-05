# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import cv2
from scipy import misc
from scipy.integrate import cumtrapz
from pylab import *
import ansyswrapper
import os
from scipy.optimize import curve_fit, minimize
from matplotlib import colors
from matplotlib.ticker import PercentFormatter


'''
gaussian_blur_size = (5, 5)
kernel = np.ones((5, 5), np.uint8)
file_name_Old = open('Old.txt', 'a')
file_name_Old.write('Dark_square  ; Mean_radius) ; STD_radius ; min(radius_3) ;max(radius_3))' + '\n')
file_name_Old.close()

def procimage(img_name):
    face = misc.imread(img_name, flatten=True, mode='L')
    face1d = face.reshape(face.size)
    hist, bin_edges = np.histogram(face1d, bins=256, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    #print(bin_centers)
    inthist = cumtrapz(hist, bin_centers, initial=0)
    #print(inthist)
    alpha = 0.05
    canny_threshold_1 = bin_centers[np.argmin(np.abs(inthist - alpha))]  # Left boundary
    canny_threshold_2 = bin_centers[np.argmin(np.abs(inthist - (1.0 - alpha)))]  # Right boundary
    #print(canny_threshold_1, canny_threshold_2)

    image = cv2.imread(img_name)
    image_predict = cv2.imread(img_name)
    image_height, image_width, image_channels = image.shape  # Image geometry
    image_prepared = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # To grayscale
    blur_img = cv2.GaussianBlur(image_prepared, gaussian_blur_size, 0)  # Bluring
    plt.imshow(blur_img, cmap=plt.cm.gray)
    plt.show()

    convert_img = cv2.bitwise_not(blur_img)  # Change white to black
    plt.imshow(convert_img, cmap=plt.cm.gray)
    #plt.show()

    open_img = cv2.morphologyEx(convert_img, cv2.MORPH_OPEN, kernel)  # Remove small white regions
    plt.imshow(open_img, cmap=plt.cm.gray)
    #plt.show()
    close_img = cv2.morphologyEx(open_img, cv2.MORPH_CLOSE, kernel)  # Remove small black hole

    plt.imshow(close_img, cmap=plt.cm.gray)
    #plt.show()
    convert_img2 = cv2.bitwise_not(close_img)  # Change white to black
 #   plt.imshow(convert_img2, cmap=plt.cm.gray)
#    plt.show()

    """
        plt.cla
        plt.clf()
        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()

        N, bins, patches = ax.hist(face1d, bins=256, density=True)

        # We'll color code by height, but you could use any scalar
        fracs = N / N.max()

        # we need to normalize the data to 0..1 for the full range of the colormap
        norm = colors.Normalize(fracs.min(), fracs.max())

        # Now, we'll loop through our objects and set the color of each accordingly
        for thisfrac, thispatch in zip(fracs, patches):
            color = plt.cm.viridis(norm(thisfrac))
            thispatch.set_facecolor(color)

        #plt.hist(face1d, density=True, bins=256, color="green")

        ax.axvline(canny_threshold_1, 0, 1, color = 'red', linewidth = 1.2, linestyle = ':')
        ax.axvline(canny_threshold_2, 0, 1, color = 'red', linewidth = 1.2, linestyle = ':')

        ax.set_xlabel('колір пікселя')
        ax.set_ylabel('шільність ймовірності')
        ax.set_title('Total pixels:' + str(len(face1d)))
        #ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        plt.show()
    """
    return close_img, canny_threshold_1, canny_threshold_2, image_height, image_width, image, image_predict
'''
# beginning
def EdDetect(close_img, canny_threshold_1, canny_threshold_2, file_name):
    image_prepared = cv2.Canny(close_img, canny_threshold_1, canny_threshold_2)  # Edge detection
    _, countours, hierarchy = cv2.findContours(image_prepared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    plt.imshow(image_prepared, cmap=plt.cm.gray)
    #plt.show()
    center_x = []
    center_y = []
    radius = []
    i = 0
    for i in countours:
        (x, y), rad = cv2.minEnclosingCircle(i)  # Сonfine circles
        center_x.append(x)
        center_y.append(y)
        radius.append(rad)

    radius_2 = []
    center_x_2 = []
    center_y_2 = []

    i = 0
    for i in range(len(countours)):  # Remove small circles
        if radius[i] > 4:
            radius_2.insert(i, radius[i])
            center_x_2.insert(i, center_x[i])
            center_y_2.insert(i, center_y[i])
        i += 1

    i = 0
    radius_3 = []
    center_x_3 = []
    center_y_3 = []
    for i in range(len(radius_2)):  # Remove the intersection of the circle
        flag = True
        j = 0
        while j < i:
            dim_c = np.sqrt((((center_x_2[i] - center_x_2[j]) ** 2) + (center_y_2[i] - center_y_2[j]) ** 2))
            rad_c = radius_2[i] + radius_2[j] + 1
            if radius_2[i] + radius_2[j] and dim_c < rad_c:
                flag = False
            j += 1

        if flag:
            radius_3.insert(i, radius_2[i])
            center_x_3.insert(i, center_x_2[i])
            center_y_3.insert(i, center_y_2[i])
    plt.cla
    plt.clf()
    print(len(radius_3))
    plt.hist(radius_3, 3)
    savename = os.path.splitext(file_name)[0]
    plt.savefig("hist_" + savename + ".png", dpi=300)
    return center_x_3, center_y_3, radius_3


def present_dispersion(radius_3, image_height, image_width):
    # Average value, dispersion, area ratio for given circles
    Mean_radius = np.mean(radius_3)
    STD_radius = np.std(radius_3)
    np_rad3 = np.array(radius_3)
    Dark_square = (np.pi * np_rad3 ** 2).sum() / (image_height * image_width)
    file_name_Old = "Old.txt"
    f = open(file_name_Old, 'a')
    f.write(str(Dark_square) + ';' + str(Mean_radius) + ';' + str(STD_radius) + ';' + str(min(radius_3)) + ';'
            + str(max(radius_3)) + '\n')
    f.close()

    return Mean_radius, STD_radius, file_name_Old, Dark_square


def received_dispersion(Mean_radius, STD_radius, center_x_3, image_height, image_width, file_name):
    num_features = len(center_x_3)
    Y, X = image_height, image_width
    rad = np.empty(num_features)

    i = 0
    for i in range(num_features):  # Obtaining new rad by normal distribution

        if (Mean_radius - STD_radius) > 1:
            rad[i] = np.abs(np.random.normal(Mean_radius, STD_radius, 1))
            continue

    rad = np.array([int(x) for x in rad])
    centr_coord_x = np.empty(num_features)
    centr_coord_y = np.empty(num_features)
    dim = []
    i = 0
    for i in range(num_features):  # Obtaining new coordination by normal distribution

        centr_coord_x[i] = np.random.uniform(0, X, 1)
        centr_coord_y[i] = np.random.uniform(0, Y, 1)
        j = 0
        while j < i:
            dim = np.sqrt((((centr_coord_x[i] - centr_coord_x[j]) ** 2) + (centr_coord_y[i] - centr_coord_y[j]) ** 2))

            if dim < rad[i] + rad[j] + 10:
                centr_coord_x[i] = np.random.uniform(0, X, 1)
                centr_coord_y[i] = np.random.uniform(0, Y, 1)
                j = 0
                continue
            j += 1

    centr_coord_x = [int(x) for x in centr_coord_x]
    centr_coord_y = [int(x) for x in centr_coord_y]

    New_Dark_square = (np.pi * rad ** 2).sum() / (image_height * image_width)
    New_Mean_radius = np.mean(rad)
    New_STD_radius = np.std(rad)
    file_name_New = 'New.txt'
    f = open(file_name_New, 'a')
    f.write(str(New_Dark_square) + ';' + str(New_Mean_radius) + ';' + str(New_STD_radius) + '\n')
    f.close()

    return centr_coord_x, centr_coord_y, rad, num_features, file_name_New, New_Mean_radius, New_STD_radius


def plotgraf(file_name_old):

    def func(x, a, b, c):
        return a * (x + b) ** c


    Square_old = np.array(array_old[:, 0], dtype=float)
    Mean_old = np.array(array_old[:, 1], dtype=float)
    Disperthon_old = np.array(array_old[:, 2], dtype=float)

    popt_o_m, pcov_o_m = curve_fit(func, Square_old, Mean_old)
    popt_o_d, pcov_o_d = curve_fit(func, Square_old, Disperthon_old)

    a_m = popt_o_m[0]
    b_m = popt_o_m[1]
    c_m = popt_o_m[2]

    a_d = popt_o_d[0]
    b_d = popt_o_d[1]
    c_d = popt_o_d[2]

    residuals = Mean_old - func(Square_old, a_m, b_m, c_m)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Mean_old - np.mean(Mean_old)) ** 2)
    r_squared_m = 1 - (ss_res / ss_tot)

    residuals = Disperthon_old - func(Square_old, a_d, b_d, c_d)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Disperthon_old - np.mean(Disperthon_old)) ** 2)
    r_squared_d = 1 - (ss_res / ss_tot)
    print(r_squared_m, r_squared_d)
    f = open('k_k.csv', 'w')
    f.write(str(a_m) + ';' + str(b_m) + ';' + str(c_m) + '\n' + str(a_d) + ';' + str(b_d) + ';' + str(c_d) + '\n')
    f.close()
    plt.cla()
    plt.clf()
    plt.scatter(Square_old, Mean_old, s=30, color='orange')
    plt.plot(np.linspace(0.04,0.19,100), func(np.linspace(0.04,0.19,100), *popt_o_m), color='orange', label="M[R]")
    plt.scatter(Square_old, Disperthon_old, s=30, color='red')
    plt.plot(np.linspace(0.04,0.19,100), func(np.linspace(0.04,0.19,100), *popt_o_d), color='red', label="D[R]")
    #  Добавляем подписи к осям:
    plt.xlabel('Концентрація включень, %')
    plt.ylabel('Розмір включень, dpi')
    plt.title('Histogram of IQ')
    plt.grid(True)
    plt.legend()
    savefig("plotgraf.png")
    plt.close()
    return a_m, b_m, c_m, a_d, b_d, c_d

def plotimg(file_name, image, image_width, image_height, num_features, centr_coord_x, centr_coord_y, rad, pref):

    ax = plt.gca()
    ax.cla()  # clear things for fresh plot
    # change default range so that new circles will work
    ax.set_xlim((0, image_width))
    ax.set_ylim((0, image_height))
    ax.set_aspect(aspect='equal')

    if pref == "predict":
        ax = plt.gca()
        ax.cla()  # clear things for fresh plot
        ax.set_xlim((0, image_width))
        ax.set_ylim((0, image_height))
        ax.set_aspect(aspect='equal')

        for i in range(num_features):
            circle = plt.Circle((centr_coord_x[i], centr_coord_y[i]), rad[i], color=(1, 0, 0), fill=True)
            ax.add_artist(circle)
        savename = os.path.splitext(file_name)[0]
        plt.savefig(pref + "_" + savename + ".png", dpi=300, color='green')

    else:
        for i in range(num_features):
            cv2.circle(image, center=(int(centr_coord_x[i]), int(centr_coord_y[i])), radius=int(rad[i]), color=(0, 0, 255), thickness=2)

        savename = os.path.splitext(file_name)[0]

        cv2.imwrite(pref + "_" + savename + ".png", image)

    cv2.destroyAllWindows()


def plot3d(a_to_m, b_to_m, c_to_m, a_to_d, b_to_d, c_to_d, all_radius, all_concentr):

    def psi_func(psi, a, b, c):
        return a * (psi + b) ** c

    def norm3d(radius, psi):
        func3d = np.exp(-0.5 * ((radius - psi_func(psi, a_to_m, b_to_m, c_to_m))
                           / psi_func(psi,a_to_d, b_to_d, c_to_d)) ** 2)\
                            / psi_func(psi, a_to_d, b_to_d, c_to_d) / np.sqrt(2 * np.pi)
        return func3d

    RR = np.linspace(start=min(min(all_radius)), stop=max(max(all_radius)), num=1000, dtype=float)
    PSI = np.linspace(start=min(all_concentr), stop=(max(all_concentr) + 0.01 * max(all_concentr)), num=1000, dtype=float)

    RR, PSI = np.meshgrid(RR, PSI)

    func3d = norm3d(RR, PSI)
    plt.cla()
    plt.clf()
    plt.figure()
    # ax = fig.gca(projection='3d')
    plt.contourf(PSI, RR, func3d)
    savename = os.path.splitext(file)[0]
    plt.savefig("norm_distrib " + savename + ".png", dpi=300)
    plt.show()
    plt.close()


def Ansys_calculation(num_features, image_height, image_width, rad, centr_coord_x, centr_coord_y, file):
    # for Ansys
    savedir = 'c:\\ans_proj\\imgproc'
    ans = ansyswrapper(projdir=savedir, jobname='myjob')
    ans.setFEByNum(183)

    ferriteid = ans.createIsotropicMat(E=210e9, nu=0.3)
    graphiteid = ans.createOrtotropicMat(c11=1060e9, c33=1060e9, c13=180e9, c12=15e9, c23=15e9, c22=36.5e9, c44=0.18e9,
                                         c55=4.35e9)

    for i in range(num_features):
        ans.circle(centr_coord_x[i], image_height - centr_coord_y[i], rad[i])

    ans.rectangle(0, 0, image_width, image_height)
    ans.overlapAreas()
    ans.delOuterArea(0, 0, image_width, image_height)
    ans.setAreaProps(np.pi * rad.max() ** 2, matId=graphiteid)
    ans.mesh()

    ans.applyTensX(0, 0, image_width, image_height)
    ans.applyTensY(0, 0, image_width, image_height)
    ans.applyTensXandY(0, 0, image_width, image_height)
    ans.applyShearXY(0, 0, image_width, image_height)
    ans.precessElasticConstants()

    ans.saveToFile(savedir + '\\test.apdl')
    ans.run()
    print("All done for " + str(file) + " image")


if os.path.exists("Old.txt"):
    os.remove("Old.txt")

if os.path.exists("New.txt"):
    os.remove("New.txt")

RadiusAll = []
ConcentrAll = []
for file in os.listdir("./images"):
    if file.endswith(".jpg"):
        print('Image ' + str(file) + ' in progress...')
        close_img, canny_threshold_1, canny_threshold_2, image_height, image_width, image, image_predict = procimage(
            file)
        center_x_3, center_y_3, radius_3 = EdDetect(close_img, canny_threshold_1, canny_threshold_2, file)
        Mean_radius, STD_radius, file_name_Old, Dark_square = present_dispersion(radius_3, image_height, image_width)
        entr_coord_x, centr_coord_y, rad, num_features, file_name_New, New_Mean_radius, New_STD_radius = received_dispersion(Mean_radius, STD_radius, center_x_3, image_height, image_width, file)

        #plotimg(file, image, image_width, image_height, num_features, centr_coord_x, centr_coord_y, rad,
                #pref="predict")

        plotimg(file, image, image_width, image_height, num_features, center_x_3, center_y_3, radius_3,
                pref="detect")

#        ns = Ansys_calculation(num_features, image_height, image_width, rad, centr_coord_x, centr_coord_y, file)

        #RadiusAll.append(radius_3)
        #ConcentrAll.append(Dark_square)

#a_m, b_m, c_m, a_d, b_d, c_d = plotgraf(file_name_Old)
#plot3d(a_m, b_m, c_m, a_d, b_d, c_d, RadiusAll, ConcentrAll)

print("Finish")
