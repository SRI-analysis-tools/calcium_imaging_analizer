import sys, os, numpy
#from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QGraphicsScene, QGraphicsPixmapItem
#from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.figure import Figure
import time
import pickle
#from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from caimager6 import *
#from matplotlib import pyplot as plt
import scipy.io
#from PyQt5.QtGui import QIcon, QPixmap, QImage
import mne
from xml.dom import minidom
import scipy.signal
from scipy import stats
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import random
#import matplotlib.image as mpimg
#import pandas as pd
import itertools
import h5py

"""If there is EEG, traces need to by synced with EEG. If traces include dropped frames, the they should match 
the N frames from the mat files, else they should match the nu,ber of detected frames form the sync signal.
A vector with the time of each frame will be generated, matching the EEG time of the first frame in every chunk.
Convert ui with pyuic5 demoLineEdit.ui -o demoLineEdit.py"""

#auxilliary functions and classes:
def ampfun(data, ampFactor):  # function that amplifies a signalwith low temporal res
    data = data.flatten()
    newvector = numpy.zeros(len(data) * ampFactor)
    for i in range(len(data) + 1):
        newvector[ampFactor * (i - 1):ampFactor * i] = data[i - 1]
    return (newvector)

def baseln(data):#gets the baseline as 10% of the smallest point
    l=len(data)
    if l<2:
        raise Exception ('Error, not enough data points',data)
    else:
        sd = data.copy()
        sd.sort()
        return(sd[int(0.1*l)])
def sub_baseln(data):
    """Substracts the baseline to an array"""
    return data-baseln(data)
def get_delta(x,sf):
    """Return the mean delta power [0-4 Hz] of an array x at sampling frec sf """
    fftmat = numpy.abs(numpy.fft.fft(x)) ** 2
    freqs = numpy.fft.fftfreq(len(x), 1 / sf)
    # now finding the pos of the 0 and maxfrec
    pos0 = numpy.argmin(numpy.abs(freqs))
    posf = numpy.argmin(numpy.abs(freqs-4))
    return fftmat[pos0:posf].mean()

def rect_eq(x0,y0,x1,y1):
    m = (y1 - y0) / (x1 - x0)
    n = y0 - ((y0-y1)/(x0-x1))*x0

    return(m,n)

def pos_LP(tpl,px,py): #calculates the distance between a straight line defined by x0,y0,x1,y1 and a point px,py. returns the coordinate of the intersection in the straight line
    (x0, y0, x1, y1) = tpl
    m,n = rect_eq(x0, y0, x1, y1)
    X = (py+(px/m)-n)/(2*(1/m))
    Y = (py+(px/m)+n)/2
    #print("m = {0}, n = {1}, X = {2}, Y = {3}".format(m,n,X,Y))

    return (X,Y)

def dist_TP(x0,y0,x1,y1):
    return((((y0-y1)**2) + ((x0-x1)**2))** 0.5)


def make_triangle(L): # returna a 3-dim array [rows,cols,color] with the triangle where the amount of red, green and blue is given by te distance to the respective centr line (hight)

    #defining the lines that form the sides of the triangle
    x0s1, y0s1 = 0,0
    x1s1, y1s1 = L/2, (L/2)*(3**0.5)
    x0s2, y0s2 = L,0
    x1s2,y1s2 = x1s1, y1s1

    ms1, ns1 =  rect_eq(x0s1,y0s1,x1s1,y1s1)
    ms2, ns2 = rect_eq(x0s2, y0s2, x1s2, y1s2)

    # defining the lines that form the hights for red and blue
    x0hr, y0hr = L*(13**0.5)/8, L*(3**0.5)/4
    x1hr, y1hr = L, 0
    x0hb, y0hb = L*(1-((13**0.5)/8)), L*(3**0.5)/4
    x1hb, y1hb = 0,0
    tr= (x0hr,y0hr,x1hr,y1hr)
    tb = (x0hb,y0hb,x1hb,y1hb)

    triangle = numpy.zeros([int((L/2)* (3**0.5)),L,3])

    #plt.plot([(x,m1*x+n1) for x in range(100)])
    #plt.plot([(x, m2 * x + n2) for x in range(100,200)])
    #plt.plot([x for x in range(100,200)],yp2)
    #plt.show()
    #Filling the matrix, point by point
    for xpos in range(L):
        for ypos in range(int((L/2)*(3**0.5))):
            if ((xpos <= L/2) and (ypos > (ms1 *xpos + ns1))) or ((xpos > L/2) and (ypos > (ms2 *xpos + ns2))):
                    r, g, b = 0, 0, 0 #outside the triangle
            else:
                r = dist_TP(pos_LP(tr, xpos, ypos)[0],pos_LP(tr, xpos, ypos)[1], x0hr, y0hr)/((L/2)*(3**0.5))
                b = dist_TP(pos_LP(tb, xpos, ypos)[0],pos_LP(tb, xpos, ypos)[1], x0hb, y0hb)/((L/2)*(3**0.5))
                g = ypos/((L/2)*(3**0.5))
            triangle[ypos,xpos,:] = [r,g,b]
    plt.imshow(triangle)
    plt.gca().invert_yaxis()
    plt.axis("off")
    plt.show()

def interp_df(df,n):#multiplies number of rows by n, interpolating data in between
    df2=pd.DataFrame(np.nan, index=np.arange(n*len(df)), columns=list(df.keys()))
    for i in list(df.index):
        df2.iloc[n*i,:]=df.iloc[i,:]
    return df2.interpolate(method='spline',order=2, limit_direction='forward', axis=0)

#core classes

class MyForm(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.Filters=[]
        self.Traces = []
        self.time_traces = []
        self.nframes_from_mats = [] #array with the number of frames of every concatenated movie
        self.Score=[]
        self.meanZscore=[] #Mean z score for each period
        self.zsc=[] #Mean z score for each period for each cell
        self.sr=15
        self.filtpath=''
        self.filtfn = ''
        self.timesscore=[]
        self.EEG_NR = []
        self.traces_NR = []
        self.starttimes = []
        self.endtimes = []
        self.frinmov = []
        self.discardmats=0
        #specify all the controls here:
        self.ui.pushButton.clicked.connect(self.loadFilt)
        # self.ui.ImgView.ui.roiBtn.hide()
        # self.ui.ImgView.ui.histogram.hide()
        # self.ui.ImgView.ui.menuBtn.hide()
        self.ui.pushButton_2.clicked.connect(self.loadTraces)
        self.ui.pushButton_3.clicked.connect(self.loadMats)
        self.ui.pushButton_4.clicked.connect(self.loadScore)
        self.ui.pushButton_5.clicked.connect(self.loadSync)
        self.ui.pushButton_6.clicked.connect(self.saveAll)
        self.ui.pushButton_9.clicked.connect(self.setStart)
        self.ui.pushButton_10.clicked.connect(self.setEnd)
        self.ui.Nperiods.valueChanged.connect(self.updateNperiods)
        self.ui.lineEdit.editingFinished.connect(self.updateFilters)
        self.show()
        self.cell_activity=[]
        self.starttimesdict={1:''}
        self.endtimesdict = {1: ''}
        self.ui.mpl.setStyleSheet("background-color:black;")
        """
        fs = 500
        f = random.randint(1, 100)
        ts = 1 / fs
        length_of_signal = 100
        t = numpy.linspace(0, 1, length_of_signal)

        cosinus_signal = numpy.cos(2 * numpy.pi * f * t)
        sinus_signal = numpy.sin(2 * numpy.pi * f * t)

        self.ui.mpl.canvas.axes.clear()
        #self.ui.mpl.canvas.axes.tight_layout()
        aux = self.ui.mpl.canvas.axes.plot(t, cosinus_signal)
        self.ui.mpl.canvas.axes.plot(t, sinus_signal)
        self.ui.mpl.canvas.axes.legend(('cosinus', 'sinus'), loc='upper right')
        self.ui.mpl.canvas.axes.set_title('Cosinus - Sinus Signal')
        #aux[0].tight_layout()
        self.ui.mpl.canvas.draw()
        """
        self.ui.mpl.canvas.axes.axis("off")
        self.mW,self.mN,self.mR = [0],[0],[0]
        #make_triangle(400)

        #now def all the functions
    def updateNperiods(self):
        self.ui.current_period.maximum = self.ui.Nperiods.value()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_F12:
            self.close()
        if e.key() == Qt.Key_T:
            self.loadFilt()

    def updateFilters(self): #The 3 immputs are vectors with the mean Z scores for the respective state for each cell
        os.chdir(self.filtpath)
        #print(self.filtpath)
        filelist = os.listdir()
        nelem = 0
        thresf = float(self.ui.lineEdit.text())
        for i in filelist:
            if self.ui.checkBox_2.isChecked(): #loading tiff files
                if np.max([c.strip() in i for c in self.colnames]):
                    im = Image.open(i)
                    matrixobj = np.array(im)
                    #print('shapeimg:',matrixobj.shape)
                    matrixobj = matrixobj / matrixobj.max()
                    matrixobj[matrixobj < thresf] = 0  # This needs to be adjusted by hand one by one
                    nrows, ncols = matrixobj.shape
                    kernel = numpy.ones((3, 3))
                    matrixobj = cv2.erode(matrixobj, kernel, iterations=1)
                    if nelem == 0:
                        monomat = numpy.zeros([len(filelist), nrows, ncols, 3])
                    if len(self.mW) > 1:
                        monomat[nelem, :, :, :] = numpy.asarray([self.mR[nelem] * (matrixobj.T), self.mW[nelem] * matrixobj.T, self.mN[nelem] * matrixobj.T]).T
                    nelem += 1
            else:
                if i.endswith(".mat"):
                    #print("opening " + i)
                    f = scipy.io.loadmat(self.filtpath + "/" + i)
                    o1 = f['Object']
                    aux = o1['Data']
                    matrixobj = aux[0, 0]
                    #print('shapeimg:',matrixobj.shape)
                    matrixobj = matrixobj / matrixobj.max()
                    matrixobj[matrixobj < thresf] = 0  # This needs to be adjusted by hand one by one
                    nrows, ncols = matrixobj.shape
                    kernel = numpy.ones((3, 3))
                    matrixobj = cv2.erode(matrixobj, kernel, iterations=1)
                    if nelem == 0:
                        monomat = numpy.zeros([len(filelist), nrows, ncols, 3])
                    if len(self.mW) > 1:
                        monomat[nelem, :, :, :] = numpy.asarray([self.mR[nelem] * (matrixobj.T), self.mW[nelem] * matrixobj.T, self.mN[nelem] * matrixobj.T]).T
                    nelem += 1
        maxmat = numpy.amax(monomat, 0)
        colormat = maxmat
        # try to plot it
        #mat2plot = numpy.amax(allmatrix, 0)
        self.mat2plot = 1.4*(colormat / colormat.max())
        self.ui.mpl.canvas.axes.clear()
        #self.ui.mpl.canvas.axes.plt.imshow(mat2plot)
        self.ui.mpl.canvas.axes.imshow(self.mat2plot)
        self.ui.mpl.canvas.axes.axis("off")
        self.ui.mpl.canvas.draw()
        self.Ncells = nelem
        #iv = self.ui.ImgView
        #iv.setImage(mat2plot)
        self.filters = colormat
    

    def setStart(self):
        #set starts time for the selected period
        period = self.ui.current_period.value()
        stime = self.ui.timeEdit.time()
        self.starttimesdict[period] = stime
        print('Start time for period {0} = {1}'.format(period,stime))

    def setEnd(self):
        #set starts time for the selected period
        period = self.ui.current_period.value()
        etime = self.ui.timeEdit_2.time()
        self.endtimesdict[period] = etime
        print('End time for period {0} = {1}'.format(period, etime))

    def loadTraces(self):
            Path = str(QFileDialog.getExistingDirectory(self, "Select Directory with Traces"))
                #self.ui.labelResponse.setText("Hello " + self.ui.lineEditName.text())
            # if self.ui.checkBox.isChecked():
            #     print(os.getcwd().rsplit('\\',1))
            #     aux1,aux2= os.getcwd().rsplit('\\',1)
            #     Path = aux1+'\\traces'
            # else:
            #     Path = str(QFileDialog.getExistingDirectory(self, "Select Directory with Traces"))
            if len(Path) > 1:
                os.chdir(Path)
                #self.ui.label.setText(filen)
                filelist = os.listdir()
                nelem=0
                plotCanv = self.ui.PlotWidget_tr
                if self.ui.checkBox_2.isChecked():
                    #load the first csv file
                    for i in filelist:
                        if i.endswith(".csv"): 
                            print('Loading ',i)
                            mydata0 = pd.read_csv(i, nrows=1)
                            mydata1 = pd.read_csv(i, skiprows=1,header=0)
                            inc_list=[mydata0[k].str.contains('accepted').any() for k in mydata0.keys()]
                            self.colnames=mydata0.loc[:,inc_list].columns
                            mydata1=mydata1.loc[:,inc_list]
                            mydata1.columns=self.colnames
                            tracemat=mydata1.values   
                            print('N points, N cells:',tracemat.shape)   
                            self.nframes=len(tracemat)
                            tracemat=tracemat.T
                else:
                    for i in filelist:
                        if i.endswith(".mat"):  # You could also add "and i.startswith('f')
                            f=scipy.io.loadmat(Path + "/"+i)
                            o1=f['Object']
                            aux=o1['Data']
                            aux2=aux[0]
                            trace=aux2[0].flatten()
                            #Trace conditioning will be done in the mat file reading part 2-min basis
                            self.nframes=trace.size #number of total frames in video
                            if nelem == 0:
                                tracemat = numpy.zeros([len(filelist),self.nframes])
                            tracemat[nelem, :] = trace
                            nelem += 1
                #try to plot it
                print("N frames detected from traces:{0}".format(self.nframes))
                taxis=(numpy.arange(self.nframes))/self.sr
                #ax=plt.figure()
                #plotCanv.clear()
                #for i in range(len(filelist)):
                #    plt.plot(taxis, tracemat[i]+10*i)
                    #plotCanv.plot(taxis, tracemat[i]+10*i, pen=(i, nelem))
                    #plotCanv.plot(tracemat[i] + 10 * i, pen=(i, nelem))
                self.Traces = tracemat
                #plt.show()
                if self.ui.checkBox.isChecked():
                    self.loadFilt()

    def loadMats(self):
            self.discardmats=0 #Ignore first 50 mats
            #self.ui.labelResponse.setText("Hello " + self.ui.lineEditName.text())
            #if self.ui.checkBox.isChecked():
            print(os.getcwd().rsplit('\\',1))
            aux1,aux2= os.getcwd().rsplit('\\',1)
            Path = aux1+'\\mats'
            #else:
            #    Path = str(QFileDialog.getExistingDirectory(self, "Select Directory with Movie Indexes"))
            if len(Path) > 1:
                os.chdir(Path)
                #self.ui.label.setText(filen)
                filelist = os.listdir()
                nelem=0
                nframes = numpy.zeros(len(filelist))
                self.starttimes=[]
                self.endtimes=[]
                self.frinmov=[]
                for i in filelist:#Movie indexes could be in .mat or in .xml format
                    if i.endswith(".mat"):
                        f=scipy.io.loadmat(Path + "/"+i,squeeze_me=True)
                        #Getting the start time
                        auxt=i.split("_",2)[-1]
                        auxstartt = time.strptime(auxt,'%Y%m%d_%H%M%S.mat')
                        dt = time.mktime(auxstartt)
                        #dt = time.asctime(auxstartt)
                        self.starttimes.append(dt)
                        #dt = datetime.fromtimestamp(mktime(auxstartt))
                        #use auxstartt.tm_hour, tm_min and tm_sec to retrieve time
                        ob_data1 = f['Object']
                        ob_data2 = ob_data1['TimeFrame'].flatten()
                        ob_data2 = ob_data2[0]#This is an array with the time points of every frame
                        nframes[nelem] = len(ob_data2)
                        self.frinmov.append(len(ob_data2))
                        dur = ob_data2[-1]
                        self.endtimes.append(self.starttimes[nelem] + dur)
                        #dtend = dt + timedelta(seconds=dur)
                        #Now it needs to be converted back to struct...
                        #print(auxstartt)
                        #print(datetime.fromtimestamp(auxstartt))
                        #print(timedelta(seconds=dur))
                        #auxendt = datetime.fromtimestamp(time.asctime(auxstartt)) + timedelta(seconds=dur)
                        nelem += 1

                    if i.endswith(".xml"):
                        header_file = minidom.parse(Path + "/"+i)
                        items = header_file.getElementsByTagName('attr')
                        for elem in items:
                            if elem.attributes['name'].value == 'frames':
                                nframes[nelem] = int(elem.firstChild.data)
                                self.frinmov.append(nframes[nelem])
                            if elem.attributes['name'].value == 'record_start':
                                aux = time.strptime(elem.firstChild.data, '%b %d, %Y %I:%M:%S.%f %p')
                                #dt = datetime.datetime.fromtimestamp(time.mktime(aux))
                                dt = time.mktime(aux)
                                #dt = time.asctime(aux)
                                self.starttimes.append(dt)
                            if elem.attributes['name'].value == 'record_end':
                                aux = time.strptime(elem.firstChild.data, '%b %d, %Y %I:%M:%S.%f %p')
                                #dt = datetime.datetime.fromtimestamp(time.mktime(aux))
                                dt = time.mktime(aux)
                                #dt = time.asctime(aux)
                                self.endtimes.append(dt)
                        nelem += 1
                nframes=nframes
                print("N frames detected from mat or xml files: {0}".format(nframes.sum()))
                #Normalizing traces from every single concatenated movie by its baseline
                print(type(self.Traces))
                print('shape traces',self.Traces.shape)
                print(nframes.sum())
                ntr,npts = self.Traces.shape
                print(ntr,npts)
                self.nframes_from_mats = nframes
                ntraceschunk = npts// 2 * 60 * self.sr  # numbers of chunks to adjust the baseline
                #Conditioning the traces (baseline and median filter for each movie chunk

                # plotCanv.plot(taxis, tracemat[i]+10*i, pen=(i, nelem))
                # plotCanv.plot(tracemat[i] + 10 * i, pen=(i, nelem))

                for ti in range(ntr):
                    k=0
                    for i in nframes:
                        #print (i, type(i))
                        i = int(i)
                        #print(i)
                        if k == 0:
                            vect = self.Traces[ti, 0:i]
                            vect = scipy.signal.medfilt(vect)
                            vect = vect - baseln(vect)
                            vect[vect < 1] = 0
                            #vect = vect - baseln(vect)
                            self.Traces[ti, 0:i] = vect
                        else:
                            if int(nframes[0:k].sum()+i)<=self.Traces.shape[1]:
                                vect = self.Traces[ti, int(nframes[0:k].sum()):int(nframes[0:k].sum()+i)]
                                vect = vect - baseln(vect)
                                vect = scipy.signal.medfilt(vect)
                                vect[vect < 1] = 0 #change to 2
                                self.Traces[ti, int(nframes[0:k].sum()):int(nframes[0:k].sum() + i)] = vect
                        k+=1
                    self.Traces[ti, self.Traces[ti, :]<0] = 0
                    #dividing by std
                    self.Traces[ti,:]/=np.std(self.Traces[ti,:])


                    

                """
                ti=0
                #deleting empty traces
                print(self.Traces.shape)
                while ti<self.Traces.shape[0]:
                    if self.Traces[ti, :].max() < 0.0000001:
                        self.Traces=numpy.delete(self.Traces, (ti), axis=0)
                        ti=0
                    else:
                        ti+=1
                print(self.Traces.shape)
                ntr, npts = self.Traces.shape
                self.Ncells=self.Traces.shape[0]"""
                #getting the pairwise correlation among all traces for each movie:
                #meancorr = []
                #list_comb = tuple(itertools.combinations(range(ntr),2))
                """#print(list_comb)
                for n,i in enumerate(nframes):
                    index = numpy.arange(int(sum(nframes[0:n])),int(sum(nframes[0:n+1])))
                    for x, y in list_comb:
                        if (self.Traces[x, index].max()>0.0000000001 and self.Traces[y, index].max()) > 0.000000001:
                            popcor=numpy.asarray(numpy.corrcoef(self.Traces[x,index],self.Traces[y,index])[0,1])
                            if numpy.isnan(popcor):
                                print('nan detected',self.Traces[x, index].max(),self.Traces[y, index].max())
                    print(n)
                    meancorr.append(popcor.mean())

                    #print(n, numpy.asarray(popcor).mean())"""
                #plotting traces
                plotCanv = self.ui.PlotWidget_tr
                taxis = (numpy.arange(self.nframes)) / self.sr
                print('taxis:',taxis.shape)
                print('traces:',self.Traces.shape)
                plotCanv.clear()
                for i in range(ntr):
                    plotCanv.plot(taxis, self.Traces[i] + 10 * i, pen=(i, ntr))
                #plot timings in hypnogram
                plotsc = self.ui.PlotWidget2
                vones = numpy.ones(numpy.shape(self.starttimes[self.discardmats:])) * 2.5
                plotsc.plot(self.starttimes[self.discardmats:], vones, pen=None, symbol='o', symbolPen=None, symbolSize=5, symbolBrush=(0,0,220))
                plotsc.plot(self.endtimes[self.discardmats:], vones, pen=None, symbol='o', symbolPen=None, symbolSize=5, symbolBrush=(0,220,0))

                #making isolated plot:
                plt.figure()
                plt.rcParams['font.size'] = 14
                ax=plt.subplot(1,1,1)
                #plt.subplot(2,1,1)
                for tri in range(10):
                    plt.plot(taxis, self.Traces[tri,:]+ 10 * tri,linewidth=0.5)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.tick_params(left=False)
                ax.tick_params(labelleft=False)
                plt.ylabel('Z-Score')
                plt.xlabel('Time (s)')
                #plt.subplot(2, 1, 2)
                #plt.plot(range(len(meancorr)),meancorr,'ko-')
                plt.show()
                #os.chdir('..')
                #pickle.dump([taxis, self.Traces.mean(axis=0),meancorr ], open("save.p", "wb"))"""
            if self.ui.checkBox.isChecked():
                self.loadScore()

    def loadFilt(self):
        if self.ui.checkBox.isChecked():
                print(os.getcwd().rsplit('\\',1))
                aux1,aux2= os.getcwd().rsplit('\\',1)
                Path = aux1+'\\filters'
        else:
            Path = str(QFileDialog.getExistingDirectory(self, "Select Directory with filters"))
        #os.chdir('F:/data2018and2019/nnOS grant/nnos_imaging/mse662_12_ctx_camK')
        #Path = str(QFileDialog.getExistingDirectory(self, "Select Directory with Filters"))
        if len(Path) > 1:
            os.chdir(Path)
            filelist = os.listdir()
            nelem = 0
            thresf= float(self.ui.lineEdit.text())
            for i in filelist:
                if self.ui.checkBox_2.isChecked(): #loading tiff files
                    if np.max([c.strip() in i for c in self.colnames]):
                        im = Image.open(i)
                        matrixobj = np.array(im)
                        #print('shapeimg:',matrixobj.shape)
                        matrixobj = matrixobj / matrixobj.max()
                        matrixobj[matrixobj < thresf] = 0  # This needs to be adjusted by hand one by one
                        nrows, ncols = matrixobj.shape
                        kernel = numpy.ones((3, 3))
                        matrixobj = cv2.erode(matrixobj, kernel, iterations=1)
                        if nelem == 0:
                            allmatrix = numpy.zeros([len(filelist), nrows, ncols])
                        allmatrix[nelem, :, :] = matrixobj
                        nelem += 1

                    else:
                        print(i,' not having ',self.colnames[0])
                    
                else:
                    if i.endswith(".mat"):
                        #print("opening " + i)
                        f = scipy.io.loadmat(Path + "/" + i)
                        o1 = f['Object']
                        aux = o1['Data']
                        matrixobj = aux[0, 0]
                        #print('shapeimg:',matrixobj.shape)
                        matrixobj = matrixobj / matrixobj.max()
                        matrixobj[matrixobj < thresf] = 0  # This needs to be adjusted by hand one by one
                        nrows, ncols = matrixobj.shape
                        kernel = numpy.ones((3, 3))
                        matrixobj = cv2.erode(matrixobj, kernel, iterations=1)
                        if nelem == 0:
                            allmatrix = numpy.zeros([len(filelist), nrows, ncols])
                        allmatrix[nelem, :, :] = matrixobj
                        nelem += 1
                # try to plot it
            mat2plot = numpy.amax(allmatrix, 0)
            mat2plot = (mat2plot / mat2plot.max())
            self.ui.mpl.canvas.axes.clear()
            #self.ui.mpl.canvas.axes.plt.imshow(mat2plot)
            self.ui.mpl.canvas.axes.imshow(mat2plot)
            self.ui.mpl.canvas.axes.axis("off")
            self.ui.mpl.canvas.draw()
            self.Ncells = nelem
            #iv = self.ui.ImgView
            #iv.setImage(mat2plot)
            self.filters = allmatrix
            self.filtpath = Path
            self.mat2plot=mat2plot
            if self.ui.checkBox.isChecked():
                    self.loadMats()


    def loadScore(self):

        if self.ui.checkBox.isChecked():
            os.chdir("../EDF")
            #looks in the current folder for the first .mat file and uses that as the score
            filelist2 = os.listdir()
            print(filelist2)
            for item in filelist2:
                if item.endswith(".mat"):
                    fileName = item
                    break
        else:
            fileName = QFileDialog.getOpenFileName(self, 'Open scoring file', '', "mat files (*.mat)")
            fileName = fileName[0]
        if len(fileName) >= 1:
            [id_animal,kk] = fileName.split(".")
            self.ui.label_5.setText(id_animal)
            #f = scipy.io.loadmat(fileName)
            with h5py.File(fileName, 'r') as f:
                auxsc=[x[0] for x in np.array(f['sc'])]
                self.score=auxsc
                #self.score=auxsc[0].flatten()
                zt0="".join([chr(value) for value in [x[0] for x in np.array(f['zt0'])]])
                self.zt0=zt0
                print('ZT0:',self.zt0)
                self.t0="".join([chr(value) for value in [x[0] for x in np.array(f['t0'])]])
                #self.t0 = str(f['t0'][0])#time of first data point
                print('T0:',self.t0)
                self.epochl=np.array(f['epocl'])[0][0]#f['epocl']
            # aux2el = aux1el[0]
            # self.epochl = aux2el[0]
            #print('Epoch length:',self.epochl)
            plotsc = self.ui.PlotWidget2
            #print(self.starttimes[0])
            edfstart = time.strptime(self.t0, '%d_%m_%Y_%H:%M:%S')
            #print(edfstart)
            dt = time.mktime(edfstart)
            #print(dt)
            self.timesscore=[]
            self.timesscore.append(dt)
            for i in range(len(self.score)):
                if i>0:
                    self.timesscore.append(self.timesscore[i-1]+float(self.epochl))

            print('len timescore:',len(self.timesscore))
            #taxis = #time of each epoch//(numpy.arange(len(self.score))) * self.epochl
            plotsc.plot(self.timesscore, self.score)
        #also plotting in the traces window (requires sync)
        #taxis = (numpy.arange(self.nframes)) / self.sr
        #plotCanv = self.ui.PlotWidget_tr
        #ampFactor = self.sr * self.epochl
        #hrscoring = ampfun(self.score, ampFactor)
        #plotCanv.plot(taxis, hrscoring,  pen='k')
        self.loadSync()


    def loadSync(self):#read the TTL signal with the frame acquisitions
        #adust the times on the matfiles to match the first frame of every movie
        #to do: add the period and start, end times
        if self.ui.checkBox.isChecked():
            # looks in the current folder for the first .edf file and uses that for sync with the selected channel
            filelist = os.listdir()
            for item in filelist:
                if item.endswith(".edf"):
                    fileName = item
                    break
        else:
            fileName = QFileDialog.getOpenFileName(self, 'Open EDF with sync data', '', "EDF files (*.edf)")
            fileName = fileName[0]
        ontimes=numpy.zeros(len(self.starttimes))#this uses the info from the mat files
        offtimes = numpy.zeros(len(self.starttimes))
        difimg=10#10 s at least between imaging times
        if len(fileName) >= 1:
            edf = mne.io.read_raw_edf(fileName)
            #print('ready reading edf')
            synchan=self.ui.sync_channel.value()-1
            #print("opening channel {0}".format(synchan))
            sync = numpy.asarray(edf.get_data(synchan).flatten())
            if self.ui.checkBox_3.isChecked():
                plt.figure()
                plt.plot(sync)
                plt.show()
            self.EEG =numpy.asarray(edf.get_data(0).flatten())
            midpoint =0.3 *(sync.max() + sync.min())
            shiftsync, aux0 = numpy.zeros(len(sync)), numpy.zeros(len(sync))
            shiftsyncn = aux0
            shiftsync[1:len(sync)] = sync[0:len(sync)-1]
            shiftsyncn[0:len(sync)-1] = sync[1:len(sync)]
            c = (sync>midpoint) & (shiftsync<midpoint) & (shiftsyncn<midpoint)
            sync[c.flatten()] = 0
            shiftsync[1:len(sync)] = sync[0:len(sync) - 1] #this goes one point behind
            cond1 = shiftsync > midpoint
            cond2 = sync < midpoint
            #cond3 = (supershiftsync1 >midpoint) & (supershiftsync2 >midpoint)
            cond4 = sync >= midpoint
            cond5 = shiftsync <= midpoint
            index_endimage = cond1 & cond2 #& cond3
            index_endimage = index_endimage.flatten()
            print('start',index_endimage.sum())
            index_image = cond4 & cond5 #& (supershiftsync1 <=midpoint) & (supershiftsync2 <=midpoint)
            index_image = index_image.flatten()
            print('end',index_image.sum())
            imgTimes = edf.times[index_image] + self.timesscore[0]
            imgendTimes = edf.times[index_endimage] + self.timesscore[0]
            vones=numpy.ones(numpy.shape(imgTimes))*midpoint#2.5
            vonese = numpy.ones(numpy.shape(imgendTimes)) * midpoint*0.9#2.49990
            #print(numpy.shape(sync))
            #print(type(sync))
            #plt.scatter(imgTimes, vones, marker='.')
            plotsc = self.ui.PlotWidget2
            #plotsc.clear()
            # plotsc.plot(edf.times + self.timesscore[0], sync)
            # plotsc.plot(imgTimes, vones, pen=None, symbol='o',symbolPen='r', symbolSize=4)
            # plotsc.plot(imgendTimes, vonese, pen=None, symbol='s', symbolPen='b', symbolSize=4)
            # plt.figure()
            # plt.plot(edf.times + self.timesscore[0], sync)
            # plt.plot(imgTimes, vones, 'o')
            # plt.plot(imgendTimes, vonese, 's')
            # plt.show()

            print("Done.")
            print("N frames detected from onset:{0}".format(len(imgTimes)))
            print("N frames detected from offset:{0}".format(len(imgendTimes)))
            # finding the first and last pulse for every movie
            ontimes[0] = imgTimes[0]
            k = 0
            for i in range(len(imgTimes)):
                if i>0:
                    if imgTimes[i]-imgTimes[i-1] > difimg:
                        k+=1
                        ontimes[k]=imgTimes[i]
            print('nmovies from ontimes:',k+1)
            k = 0
            for i in range(len(imgendTimes)-1):#The last one will be made by hand
                    if imgTimes[i+1] - imgTimes[i] > difimg:
                        offtimes[k] = imgendTimes[i]
                        k += 1
            offtimes[k] = imgendTimes[-1]
            print('nmovies from offtimes:', k + 1)
            if len(ontimes)!= len(offtimes):
                print('ERROR: Problem in the detection of on  and off times')
            if len(self.starttimes) != len(ontimes):
                print('ERROR: Mismatch between startimes (mats) and ontimes (sync)')
            if self.nframes != self.nframes_from_mats.sum():
                print('ERROR: Mismatch between nframes traaces and info in mat files')
            #identifying chunks in the hypnogram
            chunk_times_score =[]
            self.chunk_score=[]
            atimesc = numpy.asarray(self.timesscore)
            ascore = numpy.asarray(self.score)
            for k in range(len(ontimes)):
                indx = (atimesc >= ontimes[k]) & (atimesc <= offtimes[k])
                chunk_times_score = numpy.append(chunk_times_score,atimesc[indx])
                self.chunk_score = numpy.append(self.chunk_score,ascore[indx])
            voneson = numpy.ones(numpy.shape(ontimes)) * midpoint*1.2  # 2.5
            vonesoff = numpy.ones(numpy.shape(offtimes)) * midpoint * 1.4  # 2.5
            #vonesoff = numpy.ones(numpy.shape(offtimes2)) * midpoint * 1.2
            plotsc.plot(ontimes, voneson, pen=None, symbol='o', symbolPen='k', symbolSize=4)
            plotsc.plot(offtimes, vonesoff, pen=None, symbol='s', symbolPen='m', symbolSize=4)
            """Make a time vector with the same number of point than the Ca traces, aligned with the EEG timing."""
            self.timeTraces = numpy.zeros(self.nframes)
            print(self.nframes_from_mats[0])
            #idetifiying time points corresponding to first w epoch
            indxfw = []
            indxow = []
            for i in range(1,len(self.score)):
                if int(self.score[i]) == 0:
                    if int(self.score[i-1]) != 0: #first W epoch
                        indxfw.append(atimesc[i])
                    else:
                        indxow.append(atimesc[i])
            for i in range(len(self.nframes_from_mats)):
                step = (offtimes[i]-ontimes[i])/self.nframes_from_mats[i]
                aux = numpy.arange(ontimes[i],offtimes[i],step)
                aux = aux[0:int(self.nframes_from_mats[i])]
                if i==0:
                    self.timeTraces[0:int(self.nframes_from_mats[i])] = aux
                else:
                    #print(len(self.timeTraces[int(self.nframes_from_mats[0:i].sum()):int(self.nframes_from_mats[0:i+1].sum())] ))
                    #print(len(aux))
                    self.timeTraces[int(self.nframes_from_mats[0:i].sum()):int(self.nframes_from_mats[0:i+1].sum())] = aux
            print(len(self.Traces))
            timeW = chunk_times_score[numpy.round(self.chunk_score) == 0.0]
            timeN = chunk_times_score[numpy.round(self.chunk_score) == 1.0]
            timeR = chunk_times_score[numpy.round(self.chunk_score) == 2.0]
            #make arrays with the ca values for every state
            self.activityW = -1 * numpy.ones([numpy.shape(self.Traces)[0], max(len(timeW),1)])
            self.activityN = -1 * numpy.ones([numpy.shape(self.Traces)[0], max(len(timeN),1)])
            self.activityR = -1 * numpy.ones([numpy.shape(self.Traces)[0], max(len(timeR),1)])
            self.fwa = numpy.zeros([self.Ncells,len(indxfw)])
            self.owa = numpy.zeros([self.Ncells, len(indxow)])
            c = 0
            pf=0
            po=0
            for t in timeW:
                index = (self.timeTraces >= t) & (self.timeTraces < (t+self.epochl))
                if index.max():
                    self.activityW[:,c] = self.Traces[:, index].mean(axis=1) #calculate the mean Z score per epoch of that state
                if t in indxfw:
                    self.fwa[:,pf]=self.activityW[:,c]
                    pf+=1
                else:
                    self.owa[:, po] = self.activityW[:, c]
                    po += 1
                c += 1
            newEDFt = edf.times + self.timesscore[0]
            swa=[]
            meancorr=[]
            #Calculating the mean pairwize correlaton and the delta power for every epoch during NR
            c = 0
            sf_EEG = 1/(edf.times[2]-edf.times[1])
            for t in timeN:
                index = numpy.where((self.timeTraces >= t) & (self.timeTraces < (t + self.epochl)))[0]
                if len(index)>0:
                    #print('getting cross corr...',t)
                    aux = self.Traces[:, index].mean(axis=1)
                    self.activityN[:, c] = aux
                    i=numpy.where(numpy.asarray([self.activityN[:, c]>0]).flatten())[0]
                    if len(i) > 0 and len(index) > 1:
                        auxtr0 = self.Traces[:,index]
                        auxtr=auxtr0[i,:]
                        #popcor=[numpy.corrcoef(auxtr[x[0],:],auxtr[x[1],:])
                        #       for x in list(itertools.combinations(range(auxtr.shape[0]),2))]
                        #meancorr.append(numpy.asarray(popcor).mean())
                    #else:
                        #meancorr.append(0)
                    if len(self.traces_NR)>0:
                        self.traces_NR = numpy.concatenate((self.traces_NR,self.Traces[:, index]),axis=1)
                    else:
                        self.traces_NR = self.Traces[:, index]

                #Now getting the EEG parts that fall in the time limit
                # indexeeg = (newEDFt >= t) & (newEDFt < (t + self.epochl))
                # if indexeeg.max():
                #     swa.append(get_delta(self.EEG[indexeeg],sf_EEG))
                #     if len(self.EEG_NR)>0:
                #         self.EEG_NR = numpy.concatenate((self.EEG_NR,self.EEG[indexeeg]))
                #     else:
                #         self.EEG_NR = self.EEG[indexeeg]
                c += 1
            c = 0

            for t in timeR:
                index = (self.timeTraces >= t) & (self.timeTraces < (t + self.epochl))
                if index.max():
                    self.activityR[:, c] = self.Traces[:, index].mean(axis=1)
                else:
                    self.activityR[:, c] = 0
                c += 1
            self.activityR[self.activityR == -1] = 0
            print('Meancorr=',meancorr)
            print('SWA=',swa)
            # f=plt.figure(1)
            # plt.subplot(2,1,1)
            # print(self.sr)
            # teeg= numpy.arange(0,len(self.EEG_NR))/sf_EEG
            # print(self.traces_NR.shape)
            # ttraces = numpy.arange(0,self.traces_NR.shape[1])/15
            # plt.plot(teeg,self.EEG_NR)
            # #plt.xlim((1630,1631))
            # plt.subplot(2, 1, 2)
            # plt.plot(ttraces,self.traces_NR.sum(axis=0))
            # plt.plot(ttraces, self.traces_NR.T)
            # #plt.xlim((1630,1631))
            # f.show()
            #g = plt.figure()
            #xvect=numpy.arange(len(meancorr))*self.epochl
            #f=open('mcorr.txt','w')
            #f.write(",".join([str(v) for v in list(meancorr)]))
            #f.close()
            #f = open('times.txt', 'w')
            #f.write(",".join([str(v) for v in list(timeN-timeN[0])]))
            #f.close()

            #plt.plot(timeN,meancorr)
            #print(numpy.corrcoef(meancorr,swa))
            #plotCanv = self.ui.PlotWidget_tr
            #plotCanv.clear()
            #for i in range(len(self.Traces)):
            #    plotCanv.plot(self.timeTraces, self.Traces[i] + 10 * i, pen=(i, len(self.Traces)))
            # #plotCanv.plot(chunk_times_score, chunk_score *10, pen='k')
            # plt.plot(self.activityW[1,:])
            # plt.plot(self.activityN[1,:])
            # plt.plot(self.activityR[1,:])
            # plt.plot(activityW.mean(axis=1))
            # plt.plot(activityN.mean(axis=1))
            # plt.plot(activityR.mean(axis=1))
            # plt.ylabel('activityW.mean')
            # plt.show()
            self.mW , self.mN, self.mR = self.activityW.mean(axis=1), self.activityN.mean(axis=1), self.activityR.mean(axis=1)
            #print(self.mW[0] , self.mN[0], self.mR[0])
            self.updateFilters()
            #plot triangle for legend
            matlegend = numpy.zeros([1000,1000,3]) #matrix with a color triangle
            x = numpy.arange(0,1,0.001)


    def saveAll(self):
        #making data frame with the Z scores of every cell for every state, assuming 1 period
        #For mse 105_2 plt: 51 for NREM, 47 for REM and 24 for W
        #For each cell we need a table with the Zscore of an epoch and it's state
        mat_aux=[]
        k0=1
        lstate = ['Wake'] * len(self.activityW[1, :]) + ['NREM'] * len(self.activityN[1, :]) + ['REM'] * len(
            self.activityR[1, :])
        #there can be statistical differences due to larger than the rest for asingle or for 2 states (if there are 3)
        wakeL =[]
        nremL = []
        remL = []
        wakerem=[]
        waknr=[]
        remnr=[]
        mixed=[]
        indep = []
        faL=[]
        oaL=[]
        faS=[]
        pvalw=[]
        pvalrem=[]
        pvalnr=[]
        pwakemix=[]
        pnrmix=[]
        premmix=[]
        pvalind=[]
        pmix=[]
        lfwa1 = int(len(self.fwa)/2)
        lowa1 = int(len(self.owa)/2)
        pref_state=[]
        isrem=False

        if len(self.activityR[1, :]) > 1:
            isrem=True
        ncells_list=[]
        for ti in range(self.Traces.shape[0]):
            if self.Traces[ti,:].max()>0.000001:
                ncells_list.append(ti)
        for n in ncells_list:
            lact=list(self.activityW[n,:]) + list(self.activityN[n,:])+list(self.activityR[n,:])
            #lstate = ('WAKE ' * len(self.activityW)).split()+ ('NREM ' * len(self.activityN)).split() + ('REM ' * len(self.activityR)).split()
            if isrem:
                F, p = stats.f_oneway(self.activityW[n,:],self.activityN[n,:],self.activityR[n,:])
                F, pwn = stats.f_oneway(self.activityW[n, :], self.activityN[n, :])
                F, pwr = stats.f_oneway(self.activityW[n, :], self.activityR[n, :])
                F, pnrr = stats.f_oneway(self.activityN[n, :], self.activityR[n, :])

                #Check for significant differences. If there are, if one is significantly larger than the other two, save it, otherwise save as mix.
                if p<0.05:
                    k0+=1
                    #largest activity for wake 
                    if (self.activityW[n,:].mean()>self.activityN[n,:].mean()) and (self.activityW[n,:].mean()>self.activityR[n,:].mean()):
                        
                        if pwn<0.025 and pwr<0.025: #P val divided by two due to multiple comparisons
                            wakeL.append(n)
                            pref_state.append('W')
                            pvalw.append(p)
                        else:
                            #could be warem or wanr
                            if pwn<0.05:
                                wakerem.append(n)
                                pref_state.append('W-REM')
                            else:
                                if pwr<0.05:
                                    waknr.append(n)
                                    pref_state.append('W-NR')
                            if pwr>0.05 and pwn>0.05:
                                mixed.append(n)
                                pref_state.append('W-Mix')
                                pmix.append(p)
                    #largest activity for NR 
                    elif (self.activityN[n, :].mean() > self.activityW[n, :].mean()) and (
                                    self.activityN[n, :].mean() > self.activityR[n, :].mean()):
                        if pwn < 0.025 and pnrr < 0.025:  # P val divided by two due to multiple comparisons
                            nremL.append(n)
                            pvalnr.append(p)
                            pref_state.append('NR')
                        else:
                            if pnrr<0.05:
                                waknr.append(n)
                                pref_state.append('NR-W')
                            else:
                                if pwn<0.05:
                                    remnr.append(n)
                                    pref_state.append('NR-REM')
                            if pnrr>0.05 and pwn>0.05:
                                mixed.append(n)
                                pmix.append(p)
                                pref_state.append('NR-Mix')

                    elif (self.activityR[n, :].mean() > self.activityW[n, :].mean()) and (
                                    self.activityR[n, :].mean() > self.activityN[n, :].mean()):
                        if pwr<0.025 and pnrr<0.025: #P val divided by two due to multiple comparisons
                            remL.append(n)
                            pvalrem.append(p)
                            pref_state.append('REM')
                        else:
                            if pwr<0.05:
                                remnr.append(n)
                                pref_state.append('REM-NR')
                            else:
                                if pnrr<0.05:
                                    wakerem.append(n)
                                    pref_state.append('REM-W')
                        if pwr>0.05 and pnrr>0.05:
                                mixed.append(n)
                                pmix.append(p)
                                pref_state.append('REM-Mix')
                else:
                    indep.append(n)
                    print('Independent!')
                    pref_state.append('Ind')
                    pvalind.append(p)
                F, p1 = stats.f_oneway(self.fwa[n,0:lfwa1],self.owa[n,0:lowa1]) # Checking only the first half ogf the data
                F, p2 = stats.f_oneway(self.fwa[n, lfwa1:],
                                       self.owa[n, lowa1:])  # Checking only the second half ogf the data
                if (p1<0.05) and (p2<0.05):
                    if (self.fwa[n,0:lfwa1].mean()>self.owa[n,0:lowa1].mean()) and (self.fwa[n,lfwa1:].mean()>self.owa[n,lowa1:].mean()):
                        faL.append(n) #saves the number of the cell that has higher activity at the onset of W
                    else:
                        faS.append(n)
                else:
                    oaL.append(n)
            else:
                #If there is no REM, there can't be mixed states
                F, p = stats.f_oneway(self.activityW[n, :], self.activityN[n, :])
                if p<0.05:
                    print("Significant! N=",k0)
                    k0+=1
                    if (self.activityW[n,:].mean()>self.activityN[n,:].mean()):
                        wakeL.append(n)
                        pref_state.append('W')
                    else:
                        nremL.append(n)
                        pref_state.append('NR')
                else:
                    indep.append(n)
                    pref_state.append('Ind')
                #Checking the activity at onset
                F, p1 = stats.f_oneway(self.fwa[n,0:lfwa1],self.owa[n,0:lowa1]) # Checking only the first half ogf the data
                F, p2 = stats.f_oneway(self.fwa[n, lfwa1:],
                                       self.owa[n, lowa1:])  # Checking only the second half ogf the data
                if (p1<0.05) and (p2<0.05):
                    if (self.fwa[n,0:lfwa1].mean()>self.owa[n,0:lowa1].mean()) and (self.fwa[n,lfwa1:].mean()>self.owa[n,lowa1:].mean()):
                        faL.append(n) #saves the number of the cell that has higher activity at the onset of W
                    else:
                        faS.append(n)
                else:
                    oaL.append(n)
        # print("percentage of active cells at onset: ",100*len(faL)/self.Ncells)
        # print("percentage of less active cells at onset: ", 100 * len(faS) / self.Ncells)
        # print("percentage of cells indifferent to onset: ", 100 * len(oaL) / self.Ncells)
        #mixed = wakeS +nremS +remS
        if len(wakeL +nremL +remL +mixed +indep ) != self.Ncells:
            print("missing cells!!")
        #Making final figure with traces and hypnogram, jellybeans and pie chart
        labels =[]
        sizes =[]
        explode = []
        colors =[]
        colorlist = (0,1,0), 'r', 'b', (0,1,1),(1,1,0),(1,0,1),'w',(0.5,0.5,0.5)
        labellist = 'W', 'R', 'NR', 'W-NR','W-REM','REM-NR','Mixed', 'Ind'
        i=0
        for i,m in enumerate([wakeL,remL, nremL,waknr,wakerem,remnr,mixed,indep]):
            if len(m)>0:
                labels.append(labellist[i]+ '(N='+str(len(m))+')')
                sizes.append(100*len(m)/self.Ncells)
                explode.append(0)
                colors.append(colorlist[i])

        #To do: add first vs other activity during W
        #self.foa

        #making summary figure with jellybeans,
        fig=plt.figure()
        plt.rcParams['font.size'] = 14
        grid = plt.GridSpec(2, 3, wspace=0.0, hspace=0.1)
        plt.subplot(grid[0, 0])
        plt.imshow(1.4*self.mat2plot)
        plt.axis("off")
        plt.draw()
        ax1 = plt.subplot(grid[0, 1])
        ax1.pie(sizes, explode=explode, labels=labels, radius=0.6,labeldistance=0.8,colors=colors, autopct='%1.1f%%',
                    shadow=False, textprops={'weight':'bold'}, startangle=90,
                    wedgeprops={"edgecolor":"k",'linewidth': 1,'antialiased': True})
        
        # ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        #             shadow=False, textprops={'size': 'x-large', 'weight':'bold'}, startangle=90,
        #             wedgeprops={"edgecolor":"k",'linewidth': 1,'antialiased': True})
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        ax=plt.subplot(grid[0, 2]) #plotting bar plot with the difference in activity for cells who had larger activity during hte first W bout
        meanw,meann,meanr = self.activityW.mean(axis=1), self.activityN.mean(axis=1),self.activityR.mean(axis=1) #mean activity for every cell
        error = [(0,0,0),[stats.sem(meanw),stats.sem(meann), stats.sem(meanr)]]
        bp=plt.bar([0,1,2], [meanw.mean(),meann.mean(),meanr.mean()],
                   yerr=error,align='center',alpha=1, ecolor='k',capsize=5)
        plt.xticks([0, 1,2], ('WAKE', 'NREM','REM'))
        plt.ylabel('Mean Z score')
        bp[0].set_color((0,1,0))
        bp[1].set_color('b')
        bp[2].set_color('r')
        plt.tight_layout()
        # plt.bar([0,1],[self.fwa[faL, :].mean(), self.owa[faL, :].mean()])
        # plt.xticks([0,1], ('Onset W', 'Within W'))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        #plt.box(on=None)
        ax = plt.subplot(grid[1, 0:])
        #now plotting hypnogram and traces

        tracecol = 1.4* numpy.asarray([self.mR.T, self.mW.T, self.mN.T])
        tracecol = tracecol/tracecol.max()
        tscore =np.array([t*self.epochl for t in range(len(self.chunk_score))])
        taxis = (numpy.arange(self.nframes)) / self.sr
        plt.plot(tscore/3600, self.chunk_score*4.5,'k-')
        plt.xlabel('Time (h)',fontsize=15)
        locs, labels = plt.yticks() 
        plt.yticks([0,4.5,9], ['W', 'NR', 'REM'])  # Set text labels.
        #adding the typical best for every class:
        indxt = []#numpy.arange(self.Ncells)
        if len(pvalw)>0:
            indxt.append(wakeL[pvalw.index(min(pvalw))])
            pvalw[pvalw.index(min(pvalw))]=100
            indxt.append(wakeL[pvalw.index(min(pvalw))])
            pvalw[pvalw.index(min(pvalw))] = 100
            indxt.append(wakeL[pvalw.index(min(pvalw))])
            #pvalw[pvalw.index(min(pvalw))] = 100
            #indxt.append(wakeL[pvalw.index(min(pvalw))])
        if len(pvalrem)>0:
            indxt.append(remL[pvalrem.index(min(pvalrem))])
            pvalrem[pvalrem.index(min(pvalrem))]=100
            indxt.append(remL[pvalrem.index(min(pvalrem))])
            #pvalrem[pvalrem.index(min(pvalrem))] = 100
            #indxt.append(remL[pvalrem.index(min(pvalrem))])
            #pvalrem[pvalrem.index(min(pvalrem))] = 100
            #indxt.append(remL[pvalrem.index(min(pvalrem))])
        if len(pvalnr)>0:
            indxt.append(nremL[pvalnr.index(min(pvalnr))])
            #pvalnr[pvalnr.index(min(pvalnr))]=100
            #indxt.append(nremL[pvalnr.index(min(pvalnr))])
        if len(pmix)>0:
            indxt.append(mixed[pmix.index(min(pmix))])
            pmix[pmix.index(min(pmix))] = 100
            indxt.append(mixed[pmix.index(min(pmix))])

        if len(pvalind)>0:
            indxt.append(indep[pvalind.index(min(pvalind))])
            pvalind[pvalind.index(min(pvalind))]=100
            indxt.append(indep[pvalind.index(min(pvalind))])
        indxt=[24,51,47,48]
        for n,i in enumerate(indxt):
            plt.plot(taxis/3600, self.Traces[i,:] + 10 * (n+1),linewidth=0.5,color=tuple(tracecol[:,i]))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        #ax.tick_params(left=False)
        #ax.tick_params(labelleft=False)
        #plt.box(on=None)
        #ax.set_frame_on(False)
        #ax.add_axes([0., 1., 1., 0])
        #ax = plt.axes([0, 1., 0, 1.])
        #ax.get_xaxis().set_visible(True)
        #ax.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.show()
        #Make dataframe with summary ctivity of each cell and saveit as CSV
        #Mean activity for ech state and preference
        df = pd.DataFrame(columns=['Cell_ID','Experiment','Duration','hasREM','W_activity','NR_activity','REM_activity','Preference'])
        for n in ncells_list:
            df.loc[n]=[n,self.ui.label_5.text(),(self.timeTraces[-1]-self.timeTraces[0])/60,isrem,np.mean(self.activityW[n,:]),
                np.mean(self.activityN[n,:]),np.mean(self.activityR[n,:]),pref_state[n]]
        df.to_csv(self.ui.label_5.text()+'.csv',index=False)
        print('Summary saved in '+self.ui.label_5.text()+'.csv')

        """
        
        plotCanv.clear()
        for i in range(len(filelist)):
            plotCanv.plot(taxis, tracemat[i] + 10 * i, pen=(i, nelem))
        self.Traces = tracemat
                #Plot hypnogrsm during imaging and 10 random cell traces
"""


        #print("len lact,lstate =",len(lact),len(lstate))

        # plt.plot(self.activityW[1,:])
        # plt.plot(self.activityN[1,:])
        # plt.plot(self.activityR[1,:])
        # plt.ylabel('Z score per epoch')
        # plt.show()
        # plt.close()
        #print(type(self.activityW))
        #print("shape activityw =",self.activityW.shape())
        #print("lengths = ",len(self.activityW[1,:]),len(self.activityN[1,:]),len(self.activityR[1,:]))
        """d = {'Z_score': lact, 'State': lstate}
        df = pd.DataFrame(d)
        dfw = df[df['State'] == 'Wake']
        dfn = df[df['State'] == 'NREM']
        dfr = df[df['State'] == 'REM']
        print(dfw)


        #print(df.head())
        # Create a boxplot
        grps = pd.unique(df.State.values)
        d_data = {grp: df['Z_score'][df.State == grp] for grp in grps}
        print(d_data)
        k = len(pd.unique(df.State))  # number of conditions
        N = len(df.values)  # conditions times participants
        n = df.groupby('State').size()[0]  # Participants in each condition
        F, p = stats.f_oneway(d_data.values())
        #F, p = stats.f_oneway(dfw.['z_score'].values,dfn['z_score'].values,dfr['z_score'].values)
        if p<0.05:
            print("Significant! N=",k0)
            K0+=1
"""






if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())