import sys, os, numpy
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QGraphicsScene, QGraphicsPixmapItem
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import time
#from datetime import datetime, timedelta
from caimager6 import *
#from matplotlib import pyplot as plt
import scipy.io
#from PyQt5.QtGui import QIcon, QPixmap, QImage
import mne
from xml.dom import minidom
import scipy.signal
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

"""If there is EEG, traces need to by synced with EEG. If traces include dropped frames, the they should match 
the N frames from the mat files, else they should match the nu,ber of detected frames form the sync signal.
A vector with the time of each frame will be generated, matching the EEG time of the first frame in every chunk.
Convert ui with pyuic5 demoLineEdit.ui -o demoLineEdit.py"""

#auxilliary classes:
def ampfun(data, ampFactor):  # function that amplifies a signalwith low temporal res
    data = data.flatten()
    newvector = numpy.zeros(len(data) * ampFactor)
    for i in range(len(data) + 1):
        newvector[ampFactor * (i - 1):ampFactor * i] = data[i - 1]
    return (newvector)

def baseln(data):#gets the baseline as 10% of the smallest point
    l=len(data)
    data.sort()
    return(data[int(0.1*l)])


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
        #specify all the controls here:
        self.ui.pushButton.clicked.connect(self.loadFilt)
        self.ui.ImgView.ui.roiBtn.hide()
        self.ui.ImgView.ui.histogram.hide()
        self.ui.ImgView.ui.menuBtn.hide()
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

        #now def all the functions
    def updateNperiods(self):
        self.ui.current_period.maximum = self.ui.Nperiods.value()

    def updateFilters(self):
        os.chdir(self.filtpath)
        print(self.filtpath)
        filelist = os.listdir()
        nelem = 0
        thresf = float(self.ui.lineEdit.text())
        for i in filelist:
            if i.endswith(".mat"):
                # print("opening " + i)
                f = scipy.io.loadmat(self.filtpath + "/" + i)
                o1 = f['Object']
                aux = o1['Data']
                matrixobj = aux[0, 0]
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
        mat2plot = 256 * (mat2plot / mat2plot.max())
        iv = self.ui.ImgView
        iv.clear()
        [xres,yres] = numpy.shape(mat2plot)
        colormat= numpy.zeros([xres, yres, 3])
        for x in range(xres):
            for y in range (yres):
                colormat[x,y,:] = [0.1*(mat2plot[x,y])/256,0.5 * mat2plot[x,y]/256, 0.9 * mat2plot[x,y]/256]
        #print(numpy.shape(colormat))
        #type(colormat)
        #iv.setImage(mat2plot)
        #iv.setImage(colormat)
        iv.image(colormat)
        self.filters = allmatrix
        #plt.imshow(colormat)
        #plt.axis("off")
        #plt.show()


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
            #self.ui.labelResponse.setText("Hello " + self.ui.lineEditName.text())
            if self.ui.checkBox.isChecked():
                print(os.getcwd().rsplit('\\',1))
                aux1,aux2= os.getcwd().rsplit('\\',1)
                Path = aux1+'\\traces'
            else:
                Path = str(QFileDialog.getExistingDirectory(self, "Select Directory with Traces"))
            if len(Path) > 1:
                os.chdir(Path)
                #self.ui.label.setText(filen)
                filelist = os.listdir()
                nelem=0
                plotCanv = self.ui.PlotWidget_tr
                for i in filelist:
                    if i.endswith(".mat"):  # You could also add "and i.startswith('f')
                        #print("opening "+i)
                        f=scipy.io.loadmat(Path + "/"+i)
                        o1=f['Object']
                        aux=o1['Data']
                        aux2=aux[0]
                        trace=aux2[0].flatten()
                        #Trace conditioning: median filter, eliminate negativesd and substract baseline on a 2-min basis
                        trace = scipy.signal.medfilt(trace)
                        trace[trace < 0] = 0
                        self.nframes=trace.size #number of total frames in video
                        if nelem == 0:
                            tracemat = numpy.zeros([len(filelist),self.nframes])
                        tracemat[nelem, :] = trace
                        nelem += 1
                #try to plot it
                print("N frames detected from traces:{0}".format(self.nframes))
                taxis=(numpy.arange(self.nframes))/self.sr
                plotCanv.clear()
                for i in range(len(filelist)):
                    plotCanv.plot(taxis, tracemat[i]+10*i, pen=(i, nelem))
                self.Traces = tracemat
                if self.ui.checkBox.isChecked():
                    self.loadMats()

    def loadMats(self):
            #self.ui.labelResponse.setText("Hello " + self.ui.lineEditName.text())
            if self.ui.checkBox.isChecked():
                print(os.getcwd().rsplit('\\',1))
                aux1,aux2= os.getcwd().rsplit('\\',1)
                Path = aux1+'\\mats'
            else:
                Path = str(QFileDialog.getExistingDirectory(self, "Select Directory with Movie Indexes"))
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
                print("N frames detected from mat or xml files: {0}".format(nframes.sum()))
                #Normalizing traces from every single concatenated movie by its baseline
                print(type(self.Traces))
                ntr,npts = self.Traces.shape
                print(ntr,npts)
                self.nframes_from_mats = nframes
                ntraceschunk = npts// 2 * 60 * self.sr  # numbers of chunks to adjust the baseline
                for ti in range(ntr):
                    k=0
                    for i in nframes:
                        #print (i, type(i))
                        i = int(i)
                        if k == 0:
                            vect = self.Traces[ti,0:i]-baseln(self.Traces[ti,0:i])
                            self.Traces[ti, 0:i] = vect
                        else:
                            vect = self.Traces[ti, nframes[0:k].sum():nframes[0:k].sum()+i]
                            vect = vect - baseln(vect)
                            self.Traces[ti, nframes[0:k].sum():nframes[0:k].sum() + i] = vect

                        #print(vect.shape)
                #plotting traces
                plotCanv = self.ui.PlotWidget_tr
                taxis = (numpy.arange(self.nframes)) / self.sr
                plotCanv.clear()
                for i in range(ntr):
                    plotCanv.plot(taxis, self.Traces[i] + 10 * i, pen=(i, ntr))

                #plot timings in hypnogram
                plotsc = self.ui.PlotWidget2
                vones = numpy.ones(numpy.shape(self.starttimes)) * 2.5
                plotsc.plot(self.starttimes, vones, pen=None, symbol='o', symbolPen=None, symbolSize=5, symbolBrush=(0,0,220))
                plotsc.plot(self.endtimes, vones, pen=None, symbol='o', symbolPen=None, symbolSize=5, symbolBrush=(0,220,0))
            if self.ui.checkBox.isChecked():
                self.loadScore()

    def loadFilt(self):
        Path = str(QFileDialog.getExistingDirectory(self, "Select Directory with Filters"))
        if len(Path) > 1:
            os.chdir(Path)
            filelist = os.listdir()
            nelem = 0
            thresf= float(self.ui.lineEdit.text())
            print(thresf)
            for i in filelist:
                if i.endswith(".mat"):
                    #print("opening " + i)
                    f = scipy.io.loadmat(Path + "/" + i)
                    o1 = f['Object']
                    aux = o1['Data']
                    matrixobj = aux[0, 0]
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
            mat2plot = 256 * (mat2plot / mat2plot.max())
            iv = self.ui.ImgView
            iv.setImage(mat2plot)
            self.filters = allmatrix
            self.filtpath = Path
            if self.ui.checkBox.isChecked():
                self.loadTraces()


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
            f = scipy.io.loadmat(fileName)
            auxsc=f['sc']
            self.score=auxsc[0].flatten()
            self.zt0=f['zt0'][0]
            print('ZT0:',self.zt0)
            self.t0 = str(f['t0'][0])#time of first data point
            print('T0:',self.t0)
            aux1el=f['epocl']
            aux2el = aux1el[0]
            self.epochl = aux2el[0]
            #print('Epoch length:',self.epochl)
            plotsc = self.ui.PlotWidget2
            #print(self.starttimes[0])
            edfstart = time.strptime(self.t0, '%d_%m_%Y_%H:%M:%S')
            #print(edfstart)
            dt = time.mktime(edfstart)
            #print(dt)
            self.timesscore.append(dt)
            for i in range(len(self.score)):
                if i>0:
                    self.timesscore.append(self.timesscore[i-1]+float(self.epochl))

            print(len(self.timesscore))
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
            print('ready reading edf')
            synchan=self.ui.sync_channel.value()-1
            print("opening channel {0}".format(synchan))
            sync = numpy.asarray(edf.get_data(synchan).flatten())
            midpoint =0.3 *(sync.max() + sync.min())
            print(midpoint,sync.max(),sync.min())
            shiftsync, aux0 = numpy.zeros(len(sync)), numpy.zeros(len(sync))
            shiftsyncn = aux0
            shiftsync[1:len(sync)] = sync[0:len(sync)-1]
            shiftsyncn[0:len(sync)-1] = sync[1:len(sync)]
            c = (sync>midpoint) & (shiftsync<midpoint) & (shiftsyncn<midpoint)
            sync[c.flatten()] = 0
            shiftsync[1:len(sync)] = sync[0:len(sync) - 1] #this goes one point behind
            print('cond1')
            cond1 = shiftsync > midpoint
            cond2 = sync < midpoint
            #cond3 = (supershiftsync1 >midpoint) & (supershiftsync2 >midpoint)
            print('cond2')
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
            print('problem...')
            vones=numpy.ones(numpy.shape(imgTimes))*midpoint#2.5
            vonese = numpy.ones(numpy.shape(imgendTimes)) * midpoint*0.9#2.49990
            #print(numpy.shape(sync))
            #print(type(sync))
            print("plotting...")

            #plt.scatter(imgTimes, vones, marker='.')
            plotsc = self.ui.PlotWidget2
            #plotsc.clear()
            #plotsc.plot(edf.times + self.timesscore[0], sync)
            #plotsc.plot(imgTimes, vones, pen=None, symbol='o',symbolPen='r', symbolSize=4)
            #plotsc.plot(imgendTimes, vonese, pen=None, symbol='s', symbolPen='b', symbolSize=4)
            #plt.show()

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
                        offtimes[k] = imgendTifpiemes[i]
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
            chunk_score=[]
            atimesc = numpy.asarray(self.timesscore)
            ascore = numpy.asarray(self.score)
            for k in range(len(ontimes)):
                indx = (atimesc >= ontimes[k]) & (atimesc <= offtimes[k])
                chunk_times_score = numpy.append(chunk_times_score,atimesc[indx])
                chunk_score = numpy.append(chunk_score,ascore[indx])
            voneson = numpy.ones(numpy.shape(ontimes)) * midpoint*1.2  # 2.5
            vonesoff = numpy.ones(numpy.shape(offtimes)) * midpoint * 1.4  # 2.5
            #vonesoff = numpy.ones(numpy.shape(offtimes2)) * midpoint * 1.2
            plotsc.plot(ontimes, voneson, pen=None, symbol='o', symbolPen='k', symbolSize=4)
            plotsc.plot(offtimes, vonesoff, pen=None, symbol='s', symbolPen='m', symbolSize=4)
            """Make a time vector with the same number of point than the Ca traces, aligned with the EEG timing."""
            self.timeTraces = numpy.zeros(self.nframes)
            print(self.nframes_from_mats[0])
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
            plotCanv = self.ui.PlotWidget_tr
            plotCanv.clear()
            print(len(self.Traces))
            for i in range(len(self.Traces)):
                plotCanv.plot(self.timeTraces, self.Traces[i] + 10 * i, pen=(i, len(self.Traces)))
            plotCanv.plot(chunk_times_score, chunk_score *10, pen='k')
            #To do: jellybeans...

    def saveAll(self):
        #comparing periods, plotting and saving. assuming all times in sec.
        for p in starttimesdict.keys():#finidng fist an last movie for each period
            k0, kf = 0, 0
            while self.starttimes[k] < starttimesdict[p]:
                k0 += 1
            while self.endtimes[kf] < endtimesdict[p] or kf >= len(self.endtimes):
                kf += 1
            kf -= 1
            chunk = self.Traces[:, k0:kf]
            mzsc = chunk.mean()
            self.zsc.append(list(chunk.mean(1)))
            self.meanZscore.append(mzsc)
        plt.bar(starttimesdict.keys(), self.meanZscore)
        plt.ylabel('Z score')
        plt.xlabel('Period')
        #plt.legend(loc='upper left')
        plt.show()
        #now saving
        f = open('zscorep.txt','w')
        for i in self.zsc:
            f.write(str(i)+'/n')
        f.close()




"""p5 = win.addPlot(title="Scatter plot, axis labels, log scale")
x = np.random.normal(size=1000) * 1e-10
y = x*1000 + 0.005 * np.random.normal(size=1000)
y -= y.min()-1.0
mask = x > 1e-15
x = x[mask]
y = y[mask]
p5.plot(x, y, pen=None, symbol='t', symbolPen=None, symbolSize=10, symbolBrush=(100, 100, 255, 50))
p5.setLabel('left', "Y Axis", units='A')
p5.setLabel('bottom', "Y Axis", units='s')
p5.setLogMode(x=True, y=False)"""





if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())