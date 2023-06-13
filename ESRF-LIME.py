'''Adapted from script for ESRF 2021 Data (V4_afterbeamdump)
This script must be put in the file directory with the pertinent
data you want to plot. Use .h5 (HDF5) files from ESRF for XRF,
.dat files for XRD and .txt general report (Biologic) for ECHEM'''

import os
import pandas
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import matplotlib.transforms as mt
from matplotlib import colors as mcolors
import numpy
import seaborn
import h5py #For reading HDF5 files
import imageio #To make the GIF from image folder
import re #For sorted_nicely
from celluloid import Camera #Makes it easier to animate plots
import scipy
from scipy import signal #For Savitsky-Golay data smoothing
from scipy import optimize
from BaselineRemoval import BaselineRemoval #Obviously for removing the baseline. Specifically of the XRD raw diffractogram.
from dask import dataframe as dd #for faster csv import
import time
import math
import sklearn
import copy #used to make deep copies of nested lists. Used only to duplicate the data for the heatmap function


class Main:
    def __init__(self):
        ####PARAMETERS####
        self.dir = os.getcwd()
        self.XRFscanNumbers = range(526, 635, 2) #(start, end+1, number of components) Select the scan numbers of the h5 file you want to collect.
        self.stepsLFP, self.stepsSep = 11, 6 # Datapoints in each layer
        self.timestart = 1 # Time (mins) that the XRF/XRD scanning started (relative to echem start)
        self.timeend = 1260
        self.timeperscan = 55 # Approx 55s based on time assesment
        self.timefirstscan = 1.0 # Was 1.3. Why? Approximate first scan time (minutes) relative to echem start time
        self.ticksize = 18
        self.removeList = ['Lima_run2_0001_163.1'] # list of XRF scans to remove. Make sure you don't disrupt grouping step.
        self.removeListXRD = [163] # List of XRD scans to remove. Make sure you don't disrupt grouping step.
        self.concentration_electrolyte = 1.1 # total electrolyte concentration (M). For normalization of As Ka
        self.form_factor_ratio_FP_LFP = 1.863

    def sorted_nicely(self, list):  # used to sort file names alphanumerically
        '''Sorts the given iterable in the way that is expected.
        Required arguments:
        list -- The iterable to be sorted.'''
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(list, key=alphanum_key)

    def gaussian(self, x, amplitude, mean, stddev): # Used to fit XRF peaks to a gaussian.
        '''The standard Gayssian equation used for peak fitting.'''
        # Define gaussian function. Used with scipy.optimize to fit to this function later.
        return amplitude * numpy.exp(-((x - mean) / 4 / stddev) ** 2)

    def multiple_gaussian(self, x, meanlist, stddevlist, *args): # Where *args will be as many gaussian amplitudes as you want
        '''This function can be fed a list of a fixed means and fixed stddevs into a lambda function
        That lambda function will be optimized with *args, which is a list of the amplitudes for
        each gaussian contribution. Args is passed because it can vary in length to accomodate any
        number of amplitudes (and therefore gaussian terms) as your heart desires'''
        #Define gaussian function. Used with scipy.optimize to fit to this function later.
        gauss = 0
        for g in range(len(args)): # For number of gaussians you want to fit
            gauss = gauss + args[g] * numpy.exp(-((x - meanlist[g]) / 4 / stddevlist[g]) ** 2)
        return gauss

    def get_echem_files(self):  # Get all sheets of all echem files into dataframes and concatenates them into a single dataframe
        '''Gets all echem files and concatenates them together if there are multiple'''
        os.chdir(f'{self.dir}\Echem') # Go to echem directory

        listdir = os.listdir() # Get current directory

        fileList = list() # Create empty list for all echem files

        for file in listdir: # Append all files to fileList
            if file.endswith('.txt'):
                fileList.append(file)

        for i in range(len(fileList)): # For ECLab text files
            dfEchemFile = pandas.read_csv(fileList[i], delimiter="\t",engine='python', encoding='latin1')  # Read it directly into pandas Dataframe

            if i == 0:  # For the first fileDF produced, call it self.dfEchem
                self.dfEchem = dfEchemFile
            else:  # For all following fileDFs produced, concatenate them to dfEchem and add last time value of previous file for continuous time
                '''First substract first value of time from all others (in case file does not start at 0s)'''
                dfEchemFile['time/s'] = dfEchemFile['time/s'] - dfEchemFile.iloc[0, dfEchemFile.columns.get_loc('time/s')]
                '''Get last time value from dfEchem. -1 is last row index and .columns.get_loc gets appropriate column index'''
                dfEchemFile['time/s'] = dfEchemFile['time/s'] + self.dfEchem.iloc[-1, self.dfEchem.columns.get_loc('time/s')]
                self.dfEchem = pandas.concat([self.dfEchem, dfEchemFile], ignore_index=True) # concatenate with initial file

        '''Convert units and rename columns. Make sure column names correspond to raw echem .txt file'''
        #self.dfEchem['Vol(mV)'] = self.dfEchem['Vol(mV)'] * 1000  # Convert V to mV if necessary
        self.dfEchem['time/s'] = self.dfEchem['time/s'] / 60 # Convert s to mins if necessary
        self.dfEchem = self.dfEchem.rename(columns={"Ewe/V": "Vol(mV)", "<I>/mA": "Cur(mA)", "Capacity/mA.h": "Cap(mAh)", "cycle number": "Cycle ID"}, errors="raise")  # Rename certain columns

        print('Upload Echem Complete')


    def plot_echem(self):
        '''Plots only the electrochemistry'''
        figEchem, axEchem = plt.subplots(figsize=(8,8))

        Poty = self.dfEchem['Vol(mV)'] # Potential
        Potx = self.dfEchem['time/s'] # Elapsed Echem Time
        Cury = self.dfEchem["Cur(mA)"]  # Current
        Curx = self.dfEchem["time/s"]  # Elapsed Echem Time

        axEchem.plot(Potx, Poty, 'k-', linewidth = 3)

        '''If you want any vertical lines plotted'''
        verticalLinesToPlot = [30, 34, 41] # List the scan #s of the vertical lines you want
        colour = 0
        print(self.XRF_XRD_time_array)
        for i in verticalLinesToPlot:
            trans = mt.blended_transform_factory(axEchem.transData, axEchem.transAxes)  # Transform uses relative ymin and ymax values for vlines
            axEchem.vlines(x=self.XRF_XRD_time_array[i], color=self.coloursListRaw[colour], ymin=0, ymax=1, linestyles='--', linewidth=2, transform=trans)  # Plot all time vertical lines
            colour = colour + 1


        #Format plot
        #axEchem.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # Remove Potential x ticks
        axEchem.set_ylabel('E (V vs. Li)', fontsize=16)
        axEchem.set_xlabel('Time (minutes)', fontsize=16)
        axEchem.tick_params(labelsize=14)
        axEchem.set_xlim(0, 480) #0 to 126 is the first charge at 2C/5 in Echo_Modeling_Final
        axEchem.set_ylim(0.8, 2.7)
        #axEchem.yaxis.set_major_locator(plt.MaxNLocator(3))  # Choose # of ticks on y
        figEchem.tight_layout()

        plt.show()

    def echem_animation(self, TotalFrames, TimeStart, TimeEnd): #Makes an animation of Echem vs time to match with XRD/XRF animations
        '''Animates the electrochemistry plot vs time for presentations
        Total frames is the amount of frames required to emulate XRF/XRD scan/position animation
        #TimeStart and TimeEnd are the Echem times you want to animate to and from'''

        #interpolate the data
        Poty, Potx = self.dfEchem['Vol(mV)'], self.dfEchem['time/s']   # Potential and Elapsed Echem Time
        EchemInterp = scipy.interpolate.interp1d(Potx, Poty)
        self.multiplier = 2 #multiplier determines the number of datapoints to interpolate. Larger = less choppy interped data)
        pointsNum = TotalFrames*self.multiplier #Will give number of interpolated points. Large numbers (>2) don't work well for timing.
        xInterval = numpy.linspace(TimeStart, TimeEnd, pointsNum)
        PotyInterp = EchemInterp(xInterval)

        #plt.plot(xInterval, PotyInterp, 'k.')
        #plt.show()

        # Make GIF of Echem
        duration = 10 / (len(xInterval))  # to give 19.6s to match
        self.make_gif_celluloid(xInterval, PotyInterp, 'Echem_GIF', duration, 'black', TimeStart, TimeEnd, 0.8, 2.7)


    def get_xrf_files(self):  # Get all sheets of all files into dataframes and concatenates them into a single dataframe
        '''Gets the required XRF data from the .h5 file. Then takes that data
        and organizes it into numpy arrays for data treatment. Important to
        carefully select the scan numbers to collect here. Often there can be
        discontinuous data (if some scans crashed) that can cause the scans to
        be improperly classified.'''
        #os.chdir(f'{self.dir}\XRF')
        os.chdir('C:/Users/Jeremy Dawkins/Documents/ESRF_Full_Data_2022/SynologyDrive/id31/id31/Lima/Lima_run2')

        # Lists files in current directory ('.')
        files = os.listdir()

        for i in files:  # For all files in directory
            if i.endswith('.h5'):  # Only accept h5 files. Only 1
                file = i

        #Read and save data from h5 file. The Group (f) works like a dictionary
        self.h5file = h5py.File(file, "r") #Read the file as a group (dictionary style)

        # List and sort all scans (each scan is a group)
        print("Keys: %s" % self.h5file.keys()) #Print all keys from group. These can be other groups or datasets.
        allScanGroups = list(self.h5file.keys())
        allScanGroups = self.sorted_nicely(allScanGroups) #Organizes strings alphanumerically
        #for i in self.removeList:
            #allScanGroups.remove(i) #Removes scans listed in setup

        print(allScanGroups)

        # Get the raw XRF spectra
        MasterXRFData = list()  # Nested list. First call the scan #, then call the specific position within the scan for full spectrum
        for i in self.XRFscanNumbers:  #0 based indexing (so -1 from scan # you actually want). for all 4C rates use 526-635. Maybe end at 637 is actually correct. Check this.
            scanID = [allScanGroups[i], allScanGroups[i + 1]]  # Gets the scan ID of interest. Get also the next one as they come in trios (LFP, then separator)
            scanGroup = [self.h5file[ele] for ele in scanID]  # Gets the scan ID pair's scan groups
            measure = [ele['measurement'] for ele in scanGroup]  # These are measurement groups where the data is

            # Open all datasets for each scan (Raw XRF, As Ka, FeKa, CrKa)
            scanXRFdataList = [[ele['fx_det0'] for ele in measure],  # Raw XRF
                               [ele['fx_det0_As_Ka'] for ele in measure],  # AsKa
                               [ele['fx_det0_Fe_Ka'] for ele in measure],  # FeKa
                               [ele['scaled_mondio'] for ele in measure],  # Incident beam flux
                               [ele['fx_det0_fractional_dead_time'] for ele in measure],  # deadtime
                               [ele['fx_det0_elapsed_time'] for ele in measure],  # elapsed time detector
                               [ele['elapsed_time'] for ele in measure],  # elapsed time for measurement
                               [ele['epoch'] for ele in measure], # Epoch (absolute) time
                               [ele['saz2'] for ele in measure]]  #position value

            # Append the data to a master file
            for j in range(len(scanXRFdataList)):  # Where j is the type of data saved
                scanXRFdataList[j] = [ele[:] for ele in scanXRFdataList[j]]  # Get all the positions for each scan # in each dataset
                scanXRFdataList[j] = numpy.array([item for sublist in scanXRFdataList[j] for item in sublist])  # Flatten the sublist to get both electrode and separator scans together
            MasterXRFData.append(scanXRFdataList)  # Nested list.  MasterXRFData[scan#][data type][pos] for full spectrum only
            print('Scan ' + str(i + 1) + ' (electrode) and ' + str(i + 2) + ' (Sep) XRF data acquired in MasterXRFData.')  # +1 cause scans start at 1 instead of 0

        ###MasterXRFData --> each scans data (electrode and separator) --> RawXRF, AsKa, FeKa, CrKa data --> data for each position (98 total)###

        # Save all raw spectra by position at all times instead of having all positions at each time
        self.allXRFbyPosition = list()  # will be list of data for a single position at every scan
        for i in range(len(MasterXRFData[0][0])): #For each position
            XRFbyPosition = list()  # will be list of data for a single position at every scan
            for j in range(len(MasterXRFData)): #For each scan
                try:
                    XRFbyPosition.append(MasterXRFData[j][0][i]) #MasterXRFData[scan#][datatype][position]
                except IndexError:
                    XRFbyPosition.append(None)
                    print('Missing data at scan ' + str(j) + ' and position ' + str(i))
            self.allXRFbyPosition.append(XRFbyPosition) #self.allXRFbyPosition[position][scan#]

        # Save all As, Fe and Cu vs position traces for each scan
        self.allIntegralbyType = list()
        for i in range(len(MasterXRFData[0])):  # For each datatype
            allIntergralbyScan = list()  # will be list of data for each datatype
            for j in range(len(MasterXRFData)):  # For each scan
                XRFbyScan = list()
                for k in range(len(MasterXRFData[0][0])): #For each position
                    try:
                        XRFbyScan.append(MasterXRFData[j][i][k])  # MasterXRFData[scan#][datatype][position]. 1 = As, 2 = Fe, 3 = Cr, 4 = deatime, 5 = elapsed time
                    except IndexError:
                        XRFbyScan.append(None)
                        print('Missing data at scan ' + str(j) + ' for datatype ' + str(i) + ' and position ' + str(k))
                allIntergralbyScan.append(XRFbyScan)  #allIntergralbyScan[scan#][position] for one of the datatypes
            self.allIntegralbyType.append(allIntergralbyScan) #self.allIntergralbyType[datatype][scan#][position]. 1 = As, 2 = Fe, 3 = Cr, 4 = deatime, 5 = elapsed time

        self.allIntegralbyType = numpy.array(self.allIntegralbyType) #convert to array for slicing

    def baseline_correct_XRF(self):
        '''This will baseline correct all raw XRF spectra. VERY slow and not really necessary.'''
        for i in range(len(self.allXRFbyPosition)): # for each position where self.allXRFbyPosition[position][scan#]
        #for i in range(5,35):
            posData = self.allXRFbyPosition[i]
            print('Baseline Correcting XRF @ Position ' + str(i))
            for j in range(len(posData)):
                posScanData = self.allXRFbyPosition[i][j] # self.allXRFbyPosition[position][scan#]
                baseObj = BaselineRemoval(posScanData) # create baseline object
                self.allXRFbyPosition[i][j] = baseObj.ZhangFit() # fit the data and transform accordingly

        #plot an example with baseline
        baseObj = BaselineRemoval(self.allXRFbyPosition[22][0])
        baselineCorrected = baseObj.ZhangFit()
        baselineplot = self.allXRFbyPosition[22][0] - baselineCorrected
        plt.plot(baselineplot, 'r-')
        plt.plot(self.allXRFbyPosition[22][0], 'k-')
        plt.show()

    def plot_xrf_raw_spectra(self):
        '''This function will plot one (or many) RAW XRF spectra w/ gaussian fits'''
        #Common parameters
        position = 7
        scanNums = [30, 34, 41] #0 is start right after discharge at 2C. 3-14 is end of charge at 4C. 32-42 is discharge at 4C

        #Parameters that you set once
        AspeakMax = 10800
        AspeakMin = 10250
        FepeakMax = 6600
        FepeakMin = 6150
        p0 = [100, 10525, 5]  # initial guess for As Ka gaussian fit (height, mean, stdev)
        eVsPerBin = 12 # 12eV per bin for XRF detector

        '''Setup figure'''
        figRawXRF, axRawXRF = plt.subplots(figsize=(8,8))

        # Choose colours to plot
        colour_subsection = numpy.linspace(0.1, 0.9, len(scanNums))  # Splits colourmap an equal # of sections related to # of curves
        coloursListRaw = [cm.autumn_r(x) for x in colour_subsection]
        self.coloursListRaw = coloursListRaw #create class level object to use in other functions

        #Get x values (12eV bins so its number of datapoints*12)
        XRFxVals = numpy.array(range(len(self.allXRFbyPosition[0][0]))) * eVsPerBin

        '''Integrate peaks manually'''
        colour = 0  # colour index
        # First loop for each scan
        for i in range(len(self.allXRFbyPosition[position])):  # for each scan where self.allXRFbyPosition[position][scan#], choose an arbitrary position
            if i in scanNums:
                print('Integrating XRF @ Scan ' + str(i))
                # Integrate over peak angles
                spectrum = numpy.stack((XRFxVals, self.allXRFbyPosition[position][i]), axis=1)  # Stack to have (eVs, Int) together
                Aspeak = [x for x in spectrum if x[0] < AspeakMax and x[0] > AspeakMin]  # Get As Ka peak angles #Filter for wanted eVs
                Fepeak = [x for x in spectrum if x[0] < FepeakMax and x[0] > FepeakMin]  # Get Fe Ka peak angles #Filter for wanted eVs
                Aspeak, Fepeak = numpy.array(Aspeak).transpose(), numpy.array(Fepeak).transpose()  # Transpose reverses the axes (i.e seperates intensity and 2th again)
                spectrum = spectrum.transpose()
                # Gaussian fit the As Ka peak
                popt, _ = optimize.curve_fit(self.gaussian, Aspeak[0], Aspeak[1], p0=p0)  # curve_fit(function, x, y)
                gausFit = self.gaussian(Aspeak[0], *popt) #get y values of gaussian curve
                #axRawXRF.plot(spectrum[0], spectrum[1], '-', color='black')  # Plot a single black spectrum
                axRawXRF.plot(spectrum[0], spectrum[1], '.', marker = 'x', color = coloursListRaw[colour], markersize=12) #Plot raw data
                axRawXRF.plot(Aspeak[0], gausFit, color = coloursListRaw[colour], linestyle='-', linewidth = 5) #Plot Gaussian fit of As ka peak
                colour = colour + 1

        '''Customize figure Integrated XRD vs Time'''
        axRawXRF.set_xlabel('Energy (eV)', fontsize=22)
        axRawXRF.set_ylabel('Intensity (counts)', fontsize=22)
        axRawXRF.set_xlim(10200, 10850)
        axRawXRF.set_ylim(-5, 105)
        axRawXRF.tick_params(axis='both', which='major', labelsize=14)
        #axRawXRF.legend(['As K\u03B1'])
        figRawXRF.tight_layout()
        #figRawXRF.savefig('RawXRF.svg', dpi=600)
        plt.show()


    def plot_xrf_linscans_manual_integral(self):
        '''This function will calculate the average OCV integrals for XRF and then divide
        the real data (self.allIntegralsbyType) by that average
        Normalize integrated peaks to OCV measurements
        Get average values (for first n scans (n = numAvg) for each integral by position
        Plot peak intensity vs Time'''
        # Common parameters
        lowerScanNum = 0  # lower scan # to plot 32
        #upperScanNum = len(self.allXRDIntegralbyType[0]) # for all use: len(self.allIntegralbyType[1]) where self.allIntegralbyType[datatype][scan#][position]. 1 = LFP1, 2 = LFP2, 3 = background, 4 = graphite and 5 = Li graphite
        upperScanNum = 53 #0 is start right after discharge at 2C. 3-14 is end of charge at 4C. 32-42 is discharge at 4C
        stepNum = 1  # Plots every n scans between lower and upper threshold
        lowerPos = 0  # lower position value to plot. #0-11 gives LFP, 11-21 gives LTO.
        upperPos = 21 #22 is true end, but stop at 21
        numOCVscans = 3 #number of OCV scans to normalize to. Will take first n scans starting from lowerScanNum (and counting backwards)

        # Parameters that you set once
        eVsPerBin = 12 # 12eV per bin for XRF detector
        AspeakMax = 10800
        AspeakMin = 10250
        FepeakMax = 6600
        FepeakMin = 6150
        p0 = [100, 10525, 5]  # initial guess for As Ka gaussian fit (height, mean, stdev)

        '''Setup figure'''
        manualIntegralList = [] # [scan# index in selected range][position][0 = LFP peak integral, 1 = FP peak integral]. Holds all scans

        # Choose colours to plot
        colour_subsection = numpy.linspace(0.1, 0.9, len(range(lowerScanNum, upperScanNum, stepNum)))  # Splits colourmap an equal # of sections related to # of curves
        self.coloursList, self.coloursList2, self.coloursList3, self.coloursListRatio = [cm.autumn_r(x) for x in colour_subsection], [
            cm.winter_r(x) for x in colour_subsection], [cm.summer_r(x) for x in colour_subsection], [cm.gray_r(x) for x in colour_subsection]

        #Get x values (12eV bins so its number of datapoints*3)
        XRFxVals = numpy.array(range(len(self.allXRFbyPosition[0][0]))) * eVsPerBin


        '''Integrate peaks manually'''
        # First loop for each scan
        for i in range(len(self.allXRFbyPosition[upperPos-1])): # for each scan where self.allXRFbyPosition[position][scan#]. Upperpos chosen cause later positions dont have all scans
            manualIntegralallPos = []  # To hold values at all positions for a single scan
            print('Integrating XRF @ Scan ' + str(i))
            # Second loop for each position
            for j in range(lowerPos, upperPos): #and for each position within that scan
                #Integrate over peak angles
                spectrum = numpy.stack((XRFxVals, self.allXRFbyPosition[j][i]), axis=1)  # Stack to have (eVs, Int) together
                Aspeak = [x for x in spectrum if x[0] < AspeakMax and x[0] > AspeakMin]  # Get As Ka peak angles #Filter for wanted eVs
                Fepeak = [x for x in spectrum if x[0] < FepeakMax and x[0] > FepeakMin]  # Get Fe Ka peak angles #Filter for wanted eVs
                Aspeak, Fepeak = numpy.array(Aspeak).transpose(), numpy.array(Fepeak).transpose()  # Transpose reverses the axes (i.e seperates intensity and 2th again)
                #Gaussian fit the As Ka peak
                popt, _ = optimize.curve_fit(self.gaussian, Aspeak[0], Aspeak[1], p0=p0)  # curve_fit(function, x, y)
                #Integrate the gaussian fit
                Asintegral, Feintegral = scipy.integrate.simpson(self.gaussian(Aspeak[0], *popt), x=Aspeak[0],even='avg'), scipy.integrate.simpson(Fepeak[1],x=Fepeak[0],even='avg')
                #Or integrate the raw data
                #Asintegral, Feintegral = scipy.integrate.simpson(Aspeak[1], x=Aspeak[0],even='avg'), scipy.integrate.simpson(Fepeak[1],x=Fepeak[0],even='avg')
                manualIntegralallPos.append((Asintegral, Feintegral)) # [scan number in chosen range][0 = As, 1 = Fe]
            manualIntegralList.append(numpy.array(manualIntegralallPos)) # manualIntegralList[scan#][pos#][As/Fe]
        manualIntegralList = numpy.array(manualIntegralList) # Convert list to array
        manualIntegralList = manualIntegralList.transpose(0, 2, 1)  # transpose pos# as last value manualIntegralList[scan#][As=0/Fe=1][pos#]
        #self.manualIntegralList = manualIntegralList #create a class level variable for use in other functions
        self.manualIntegralforHeatmap = numpy.asarray(copy.deepcopy(manualIntegralList))  # if you dont use copy.deepcopy, it will simply reference the original object

        # Setup Plot
        figXRF, (axXRF, axXRF_Echem) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 0.25]})
        axXRF_Current = axXRF_Echem.twinx()

        '''Get OCV scans for normalization'''
        OCVAs, OCVFe = [], []
        for j in range(lowerPos, upperPos):  # for each position within that scan
            # Integrate over peak angles
            posOCVintegralAs = []
            posOCVintegralFe = []
            for i in range(lowerScanNum, lowerScanNum-numOCVscans, -1): #By including lowerScanNum, we will get a RELATIVE baseline from the start of the current cycle only
                spectrum = numpy.stack((XRFxVals, self.allXRFbyPosition[j][i]), axis=1)  # Stack to have (eVs, Int) together
                Aspeak = [x for x in spectrum if x[0] < AspeakMax and x[0] > AspeakMin]  # Get As Ka peak angles #Filter for wanted 2th
                Fepeak = [x for x in spectrum if x[0] < FepeakMax and x[0] > FepeakMin]  # Get Fe Ka peak angles #Filter for wanted 2th
                Aspeak, Fepeak = numpy.array(Aspeak).transpose(), numpy.array(Fepeak).transpose()  # Transpose reverses the axes (i.e seperates intensity and 2th again)
                #Gaussian fit the As Ka peak
                popt, _ = optimize.curve_fit(self.gaussian, Aspeak[0], Aspeak[1], p0=p0)  # curve_fit(function, x, y)
                #Integrate the gaussian fit
                AsintegralScan, FeintegralScan = scipy.integrate.simpson(self.gaussian(Aspeak[0], *popt), x=Aspeak[0],even='avg'), scipy.integrate.simpson(Fepeak[1],x=Fepeak[0],even='avg')
                #Or Integrate the raw data
                #AsintegralScan, FeintegralScan = scipy.integrate.simpson(Aspeak[1], x=Aspeak[0],even='avg'), scipy.integrate.simpson(Fepeak[1],x=Fepeak[0],even='avg')
                posOCVintegralAs.append(AsintegralScan), posOCVintegralFe.append(FeintegralScan) #Add the integral at pos j and scan i into a list.
            Asintegral, Feintegral = sum(posOCVintegralAs) / len(posOCVintegralAs), sum(posOCVintegralFe) / len(posOCVintegralFe) #Take average of n scans at that position
            OCVAs.append(Asintegral), OCVFe.append(Feintegral)
        self.OCVAsforHeatmap = copy.deepcopy(OCVAs)  # if you don't use copy.deepcopy it will simply reference to the original object
        #Correct OCV scan for deadtime
        deadTimeCorr = ((1 - self.allIntegralbyType[4, 0, lowerPos:upperPos]) / self.allIntegralbyType[5, 0, lowerPos:upperPos])
        OCVAs, OCVFe = OCVAs / deadTimeCorr, OCVFe / deadTimeCorr #deadtime correct
        #Normalize OCV scan to incident beam flux
        OCVAs, OCVFe = OCVAs/self.allIntegralbyType[3, 0, lowerPos:upperPos], OCVFe/self.allIntegralbyType[3, 0, lowerPos:upperPos]
        #OCVAs = OCVAs / OCVFe #Normalize As OCV to Fe signal.
        #OCVAs, OCVFe = 1, 1 # use this to NOT normalize to OCV values (skews results when peak not present in first scan)



        '''Get all scans, normalize, smooth and plot'''
        self.normalizedAs = [] #class level variable to store concentration profiles
        for i in range(lowerScanNum, upperScanNum, stepNum):
            # Deadtime correction of integrated peaks (based on wikipedia equation for correction: counts = raw counts / ((1-deadtime)/elapsed time)
            deadTimeCorr = ((1 - self.allIntegralbyType[4, i, lowerPos:upperPos]) / self.allIntegralbyType[5, i, lowerPos:upperPos]) # [datatyp][scan#][pos] :-1 is because last scan contains "none" values for last 10 scans and bugs out the entire position
            manualIntegralList[i][0][:] = manualIntegralList[i][0][:] / deadTimeCorr # As
            manualIntegralList[i][1][:] = manualIntegralList[i][1][:] / deadTimeCorr # Fe
            # Incident flux normalization
            manualIntegralList[i][0][:] = manualIntegralList[i][0][:] / self.allIntegralbyType[3, i, lowerPos:upperPos]  # Where self.allIntegralbyType[3] is scaled_mondio
            manualIntegralList[i][1][:] = manualIntegralList[i][1][:] / self.allIntegralbyType[3, i, lowerPos:upperPos]  # Fe
            # Calibrate XRF to first scan @ OCV
            #normalizedAs, normalizedFe = manualIntegralList[i][0][:] / manualIntegralList[i][1][:] / OCVAs, manualIntegralList[i][1][:] / OCVFe  #normalize to Fe signal AND OCV. Need to normalize the OCV lists above as well.
            normalizedAs, normalizedFe = (manualIntegralList[i][0][:] / OCVAs) * self.concentration_electrolyte, (manualIntegralList[i][1][:] / OCVFe)  # These are the integrated peaks normalized to the first scan
            self.normalizedAs.append(normalizedAs) #get class level variable to store date
            #normalizedAs, normalizedFe = manualIntegralList[i][0][lowerPos:upperPos], manualIntegralList[i][1][:] # These are integrated peak with manual pos choice
            # Apply Savitzky-Golay data smoothing
            movingBox = 5
            #normalizedSmoothedAs = scipy.signal.savgol_filter(normalizedAs, window_length=movingBox, polyorder=2)
            #normalizedSmoothedFe = scipy.signal.savgol_filter(normalizedFe, window_length=movingBox, polyorder=2)
            # Calculate As/Fe Ratio
            #AstoFeRatio = normalizedSmoothedAs / normalizedSmoothedFe
            # Plot (seperately plot LFP so they don't connect)
            '''If you want position in microns...'''
            #axXRF.plot(self.XRF_XRD_pos_array[lowerPos:upperPos], normalizedAs, '-', color=self.coloursList[int((i - lowerScanNum) / stepNum)], linewidth=2)  # [scan# index in selected range][0 = LFP peak integral, 1 = FP peak integral][pos#]
            axXRF.plot(range(lowerPos, upperPos, 1), normalizedAs[lowerPos:upperPos], '-', color=self.coloursList[int((i - lowerScanNum) / stepNum)], linewidth=2)  # [scan# index in selected range][0 = LFP peak integral, 1 = FP peak integral][pos#]
            '''If you want position # only...'''
            #axXRF.plot(self.XRF_XRD_pos_array[0:11], normalizedAs[0:11], '-', color=self.coloursList[int((i - lowerScanNum) / stepNum)], linewidth=2)  # [scan# index in selected range][0 = LFP peak integral, 1 = FP peak integral][pos#]
            #axXRF.plot(self.XRF_XRD_pos_array[11:21], normalizedAs[11:21], '-', color=self.coloursList[int((i - lowerScanNum) / stepNum)], linewidth=2)


        # Plot the Echem
        x = self.dfEchem['time/s']
        y = self.dfEchem['Vol(mV)']
        y2 = self.dfEchem["Cur(mA)"]
        axXRF_Echem.plot(x, y, 'k-', linewidth=2)
        axXRF_Current.plot(x, y2, 'b-', linewidth=2)

        # Plot the Echem vertical lines
        for i in range(lowerScanNum, upperScanNum, stepNum):
            trans = mt.blended_transform_factory(axXRF_Echem.transData, axXRF_Echem.transAxes)  # Transform uses relative ymin and ymax values for vlines
            axXRF_Echem.vlines(x=self.XRF_XRD_time_array[i], color=self.coloursList[int((i - lowerScanNum) / stepNum)], ymin=0, ymax=1, linestyles='--', linewidth=2, transform=trans)  # Plot all time vertical lines


        '''Customize figure Integrated XRD vs Time'''
        axXRF.set_xlabel('Position (\u03BCm)', fontsize=22)
        #axXRF.set_ylabel('Absolute Intensity (counts)', fontsize=22)
        #axXRF.set_ylabel('Normalized Intensity', fontsize=22)
        axXRF.set_ylabel('Li$^+$ Concentration (M)', fontsize=22)
        # axPeakRatio.set_ylabel('Ratio', fontsize=22)
        # axPeakRatio.tick_params(axis='both', which='major', labelsize=14)
        axXRF.set_xlim(lowerPos, upperPos-1)
        axXRF.set_ylim(0, 2)
        axXRF.tick_params(axis='both', which='major', labelsize=14)
        axXRF.legend(['As K\u03B1'])
        # Customize figure Gradient for Echem
        axXRF_Echem.tick_params(axis='both', which='major', labelsize=self.ticksize)
        axXRF_Echem.set_ylabel('E (V vs. Li)', fontsize=22)
        axXRF_Current.set_ylabel('I (mA)', fontsize=22)
        axXRF_Echem.set_xlabel('Time (Minutes)', fontsize=22)
        axXRF_Echem.tick_params(labelsize=self.ticksize)
        axXRF_Current.tick_params(labelsize=self.ticksize)
        # axXRF_Echem.set_xlim(self.timestart, self.timeend)
        axXRF_Echem.set_ylim(0.8, 2.7)
        axXRF_Echem.set_xlim(self.XRF_XRD_time_array[lowerScanNum]-1, self.XRF_XRD_time_array[upperScanNum])
        # Hide the right and top spines
        axXRF_Echem.spines['right'].set_visible(False)
        axXRF_Echem.spines['top'].set_visible(False)

        # Adjust spacing of subplots
        figXRF.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.25)
        figXRF.set_size_inches(8, 8)
        #figXRF.set_size_inches(7, 8)
        figXRF.tight_layout()

        #save fig as svg
        #figXRF.savefig('Figure_ManualXRFIntegrals.svg', format='svg', dpi=600)
        print('Plotted Manual XRF Integrals')

        #plt.show()

        ####################################3d plot###########################################
        '''Will also plot concentration linescans in 3d if you want'''

        x = range(lowerPos, upperPos, 1) #x axis is always the same. Position #'s selected
        verts = list()
        time = list()
        #Get data
        for i in range(lowerScanNum, upperScanNum, stepNum):
            # Get As Ka (y)
            y = manualIntegralList[i][0][:] / OCVAs #As Ka integral (normalized to OCV) vs position
            y[0], y[-1] = 0.3, 0.3  # Change first and last values of x to get a nice looking polygon
            # Get time value (z)
            z = self.XRF_XRD_time_array[i] #Get time of that scan
            time.append(z) #append time to time list
            verts.append(list(zip(x, y)))  # Zip joins the 2 into tuples and then the whole thing in verts list

        fig3d = plt.figure()
        ax3d = fig3d.gca(projection='3d')

        #for i in range(len(verts)):
            #print(i)
            #print(verts[i])
            #print(self.coloursList[i])
            # poly = PolyCollection(self.verts, edgecolor=self.colourListShort, facecolors='w', linewidths=2) #If you want white face
            #poly = PolyCollection(verts[i], edgecolor=self.coloursList[i], facecolors=self.coloursList[i])  # Converts vertices into polygons
            # poly.set_alpha(0) #If you want invisible face
            #ax3d.add_collection3d(poly, zs=time[i], zdir='y')

        poly = PolyCollection(verts, edgecolor=self.coloursList, facecolors=self.coloursList)  # Converts vertices into polygons
        # poly.set_alpha(0) #If you want invisible face
        ax3d.add_collection3d(poly, zs=time, zdir='y')

        # Modify 3d plot
        ax3d.set_xlabel('Depth (\u03bcm)', fontsize=16)
        ax3d.set_xlim3d(0, upperPos)
        # ax3d.set_xlabel('As K\u03B1 (counts)', fontsize=16)
        ax3d.set_zlabel('Conc. As (M)', fontsize=16)
        ax3d.set_zlim3d(0, 2)
        ax3d.set_ylabel('Time (minutes)', fontsize=16)
        ax3d.set_ylim3d(0, self.XRF_XRD_time_array[upperScanNum])
        # Hide the right and top spines
        ax3d.spines['right'].set_visible(False)
        ax3d.spines['top'].set_visible(False)

        fig3d.set_size_inches(7, 6)

        plt.show()


    def XRF_GIF_Pos(self):
        ### Make GIF per POSITION dataset ###
        posToGIF = 5
        duration = 10 / len(self.allXRFbyPosition[posToGIF])  # Duration for each frame in GIF
        #duration = 0.2  # Duration for each frame in GIF

        # Choose colours to plot
        colour_subsection = numpy.linspace(0.1, 0.9, len(self.allXRFbyPosition[0]))  # Splits colourmap an equal # of sections related to # of curves
        self.coloursList = [cm.autumn_r(x) for x in colour_subsection]

        # Create x values
        eVsPerBin = 12 # 3eV per bin for XRF detector
        xValues = [range(0, len(self.allXRFbyPosition[0][0])*eVsPerBin, eVsPerBin)] * len(self.allXRFbyPosition[0])  # Just take length one spectrum as the x values as there are none with the raw data

        # Make GIF
        self.make_gif_celluloid(xValues, self.allXRFbyPosition[posToGIF], 'GIF_Raw_Pos_' + str(posToGIF), duration, self.coloursList, 6000, 12300, 0, 200)  # using coloursList[color] uses the functions autumn colour map

    def XRF_GIF_Scan(self):
        figIntegratedXRF, axIntegratedXRF = plt.subplots(figsize=(8, 8))
        ### Make GIF per SCAN dataset ###
        #scanToGIF = 1
        scansToGIF = [15, 34, 41] #if multiple scans wanted
        duration = 0.2  # Duration for each frame in GIF

        # Create x values
        #xValues = [range(len(self.allXRFbyPosition[0][0]))] * len(self.allXRFbyPosition)
        xValues = range(len(self.manualIntegralList[0][0]))#where manualIntegralList[scan#][As=0/Fe=1][pos#]

        # Get y values
        #plotData = list()
        #for i in range(len(self.allXRFbyPosition)):
            #plotData.append(self.allXRFbyPosition[i][scanToGIF])

        # Get y values (if multiple scans)
        plotData = list()

        colour = 0
        for i in scansToGIF: #for each scan
            plotData.append(self.manualIntegralList[i][0])
            #axIntegratedXRF.plot(xValues, self.normalizedAs[i], color=self.coloursListRaw[colour]) #if you want concentration profiles
            axIntegratedXRF.scatter(xValues, self.manualIntegralList[i][0], color=self.coloursListRaw[colour], marker='x') #if you want intergrated As Ka
            colour = colour + 1


        # Customize figure Integrated XRD vs Time
        axIntegratedXRF.set_xlabel('Position', fontsize=22)
        axIntegratedXRF.set_ylabel('As K\u03B1 Peak Integral', fontsize=22)
        #axIntegratedXRF.set_ylabel('Li$^+$ Concentration (M)', fontsize=22)
        #axIntegratedXRF.set_xlim(10200, 10850)
        #axIntegratedXRF.set_ylim(0, 2)
        axIntegratedXRF.tick_params(axis='both', which='major', labelsize=14)
        axIntegratedXRF.legend(['End of Charge', 'Beginning of Discharge', 'End of Discharge'])
        figIntegratedXRF.tight_layout()
        #figRawXRF.savefig('RawXRF.svg', dpi=600)
        figIntegratedXRF.show()
        plt.show()


        # Make GIF
        #self.make_gif_celluloid(xValues, plotData, 'GIF_XRF_Scan_' + str(scanToGIF), duration, 'black', 1600, 4200, 0, 1200) # for a single scan but all positions

        # Close the h5 file
        #self.h5file.close()

    def animate_scan_gif(self): #Animates and saves a GIF of a XRF scan being made point by point
        figScanGIF, axScanGIF = plt.subplots()

        # Customize figure ScanGIF
        # axScanGIF.set_xlabel('Depth (\u03bcm)', fontsize=22)
        axScanGIF.set_xlabel('Depth (#)', fontsize=22)
        # axScanGIF.set_ylabel('As K\u03B1', fontsize=22)
        axScanGIF.set_ylabel('Intensity (counts)', fontsize=22)
        # self.axScanGIF.set_ylabel('Li$^+$ Concentration (M)', fontsize=22)
        axScanGIF.set_xlim(0, 98)
        axScanGIF.tick_params(axis='both', which='major', labelsize=14)
        # axScanGIF.legend(['As', 'Fe', 'Cr'])
        figScanGIF.set_size_inches(6, 6)
        figScanGIF.tight_layout()

        camera = Camera(figScanGIF)
        duration = 0.2 #Time each image shows up

        for j in range(len(self.allIntegralbyType[1][1][0:98])):
            axScanGIF.plot(self.allIntegralbyType[1][1][0:j], '.', color='orange', linewidth=2)
            axScanGIF.plot(self.allIntegralbyType[2][1][0:j], '.', color='blue', linewidth=2)
            axScanGIF.plot(self.allIntegralbyType[3][1][0:j], '.', color='purple', linewidth=2)

            # Add Custom Text
            axScanGIF.text(0.75, 0.9, 'Pos. ' + str(j), size=22, color='black', transform=axScanGIF.transAxes)

            # Snapshot
            figScanGIF.tight_layout()
            camera.snap()  # take a snapshot of the figure in the current state

        animation = camera.animate()  # Animate these snapshots
        animation.save('Animated Scan.gif', writer='Pillow', fps=1 / duration)

    def heatmap_XRF(self):
        '''Create Heatmap Object. MUST plot XRF from scan 0 previously to get correct OCV scan'''
        figHeatmap = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(4, 3, height_ratios=[0.05, 1, 0.4, 0.4])
        axColorbar = plt.subplot(gs[1])  # Here I assign each axis a zone of the greated gs object
        axHeatmap = plt.subplot(gs[3:6])
        axEchem = plt.subplot(gs[6:9])
        axCur = plt.subplot(gs[9:12])
        ytickreduction = 5
        endPos = 21  # must match endPos taken when integrating the data in other function
        ticksize = 14
        startscan = 0  # 0 to 54 is everything at 4C
        endscan = 54  # The plot_manual_integral function will determine where the OCV is taken from

        '''Get heatmap data'''
        #Deadtime and incident flux correction of OCV
        # Deadtime correction of integrated peaks (based on wikipedia equation for correction: counts = raw counts / ((1-deadtime)/elapsed time)
        deadTimeCorr = ((1 - self.allIntegralbyType[4, startscan, :endPos]) / self.allIntegralbyType[5, startscan, :endPos])  # [datatyp][scan#][pos] :-1 is because last scan contains "none" values for last 10 scans and bugs out the entire position
        self.OCVAsforHeatmap = self.OCVAsforHeatmap / deadTimeCorr
        # Incident flux normalization
        self.OCVAsforHeatmap = self.OCVAsforHeatmap / self.allIntegralbyType[3, startscan, :endPos]

        heatmapArray = []  # setup heatmap array as a list
        for i in range(startscan, endscan, 1):  # for each scan
            # Deadtime correction of integrated peaks (based on wikipedia equation for correction: counts = raw counts / ((1-deadtime)/elapsed time)
            deadTimeCorr = ((1 - self.allIntegralbyType[4, i, :endPos]) / self.allIntegralbyType[5, i, :endPos])  # [datatyp][scan#][pos] :-1 is because last scan contains "none" values for last 10 scans and bugs out the entire position
            self.manualIntegralforHeatmap[i][0][:] = self.manualIntegralforHeatmap[i][0][:] / deadTimeCorr  # As
            # Incident flux normalization
            self.manualIntegralforHeatmap[i][0][:] = self.manualIntegralforHeatmap[i][0][:] / self.allIntegralbyType[3, i, :endPos]  # Where self.allIntegralbyType[3] is scaled_mondio
            # Calibrate XRF to first scan @ OCV
            normalizedAs = (self.manualIntegralforHeatmap[i][0][:] / self.OCVAsforHeatmap) * self.concentration_electrolyte  # These are the integrated peaks normalized to the first scan chosen in conc. profile function
            normalizedAs = numpy.asarray(normalizedAs, dtype=numpy.float64)
            heatmapArray.append(normalizedAs)
        heatmapArray = numpy.asarray(heatmapArray, dtype=numpy.float64)  # convert to array. Dtype is important for plotting

        yAxis = range(0, endPos, 1)  # y axis for position values

        '''Plot heatmap'''
        colour_subsection = numpy.linspace(0, 1, 1000)  # Splits colourmap an equal # of sections related to # of curves
        cmap = [cm.jet(x) for x in colour_subsection]

        heatmapArray = heatmapArray.transpose()
        HM = seaborn.heatmap(heatmapArray, vmin=0.5, vmax=2.0, cmap=cmap, ax=axHeatmap, cbar_ax=axColorbar,
                             cbar_kws={"orientation": "horizontal", 'label': 'As K\u03B1 (Normalized counts)'},
                             xticklabels=0,
                             yticklabels=ytickreduction)  # The mask is for null values in the array(?) #vmin 0.01 and vmax 0.19 OR 0.15 and 0.5 for sep OR 0.1 - 3.0 for concentration

        '''Plot Echem'''
        # Plot the Echem
        x = self.dfEchem['time/s']
        y = self.dfEchem['Vol(mV)']
        y2 = self.dfEchem["Cur(mA)"]
        axEchem.plot(x, y, 'k-', linewidth=2)
        axCur.plot(x, y2, 'b-', linewidth=2)

        '''Modify Colorbar details'''
        # self.axColorbar.set_xlabel('As K\u03B1 (Normalized counts)', fontsize=16)
        axColorbar.set_xlabel('Li$^+$ concentration (M)', fontsize=18)  # was font 16 for paper fig
        axColorbar.xaxis.tick_top()  # Set colourbar ticks to top
        axColorbar.xaxis.set_label_position('top')  # Set colourbar label to top
        axColorbar.tick_params(axis='x', which='major', labelsize=ticksize - 4)

        '''Modify Heatmap Details'''
        HM.set(xlabel='', xticklabels=[], yticklabels=yAxis[0::ytickreduction])  # The 0::x just takes every nth element of list
        HM.tick_params(labelsize=ticksize)
        # HM.set_ylabel(ylabel='Depth (10\u00b2 \u03bcm)', fontsize=16)
        HM.set_ylabel(ylabel='Position (#)', fontsize=18)
        HM.set_yticklabels(HM.get_yticklabels(), rotation=0)  # Make sure labels are not rotated
        # self.axHeatmap.get_yaxis().set_major_formatter(plt.ScalarFormatter())  # Sets y axis to a scalar formatter
        # self.axHeatmap.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) #Changes the formatter to use sci notation
        figHeatmap.set_size_inches(10, 7.5)

        '''Modify Echem details'''
        axCur.set_ylabel(ylabel='I (mA)', fontsize=18)
        axEchem.set_ylabel(ylabel='E (V vs Li)', fontsize=18)
        axCur.set_xlim(self.XRF_XRD_time_array[startscan], self.XRF_XRD_time_array[endscan-1])
        axEchem.set_xlim(self.XRF_XRD_time_array[startscan], self.XRF_XRD_time_array[endscan-1])
        axCur.set_ylim(-14, 14)
        axEchem.set_ylim(0.8, 2.7)

        plt.show()

        '''Get a smoothed version of the heatmap'''
        figHMsmoothed = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(3, 3, height_ratios=[1, 0.4, 0.4], width_ratios=[1, 1, 1])
        axHMsmoothed = plt.subplot(gs[0:3])  # Here I assign each axis a zone of the greated gs object
        axEchemSmoothed = plt.subplot(gs[3:6])
        axCurSmoothed = axEchemSmoothed.twinx()
        axHeatmapSoC = plt.subplot(gs[6:9])

        HMinterp = axHMsmoothed.imshow(heatmapArray, vmin = 0.5, vmax = 2.0, interpolation='quadric', cmap='jet', aspect="auto") #Plot the interpolated heatmap
        cbar = axHMsmoothed.figure.colorbar(HMinterp, ax=axHMsmoothed, location='top', fraction=0.15, shrink=0.3) #Plot a color bar
        axEchemSmoothed.plot(x, y, 'k-', linewidth=2)
        axCurSmoothed.plot(x, y2, 'b-', linewidth=2)

        '''(Optional) Plot average SoC of LFP on top'''
        #axHeatmapSoC = axHMsmoothed.twinx()
        #Get average SoC of LFP for Heatmap'''
        for i in range(startscan, endscan, 1): # for all scans
            self.XRDIntegralListforHeatmap[i][1][:] = self.XRDIntegralListforHeatmap[i][1][:] / self.form_factor_ratio_FP_LFP #normalize LFP SoC for form factor
            AvgSoCLFP = numpy.mean((self.XRDIntegralListforHeatmap[i][1][:] / (self.XRDIntegralListforHeatmap[i][1][:] + self.XRDIntegralListforHeatmap[i][0][:])) * 100)
            axHeatmapSoC.scatter(i, AvgSoCLFP, marker='o', color='black')
        axHeatmapSoC.set_ylabel('SoC (%)', fontsize=18)
        axHeatmapSoC.set_ylim(0, 105)
        axHeatmapSoC.set_xlim(startscan, endscan)



        '''Smoothed heatmap settings'''
        axHMsmoothed.tick_params(labelsize=ticksize)
        axHMsmoothed.set_ylabel('Position (#)', fontsize=18)
        cbar.ax.set_xlabel('Li$^+$ concentration (M)', rotation=0, loc='center', verticalalignment='top', fontsize=14, labelpad=20)
        axCurSmoothed.set_ylabel(ylabel='I (mA)', color='b', fontsize=18)
        axEchemSmoothed.set_ylabel(ylabel='E (V vs Li)', fontsize=18)
        axEchemSmoothed.set_xlabel(xlabel='Time (Minutes)', fontsize=18)
        axCurSmoothed.set_xlim(self.XRF_XRD_time_array[startscan], self.XRF_XRD_time_array[endscan - 1])
        axEchemSmoothed.set_xlim(self.XRF_XRD_time_array[startscan], self.XRF_XRD_time_array[endscan - 1])
        axCurSmoothed.set_ylim(-14, 14)
        axEchemSmoothed.set_ylim(0.8, 2.7)
        figHMsmoothed.set_size_inches(10, 7.5)


        plt.tight_layout()
        #figHMsmoothed.savefig('Figure_Heatmapsmoothed.svg', format='svg', dpi=600)

        plt.show()

    #################################Diffraction####################################
    ### ONLY DO ONCE ###
    def group_dat_files_by_position(self): #this function will take the .dat files and save them into .csv's based on position
    ### ONLY DO ONCE ###
        os.chdir(f'{self.dir}\XRD\Lima_run2_4Cpart')

        # Lists files in current directory ('.')
        files = os.listdir()

        fileListXRD = list()
        for i in files:  # For all files in directory
            if i.endswith('.dat'):  # Only accept .dat files.
                fileListXRD.append(i)

        # Sort file names alphanumerically
        fileListXRD = self.sorted_nicely(fileListXRD)
        print(fileListXRD)

        # Read and save data from .dat file.
        allScansXRD = dict() # where keys to data (raw XRD diffractograms) are [pos#][scan#]

        firstScanNum = int(fileListXRD[0][-12:-9]) # checks first files scan # (from file name)

        #Identify which scans are LFP or sep
        LFPscans = [*range(527, 636, 2)] #Uses actual scan numbers here, not 0 based index. LFP starts again at 527 and ends with 635 (scan 636 is the last LTO scan)
        Sepscans = [x + 1 for x in LFPscans]


        for i in range(len(fileListXRD)): # For each file in fileList
            print('Converting File ' + str(i) + ' of ' + str(len(fileListXRD)))
            scanNum = int(fileListXRD[i][-12:-9]) # last digits of file for scan #
            posNum = int(fileListXRD[i][-7:-4]) # last digits of file for position
            #if ((scanNum-firstScanNum) % 2) != 0 and scanNum < 11: # if its opposite (odd or even) to first file then add 50 to the position and -1 from scan# to add it to the previous scan
                #posNum = posNum + 50  # add 50 to position to continue from last scan
            if scanNum in self.removeListXRD: #ignore scans in removeList. Should not be present anyways.
                print('SKIPPING XRD SCAN: ' + str(i))
                continue
            if scanNum < 527 or scanNum > 635: # All scans done at the fast rate
                print('SKIPPING XRD SCAN: ' + str(i))
                continue
            if scanNum in Sepscans:  # Determine if Sep scan
                posNum = posNum + 11  # if Sep scan, add 17.  ELIF will be skipped
                print(str(scanNum) + ' is a Seperator scan')
                pass
            elif scanNum in LFPscans: # If the file being read is LFP
                print(str(scanNum) + ' is a LFP scan')
                pass
            if posNum in allScansXRD: # if this position number exists in the dictionary append data to it
                allScansXRD[posNum].append(numpy.loadtxt(fileListXRD[i], dtype=None, usecols=(0,1)))  # Read the files where column 1 is data and column 0 is 2theta
            else: # if position number does not yet exist in dictionary create the list before appending data
                allScansXRD[posNum] = list() #Where allScansXRD[posNum][scanNum][data for scan][theta and intensity]
                allScansXRD[posNum].append(numpy.loadtxt(fileListXRD[i], dtype=None, usecols=(0,1)))  # Read the files where column 1 is data and column 0 is 2theta
                #print('XRD Scan ' + str(scanNum-1) + ' loaded.')

        for i in range(len(allScansXRD)): #(posNum)(scanNum)(2500 data)(2 data points)
            allScansXRD[i] = numpy.array(allScansXRD[i]) #convert to array (scanNum)(2000 data)(2 data points)
            allScansXRD[i] = allScansXRD[i].transpose((0,2,1)) #Transpose the 2000 data and 2 datapoints (2theta and Int)
            # Now reshape so that we have the 2theta column follow by intensity column for every scan (new shape is (2*scanNum)(2000 data)
            # It's 2*scanNum because now it hold the 2theta and then intensity values (2000 each) sequentially
            # This is done so that we have the form we want to see in an excel sheet (i.e 2d array)
            allScansXRD[i] = allScansXRD[i].reshape((allScansXRD[i].shape[0]*allScansXRD[i].shape[1]), allScansXRD[i].shape[2])
            df = pandas.DataFrame(allScansXRD[i]) #Get dataframe for position i
            df.to_csv('Position ' + str(i) + '.csv', index=False)
            print('XRD CSV Position ' + str(i) + ' created.')

        #with open('Position ' + str(posNum) + '.csv', 'w', newline='') as csvfile:
            #writer = csv.writer(csvfile)
            #writer.writerows(listoflists)


    def get_xrd_files(self): # Get all sheets of all files into dataframes and concatenates them into a single dataframe
        ### ONLY DO ONCE. Convert all the .dat files into .csv files for quick data processing later. ONLY DO ONCE. ###
        #m.group_dat_files_by_position() #Use this to create the .csv's for each position (raw diffractograms only)

        ### Get all raw XRD data from .csv files ###
        os.chdir(f'{self.dir}\XRD\Lima_run2_4Cpart')

        # Lists files in current directory ('.')
        files = os.listdir()

        fileListXRD = list()
        for i in files:  # For all files in directory
            if i.endswith('.csv'):  # Only accept .csv files that are created with m.group_dat_files_by_position()
                fileListXRD.append(i)

        # Sort file names alphanumerically
        fileListXRD = self.sorted_nicely(fileListXRD)

        # Get CSV data into a list
        self.allRawXRDbyPos = list() # where allRawXRDbyPos[pos][scan#]


        for i in range(len(fileListXRD)):
        #for i in range(0,28):
            start = time.time()
            file = pandas.read_csv(fileListXRD[i], delimiter=',')
            #file = dd.read_csv(fileListXRD[i], delimiter=',') #Reads much faster since not all data loaded at once
            end = time.time()
            print("Read csv without chunks: ", (end - start), "sec")
            rawXRDbyPos = [list(row) for row in file.values] # read .csv into a list (scan #) of lists (diffractogram intensities) for each position
            rawXRDbyPos = numpy.asarray(rawXRDbyPos)
            rawXRDbyPos = rawXRDbyPos.reshape(int(rawXRDbyPos.shape[0]/2), 2, rawXRDbyPos.shape[1]) #Reshape for a 3d array that hold both 2theta and Intensity
            self.allRawXRDbyPos.append(rawXRDbyPos) #where self.allRawXRDbyPos[posNum][scanNum][2theta(0) or Int (1)][datapoint]
            print('Raw XRD at Position ' + str(i) + ' uploaded.')


    def baseline_correct_XRD(self): #This will baseline correct all raw XRD diffractograms
        for i in range(len(self.allRawXRDbyPos)): #for each position where self.allRawXRDbyPos[posNum][scanNum][2theta(0) or Int (1)][datapoint]
            posData = self.allRawXRDbyPos[i]
            print('Baseline Correcting XRD @ Position ' + str(i))
            for j in range(len(posData)):
                posScanData = self.allRawXRDbyPos[i][j][1]
                baseObj = BaselineRemoval(posScanData) #where self.allRawXRDbyPos[posNum][scanNum][2theta(0) or Int (1)][datapoint]
                self.allRawXRDbyPos[i][j][1] = baseObj.ZhangFit()

        # plot an example with baseline
        baseObj = BaselineRemoval(self.allRawXRDbyPos[2][0][1])
        baselineCorrected = baseObj.ZhangFit()
        baselineplot = self.allRawXRDbyPos[2][0][1] - baselineCorrected
        plt.plot(self.allRawXRDbyPos[2][0][0], baselineplot, 'r-')
        plt.plot(self.allRawXRDbyPos[2][0][0], self.allRawXRDbyPos[2][0][1], 'k-')
        plt.show()

    def plot_xrd_integrals(self):
        # Plot peak intensity vs Time
        # Variables
        lowerScanNum = 0  # lower scan # to plot
        #upperScanNum = len(self.allXRDIntegralbyType[0]) # for all use: len(self.allIntegralbyType[1]) where self.allIntegralbyType[datatype][scan#][position]. 1 = LFP1, 2 = LFP2, 3 = background, 4 = graphite and 5 = Li graphite
        upperScanNum = 42 #0 is start right after discharge at 2C. 3-14 is the charge at 4C. 32-42 is discharge at 4C, 53 is where CV discharge ends
        stepNum = 1  # Plots every n scans between lower and upper threshold
        lowerPos = 0  # lower position value to plot. LFP starts at pos 1
        upperPos = 10  # upper position value to plot. LFP ends at 11, but can cut last data point cause its bad

        # Choose colours to plot
        colour_subsection = numpy.linspace(0.1, 0.9, math.ceil((upperScanNum - lowerScanNum) / stepNum))  # Splits colourmap an equal # of sections related to # of curves


        self.coloursList, self.coloursList2, self.coloursList3, self.coloursListRatio = [cm.autumn_r(x) for x in colour_subsection], [
            cm.winter_r(x) for x in colour_subsection], [cm.summer_r(x) for x in colour_subsection], [cm.autumn_r(x) for x in colour_subsection]


        # Manual peak interpolation and integration
        LFPpeakMin, LFPpeakMax =  3.31, 3.44 #the (200) peak of LFP
        FPpeakMin, FPpeakMax = 3.452, 3.60 #the (200) peak of FP
        #FPpeakMin, FPpeakMax = 6.370, 6.455  # the (301) peak of FP
        #FPpeakMin, FPpeakMax = 5.925, 6.060  # the (020) peak of FP
        manualIntegralList = [] # [scan# index in selected range][position][0 = LFP peak integral, 1 = FP peak integral]. Holds all scans

        # First loop for each scan
        for i in range(len(self.allRawXRDbyPos[6])): #for each scan. pos 6 chosen cause it has less scans for some reason
            manualIntegralallPos = []  # To hold values at all positions for a single scan
            print('Interpolating and integrating XRD @ Scan ' + str(i))
            # Second loop for each position
            for j in range(lowerPos, upperPos): #and for each position within that scan
                #Get interpolation
                XRDinterp = scipy.interpolate.interp1d(self.allRawXRDbyPos[j][i][0], self.allRawXRDbyPos[j][i][1]) # where self.allRawXRDbyPos[posNum][scanNum][2theta(0) or Int (1)][datapoint]
                #Apply interpolation to x values to get continuous y values
                xInterval = numpy.linspace(self.allRawXRDbyPos[j][i][0][0], 10, num=3000) #first degree value to 10 degrees
                XRDinterpIntensity = XRDinterp(xInterval) # Calculate interpolation from start of x values until arbitrary angle
                #Plot interp over real data
                #plt.plot(xInterval, XRDinterpIntensity, '.b') # Plot that interpolation
                #plt.plot(self.allRawXRDbyPos[22][i][0], self.allRawXRDbyPos[22][i][1], 'r')
                #Integrate over peak angles
                diffracto = numpy.stack((xInterval,XRDinterpIntensity), axis=1) #Stack to have (2th, Int) together
                LFPpeak = [x for x in diffracto if x[0] < LFPpeakMax and x[0] > LFPpeakMin] # Get LFP peak angles #Filter for wanted 2th
                FPpeak = [x for x in diffracto if x[0] < FPpeakMax and x[0] > FPpeakMin]  # Get FP peak angles #Filter for wanted 2th
                LFPpeak, FPpeak = numpy.array(LFPpeak).transpose(), numpy.array(FPpeak).transpose() #Transpose reverses the axes (i.e seperates intensity and 2th again)
                # Draw baseline (straight line) at bottom of integrated peak from first to last selected point in x
                baselineLFPslope, baselineFPslope = (LFPpeak[1][-1] - LFPpeak[1][0])/(LFPpeak[0][-1] - LFPpeak[0][0]), (FPpeak[1][-1] - FPpeak[1][0])/(FPpeak[0][-1] - FPpeak[0][0])  # find slop dY/dX
                baselineLFPyintercept, baselineFPyintercept = LFPpeak[1][0] - (baselineLFPslope * LFPpeak[0][0]), FPpeak[1][0] - (baselineFPslope * FPpeak[0][0])  # find y-intercept (y - mx = b)
                baselineLFP, baselineFP = baselineLFPslope * LFPpeak[0] + baselineLFPyintercept, baselineFPslope * FPpeak[0] + baselineFPyintercept  # baseline is y = mx+b
                # Subtract baseline from peak
                LFPpeakCorr, FPpeakCorr = LFPpeak[1] - baselineLFP, FPpeak[1] - baselineFP
                # Integrate baseline corrected peak
                LFPintegral, FPintegral = scipy.integrate.simpson(LFPpeakCorr, x=LFPpeak[0], even='avg'), scipy.integrate.simpson(FPpeakCorr, x=FPpeak[0], even='avg')  # Integrate using Simpson's rule
                manualIntegralallPos.append((LFPintegral, FPintegral))  # [scan number in chosen range][0 = LFP, 1 = FP]
            manualIntegralList.append(numpy.array(manualIntegralallPos)) # manualIntegralList[scan#][pos#][LFP/FP]
        manualIntegralList = numpy.array(manualIntegralList) # Convert list to array
        manualIntegralList = manualIntegralList.transpose(0, 2, 1)  # transpose pos# as last value manualIntegralList[scan#][LFP=0/FP=1][pos#]
        self.XRDIntegralListforHeatmap = copy.deepcopy(manualIntegralList)

        # Setup Plot
        figXRD, (axXRD, axXRD_Echem) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 0.25]})
        axXRD_Current = axXRD_Echem.twinx()
        #axPeakRatio = axXRD.twinx()

        # Normalize and smooth integrated data. Plot
        for i in range(lowerScanNum, upperScanNum, stepNum):  # [scan# index in selected range][0 = LFP peak integral, 1 = FP peak integral][pos#]
        #for i in [32, 37, 52]:
            # Calibrate XRD to first scan @ OCV
            #OCVLFP, OCVFP = manualIntegralList[0][0][:], manualIntegralList[0][1][:]
            #OCVLFP, OCVFP = 1, 1 # use this to NOT normalize to OCV values (skews results when peak not present in first scan)
            #normalizedLFP, normalizedFP = manualIntegralList[i][0][:] / OCVLFP, manualIntegralList[i][1][:] / OCVFP  # These are the integrated peaks normalized to the first scan
            # Apply Savitzky-Golay data smoothing
            #movingBox = 3
            #normalizedSmoothedLFP = scipy.signal.savgol_filter(normalizedLFP, window_length=movingBox, polyorder=2)
            #normalizedSmoothedFP = scipy.signal.savgol_filter(normalizedFP, window_length=movingBox, polyorder=2)
            #Normalize FP integral based on Form factor ratio
            manualIntegralList[i][1][:] = manualIntegralList[i][1][:] / self.form_factor_ratio_FP_LFP
            # Calculate SoC
            SoCLFP = (manualIntegralList[i][1][:] / (manualIntegralList[i][1][:] + manualIntegralList[i][0][:])) * 100
            # Plot XRD
            '''If you want position in # only...'''
            axXRD.plot(range(lowerPos, upperPos, 1), SoCLFP, '-', color=self.coloursList3[int((i - lowerScanNum) / stepNum)], linewidth=2)  # plot the LFP SoC
            #axXRD.plot(range(lowerPos, upperPos, 1), normalizedLFP, '-', color=self.coloursList[int((i - lowerScanNum) / stepNum)], linewidth=2)  # [scan# index in selected range][0 = LFP peak integral, 1 = FP peak integral][pos#]
            #axXRD.plot(range(lowerPos, upperPos, 1), normalizedFP, '-', color=self.coloursList3[int((i - lowerScanNum) / stepNum)], linewidth=2)
            '''If you want position in microns...'''
            #axXRD.plot(self.XRF_XRD_pos_array[lowerPos:upperPos], SoCLFP, '-', color=self.coloursList3[int((i - lowerScanNum) / stepNum)], linewidth=2) #plot the LFP SoC

        # Plot Echem
        x = self.dfEchem['time/s']
        y = self.dfEchem['Vol(mV)']
        y2 = self.dfEchem["Cur(mA)"]
        axXRD_Echem.plot(x, y, 'k-', linewidth=2)
        axXRD_Current.plot(x, y2, 'b-', linewidth=2)

        # Plot the Echem vertical lines
        for i in range(lowerScanNum, upperScanNum, stepNum):
        #for i in [32, 37, 52]:
            trans = mt.blended_transform_factory(axXRD_Echem.transData, axXRD_Echem.transAxes)  # Transform uses relative ymin and ymax values for vlines
            axXRD_Echem.vlines(x=self.XRF_XRD_time_array[i], color=self.coloursList3[int((i - lowerScanNum) / stepNum)], ymin=0, ymax=1, linestyles='--', linewidth=2, transform=trans)  # Plot all time vertical lines

        # Customize figure Integrated XRD vs Time
        axXRD.set_xlabel('Position (\u03BCm)', fontsize=22)
        #axXRD.set_ylabel('Normalized Intensity', fontsize=22)
        axXRD.set_ylabel('State of Charge (%)', fontsize=22)
        axXRD.tick_params(axis='both', which='major', labelsize=14)
        axXRD.set_xlim(lowerPos, upperPos+7)
        #axXRD.set_xlim(lowerPos, 20)
        axXRD.set_ylim(0, 105)
        #axXRD.legend(['$LiFePO_{4}$ (200)', '$FePO_{4}$ (301)'])
        figXRD.tight_layout()
        figXRD.set_size_inches(6, 6)
        # Customize figure Gradient for Echem
        axXRD_Echem.tick_params(axis='both', which='major', labelsize=self.ticksize)
        axXRD_Echem.set_ylabel('E (V vs. Li)', fontsize=22)
        axXRD_Current.tick_params(axis='both', which='major', labelsize=self.ticksize)
        axXRD_Current.set_ylabel('I (mA)', fontsize=22)
        axXRD_Echem.set_xlabel('Time (Minutes)', fontsize=22)
        axXRD_Echem.tick_params(labelsize=self.ticksize)
        # axXRF_Echem.set_xlim(self.timestart, self.timeend)
        axXRD_Echem.set_ylim(0.8, 2.7)
        axXRD_Echem.set_xlim(self.XRF_XRD_time_array[lowerScanNum]-1, self.XRF_XRD_time_array[upperScanNum])
        # Hide the right and top spines
        axXRD_Echem.spines['right'].set_visible(False)
        axXRD_Echem.spines['top'].set_visible(False)

        # Adjust spacing of subplots
        figXRD.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.25)
        figXRD.set_size_inches(8, 8)
        figXRD.tight_layout()

        plt.show()

        #Save figure as svg
        os.chdir('C:/Users/Jeremy Dawkins/Documents/ESRF_Full_Data_2022/SynologyDrive/id31/id31/Data Treatment')
        #figXRD.savefig('Figure_ManualXRDIntegrals.svg', format='svg', dpi=600)

        print('Plotted')

    def makeGIFXRDatPosition(self):### Make a GIF of raw XRD spectra vs time at a fixed position ###
        posToGIF = 5
        scanLimits = (0, 54)  # 0 to 54 is all
        duration = 10 / len(self.allRawXRDbyPos[posToGIF][scanLimits[0]:scanLimits[1]+1])  # Duration for each frame in GIF
        # Choose colours for plotting
        colour_subsection = numpy.linspace(0.1, 0.9, len(self.allRawXRDbyPos[0]))  # Splits colourmap an equal # of sections related to # of curves
        self.coloursList = [cm.autumn_r(x) for x in colour_subsection]

        # Create x, y values
        dataXRDGif = self.allRawXRDbyPos[posToGIF][scanLimits[0]:scanLimits[1]+1] #where [PosNum][scanNum][2theta/Int]. +1 for correct indexing
        #transpose data for GIF (such that we have [2theta/Int] before [scanNum]
        dataXRDGif = dataXRDGif.transpose((1, 0, 2))
        dataXRDGifIntensity = dataXRDGif[1] #Take the intensity column
        dataXRDGif2theta = dataXRDGif[0] #Take the 2theta column

        # create the GIF for one position.
        self.make_gif_celluloid(dataXRDGif2theta, dataXRDGifIntensity, 'GIF_XRD_Pos_' + str(posToGIF), duration, self.coloursList, 3, 5.2, 0.0001, 1000000000)  # using coloursList[color] uses the functions autumn colour map



        ### Create a GIF for all positions for one scan ###
        scanToGIF = 1
        duration = 0.2  # Duration for each frame in GIF

        # Get y values
        plotData = list()
        xValues = list()
        for i in range(len(self.allRawXRDbyPos)):
            plotData.append(self.allRawXRDbyPos[i][scanToGIF][1]) #where [PosNum][scanNum][2theta/Int]
            xValues.append(self.allRawXRDbyPos[i][scanToGIF][0]) #where [0] at end is 2theta column

        #self.make_gif_celluloid(xValues, plotData, 'GIF_XRD_Scan_'+str(scanToGIF), duration, 'black', 0, 2000,0, 0.0015)

    def reference_XRD(self): #Gets the reference LFP and FP peaks that were simulated using GSAS-2. Also gets 2theta values.
        # Get reference peaks for LFP and FP
        os.chdir(f'{self.dir}\Simulated_Powder_Diffraction')
        with open('LiFePO4_CODid_2100916.pkslst', 'r') as lfp, open('FePO4_CODid_1525576.pkslst', 'r') as fp:  # remove brackets from file
            headerText = (next(lfp), next(fp))  # removes headers and saves seperately
            text = (lfp.read(), fp.read())  # reads the text files
            peakData = [re.sub(r"[\([{})]", "", i) for i in text]  # remove brackets except close brackets from files
            for i in range(len(peakData)): #for each element in text (i.e for each simulated PXRD)
                peakData[i] = re.sub(r"[\]]", ",", peakData[i])  # now replace close brackets ] by ,
                peakData[i] = numpy.fromstring(peakData[i], sep=',')  # Get the value strings in a numpy array. Seperate by ,
                peakData[i] = numpy.delete(peakData[i], -1)  # removes last value which is erroneously added because of the last close bracket
                peakData[i] = numpy.reshape(peakData[i], (int(len(peakData[i]) / 8), 8))  # reshape the array to get each line seperately (each line has 8 numbers)
        self.LFPpeakPos, self.FPpeakPos = [peakData[0][i][0] for i in range(len(peakData[0]))], [peakData[1][i][0] for i in range(len(peakData[1]))]  # Get just the peak position in numpy array

    def rietveld(self): #Used to perform Rietveld analysis on one or multiple diffractograms
        scanNum = 0  # Choose scan number (time)
        scanNumRange = [30, 37, 43] # Or choose a range
        #scanNumRange = range(5, 40, 1) # Or choose a range
        posNum = 18  # Choose position in electrode. From 0 (LFP) to 21 (LTO)

        # plot
        figRietveld, axRietveld = plt.subplots()

        #Plot mutliple
        colour_subsection = numpy.linspace(0.1, 0.9, len(scanNumRange))  # Splits colourmap an equal # of sections related to # of curves
        self.coloursList = [cm.summer_r(x) for x in colour_subsection]
        for i in range(len(scanNumRange)):
            baseObj = BaselineRemoval(self.allRawXRDbyPos[posNum][scanNumRange[i]][1])
            baselineCorrected = baseObj.ZhangFit()
            baselineplot = self.allRawXRDbyPos[posNum][scanNumRange[i]][1] - baselineCorrected
            axRietveld.plot(self.allRawXRDbyPos[posNum][scanNumRange[i]][0], baselineCorrected, marker='+' ,linewidth=2, color=self.coloursList[i]) #baseline corrected



        #Plot baseline
        baseObj = BaselineRemoval(self.allRawXRDbyPos[posNum][scanNum][1])
        baselineCorrected = baseObj.ZhangFit()
        baselineplot = self.allRawXRDbyPos[posNum][scanNum][1] - baselineCorrected
        #plt.plot(self.allRawXRDbyPos[posNum][scanNum][0], baselineplot, 'r-') #plot the baseline

        #Plot only one
        #axRietveld.plot(self.allRawXRDbyPos[posNum][scanNum][0], baselineCorrected, marker='+', color='k', linewidth=2) #baseline corrected
        #axRietveld.plot(self.allRawXRDbyPos[posNum][scanNum][0], self.allRawXRDbyPos[posNum][scanNum][1], marker='+', color='black', linewidth=2) #raw

        #Plot reference peaks
        trans = mt.blended_transform_factory(axRietveld.transData, axRietveld.transAxes) #Transform uses relative ymin and ymax values for vlines
        #axRietveld.vlines(x=self.LFPpeakPos, ymin=0, ymax=0.03, color='r', linestyles='-', linewidth=2, transform=trans) #Plot all LFP reference peaks.
        #axRietveld.vlines(x=self.FPpeakPos, ymin=0, ymax=0.03, color='b', linestyles='-', linewidth=2,transform=trans)  # Plot all FP reference peaks.

        # Customize figure Integrated XRD vs Time
        # axRietveld.set_xlabel('Depth (\u03bcm)', fontsize=22)
        axRietveld.set_xlabel('2\u03F4 (degrees)', fontsize=22)
        axRietveld.set_ylabel('Intensity (counts)', fontsize=22)
        # self.axRietveld.set_ylabel('Li$^+$ Concentration (M)', fontsize=22)
        #axRietveld.set_xlim(3, 5.2) #For LFP
        axRietveld.set_xlim(20, 21) #For LTO
        #axRietveld.set_ylim(0.0001, 0.001)
        axRietveld.tick_params(axis='both', which='major', labelsize=14)
        # axRietveld.legend(['As', 'Fe', 'Cr'])
        figRietveld.tight_layout()
        figRietveld.set_size_inches(8, 8)

        plt.show()


    def XRF_XRD_time(self):  # Gets array of START time for each XRD scan as: self.XRD_time_array[scan#]
        '''Get time'''
        XRF_time_list = list()  # List to hold all times relative to first scan
        XRF_raw_times = self.allIntegralbyType[7][:][:]   # [Datatype][scan#][position], where 6 = elapsed time for measurement, 7 = epoch time

        for i in range(len(XRF_raw_times)):  # For each scan where i = scan. [scan#][position]
            try:
                #dataset_start_time = XRF_raw_times[0][0] #Start time of whole dataset (in epoch)
                dataset_start_time = 1644946580.3361378 #Start time of the first Lima_run2 scans (in epoch)
                time_at_scan = XRF_raw_times[i][0]  # This takes the start time of the scan
                XRF_time_list.append(time_at_scan - dataset_start_time)  # This is a list of time elapsed relative to first scan start time.
            except TypeError:
                print('Scan ' + str(i) + ' has no final time')
                XRF_time_list.append(0)
                continue
        self.XRF_XRD_time_array = numpy.array(XRF_time_list[:-1])  # convert to array. This is the time (secs) when each scan STARTED so drop the last value for numbers to add up
        self.XRF_XRD_time_array = self.XRF_XRD_time_array/60 + self.timestart #convert to minutes and add start time
        print('XRF-XRD Time Array Constructed')

        '''Get position array'''
        posList =  self.allIntegralbyType[8][0][:]  # [Datatype][scan#][position], where 8 = saz2 (i.e. position)
        self.XRF_XRD_pos_array = (posList - posList[0]) * 1000 #get positions relative to initial value, and in microns
        print('XRF-XRD Position Array Constructed')

    def XRD_2d_plot(self): #Plots 2d map of XRD intensities (colour) vs time (y-axis) at all angles (x-axis) for a single position
        #Includes Echem subplot so initiate that data first

        #Params
        position = 3  # Position in the electrode to plot. Pos 3 for LFP is good. Pos 17 for LTO is good.
        scanLower = 0 #Start scan #
        scanUpper = 53 #End scan #
        EchemtimeStart = self.XRF_XRD_time_array[scanLower] # Used to plot Echem so times match to XRD times.
        EchemtimeEnd = self.XRF_XRD_time_array[scanUpper]

        #Get Z-values (Intensity)
        XRD_2d_array = numpy.asarray(self.allRawXRDbyPos[position][:][:])  # where self.allRawXRDbyPos[posNum][scanNum][2theta(0) or Int (1)][datapoint]
        XRD_2d_array = XRD_2d_array.transpose(1, 0, 2) #swap to [2theta(0) or Int (1)][scanNum][datapoint]
        XRD_2d_array = XRD_2d_array[1] # Keep only intensities. Now [scanNum][datapoint]
        XRD_2d_array = XRD_2d_array[scanLower:(scanUpper+1)] #Filter for requested scan numbers. +1 for proper indexng

        #Get X-Values
        Xvals2d = self.allRawXRDbyPos[position][0][0] #Takes 2theta from first scan at same position
        Potx = self.dfEchem['Vol(mV)']  # Echem Potential

        #Get Y-Values
        #Yvals2d = range(scanLower, scanUpper) #if scan # required
        Yvals2d = self.XRF_XRD_time_array[scanLower:(scanUpper+1)] #if time (minutes) required
        Poty = self.dfEchem['time/s']  # Elapsed Echem Time

        #Plot
        # Setup
        figXRD_2d, (axXRD_2d, axXRD_2d_Echem) = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [1, 0.25]}) #Setup XRD/Echem subplots
        # XRD
        axXRD_2d.pcolormesh(Xvals2d, Yvals2d, XRD_2d_array, cmap='RdYlGn_r')  # X, Y and Z where Z is 2d array of intensity values

        # Echem
        axXRD_2d_Echem.plot(Potx, Poty, 'k-', linewidth=2)

        # Reference peaks
        trans = mt.blended_transform_factory(axXRD_2d.transData, axXRD_2d.transAxes) #Transform uses relative ymin and ymax values for vlines
        axXRD_2d.vlines(x=self.LFPpeakPos, ymin=0, ymax=0.05, color='r', linestyles='-', linewidth=2, transform=trans) #Plot all LFP reference peaks.
        axXRD_2d.vlines(x=self.FPpeakPos, ymin=0, ymax=0.05, color='b', linestyles='-', linewidth=2,transform=trans)  # Plot all FP reference peaks.

        #Format Plot
        # axXRD_2d.set_xlabel('Depth (\u03bcm)', fontsize=22)
        axXRD_2d.set_xlabel('2\u03F4 (degrees)', fontsize=22)
        axXRD_2d.set_ylabel('Time (Minutes)', fontsize=22)
        #self.axXRD_2d.set_ylabel('Li$^+$ Concentration (M)', fontsize=22)
        #axXRD_2d.set_xlim(3, 5.2) #for LFP
        axXRD_2d.set_xlim(20, 21)  # for LFP
        #axXRD_2d.set_ylim(0.0001, 0.001)
        axXRD_2d.tick_params(axis='both', which='major', labelsize=14)
        #axXRD_2d.legend(['As', 'Fe', 'Cr'])
        figXRD_2d.tight_layout()
        figXRD_2d.set_size_inches(8, 8)
        #Echem
        axXRD_2d_Echem.tick_params(axis='both', which='major', labelsize=self.ticksize)
        axXRD_2d_Echem.set_xlabel('E (V vs. Li)', fontsize=22)
        axXRD_2d_Echem.set_ylabel(None, fontsize=22)
        axXRD_2d_Echem.tick_params(labelsize=self.ticksize)
        axXRD_2d_Echem.set_ylim(EchemtimeStart, EchemtimeEnd)
        axXRD_2d_Echem.invert_xaxis()
        #axXRD_2d_Echem.set_xlim(0.8, 2,7)
        # Hide the right and top spines and left ticks
        axXRD_2d_Echem.spines['right'].set_visible(False)
        axXRD_2d_Echem.spines['top'].set_visible(False)
        #axXRD_2d_Echem.spines['left'].set_visible(False)
        axXRD_2d_Echem.tick_params(axis='y',which='both', left=False,  right=False, labelleft=False) #remove yticks echem

        plt.show()

    def gradient_and_SoC_gif(self):  # Animates and saves a GIF the conc. gradient with bar charts in the background as SoC
        os.chdir('C:/Users/Jeremy Dawkins/Documents/ESRF_Full_Data_2022/SynologyDrive/id31/id31/Data Treatment')

        '''variables'''
        lowerPos = 0
        upperPos = 21


        figGradientSoC, axGradientSoC = plt.subplots()
        axGradientSoC_bars = axGradientSoC.twinx()

        # Customize figure ScanGIF
        # axScanGIF.set_xlabel('Depth (\u03bcm)', fontsize=22)
        axGradientSoC.set_xlabel('Depth (#)', fontsize=22)
        # axScanGIF.set_ylabel('As K\u03B1', fontsize=22)
        axGradientSoC.set_ylabel('Concentration (M)', fontsize=22)
        axGradientSoC_bars.set_ylabel('SoC (%)', fontsize=22)
        axGradientSoC_bars.tick_params(axis='both', which='major', labelsize=14)
        # self.axScanGIF.set_ylabel('Li$^+$ Concentration (M)', fontsize=22)
        axGradientSoC_bars.set_ylim(-5, 110)
        axGradientSoC.tick_params(axis='both', which='major', labelsize=14)
        # axScanGIF.legend(['As', 'Fe', 'Cr'])
        figGradientSoC.set_size_inches(8, 8)
        figGradientSoC.tight_layout()

        camera = Camera(figGradientSoC)
        duration = 0.2  # Time each image shows up

        '''Get data and plot one at a time'''
        for i in range(len(self.normalizedAs)):
            SoCLFP = (self.XRDIntegralListforHeatmap[i][1][:] / (self.XRDIntegralListforHeatmap[i][1][:] + self.XRDIntegralListforHeatmap[i][0][:])) * 100 #Get LFP SoC
            axGradientSoC.plot(range(len(self.normalizedAs[i])), self.normalizedAs[i], '-', color='red', linewidth=2)  # Get sol'n data
            axGradientSoC_bars.bar(range(11, len(SoCLFP)+11, 1), SoCLFP, color='green', linewidth=2, alpha=0.5)  # LFP

            # Add Custom Text
            axGradientSoC.text(0.75, 0.9, 'Scan. ' + str(i), size=22, color='black', transform=axGradientSoC.transAxes)

            # Snapshot
            figGradientSoC.tight_layout()
            camera.snap()  # take a snapshot of the figure in the current state

        animation = camera.animate()  # Animate these snapshots
        animation.save('Gradient and SoC Animated Scan.gif', writer='Pillow', fps=1 / duration)

    def animate_XRD_scan_gif(self):  # Animates and saves a GIF of a XRD scan being made point by point
        figXRDScanGIF, axXRDScanGIF = plt.subplots()

        # Customize figure ScanGIF
        # axScanGIF.set_xlabel('Depth (\u03bcm)', fontsize=22)
        axXRDScanGIF.set_xlabel('Depth (#)', fontsize=22)
        # axScanGIF.set_ylabel('As K\u03B1', fontsize=22)
        axXRDScanGIF.set_ylabel('Intensity (counts)', fontsize=22)
        axPeakRatio = axXRDScanGIF.twinx()
        axPeakRatio.set_xlabel('Ratio', fontsize=22)
        axPeakRatio.tick_params(axis='both', which='major', labelsize=14)
        # self.axScanGIF.set_ylabel('Li$^+$ Concentration (M)', fontsize=22)
        axXRDScanGIF.set_xlim(0, 98)
        axXRDScanGIF.tick_params(axis='both', which='major', labelsize=14)
        # axScanGIF.legend(['As', 'Fe', 'Cr'])
        figXRDScanGIF.set_size_inches(6, 6)
        figXRDScanGIF.tight_layout()

        camera = Camera(figXRDScanGIF)
        duration = 0.2  # Time each image shows up

        for j in range(len(self.allXRDIntegralbyType[1][0])): #1 = LFP1, 2 = LFP2, 3 = background, 4 = graphite and 5 = Li graphite
            axXRDScanGIF.plot(self.allXRDIntegralbyType[0][0][0:j], '.', color='red', linewidth=2) #LFP
            axXRDScanGIF.plot(self.allXRDIntegralbyType[1][0][0:j], '.', color='green', linewidth=2) #FP
            #peakRatio = self.allXRDIntegralbyType[1][0][0:j]/self.allXRDIntegralbyType[0][0][0:j]
            percentageLithiation = (self.allXRDIntegralbyType[1][0][0:j]/(self.allXRDIntegralbyType[1][0][0:j]+self.allXRDIntegralbyType[0][0][0:j]))*100
            #axXRDScanGIF.plot(self.allXRDIntegralbyType[2][0][0:j], '.', color='black', linewidth=2)
            #axPeakRatio.plot(peakRatio, '.', color='black', linewidth=2)
            axPeakRatio.plot(percentageLithiation, '.', color='black', linewidth=2)

            # Add Custom Text
            axXRDScanGIF.text(0.75, 0.9, 'Pos. ' + str(j), size=22, color='black', transform=axXRDScanGIF.transAxes)

            # Snapshot
            figXRDScanGIF.tight_layout()
            camera.snap()  # take a snapshot of the figure in the current state

        animation = camera.animate()  # Animate these snapshots
        animation.save('XRD Animated Scan.gif', writer='Pillow', fps=1 / duration)

    def make_gif_celluloid(self, datasetX, datasetY, imageFolder, duration, colour, xlimbot, xlimtop, ylimbot, ylimtop):
        #This function interates over the length of the dataset (list of plot values), creates a folder and saves a GIF of those plots.
        # datasetX/Y = h5 dataset type or list of plot values (if no Y, use None), imageFolder = name of folder to create, duration = time for each frame
        # colour = the colour to plot. Use coloursList[color] for the functions default "autumn" colourmap

        # First make the folder and get in there
        if not os.path.exists(str(self.dir) + '/' + str(imageFolder)):
            os.mkdir(str(self.dir) + '/' + str(imageFolder))  # Make the folder if it doesn't exist
        os.chdir(str(self.dir) + '/' + str(imageFolder))  # Get to the image folder
        print(os.getcwd())

        #Second make the plots and animate using celluloid
        figGif, axGif = plt.subplots() #Make a figure
        #axGif.set_xlabel('Energy (eV)', fontsize=22)
        axGif.set_xlabel('Diffraction Angle (2\u03F4)', fontsize=22)
        #axGif.set_xlabel('Time (mins)', fontsize=22)
        axGif.set_ylabel('Intensity (counts)', fontsize=22)
        #axGif.set_ylabel('Potential (V vs Li)', fontsize=22)
        axGif.set_ylim(ylimbot, ylimtop)  # Format plot
        axGif.set_xlim(xlimbot, xlimtop)  # Format plot
        # self.axGif.set_ylabel('Li$^+$ Concentration (M)', fontsize=22)
        axGif.tick_params(axis='both', which='major', labelsize=14)
        # axGif.legend(['As', 'Fe', 'Cr'])
        figGif.tight_layout()
        figGif.set_size_inches(6, 6)

        camera = Camera(figGif) #Apply celluloid's camera function to the fig


        for i in range(len(datasetX)): #For all element in dataset
            if type(colour) != str: #if the colour is a list instead of a string
                colourscan = colour[i]
            else:
                colourscan = colour

            #Plot
            axGif.plot(datasetX[i], datasetY[i], color=colourscan, linewidth=3) #plot XRF/XRD
            #axGif.plot(datasetX[0:i], datasetY[0:i], color=colourscan, linewidth=5) #plot data echem
            #axGif.plot(datasetX[i], datasetY[i], 'o', color=colourscan, markersize=10) #for echem plot ball at current point

            #Plot vertical lines for reference
            trans = mt.blended_transform_factory(axGif.transData, axGif.transAxes)  # Transform uses relative ymin and ymax values for vlines
            axGif.vlines(x=self.LFPpeakPos, ymin=0, ymax=0.05, color='r', linestyles='--', linewidth=2, transform=trans)  # Plot all reference peaks.
            axGif.vlines(x=self.FPpeakPos, ymin=0, ymax=0.05, color='b', linestyles='--', linewidth=2,transform=trans)  # Plot all reference peaks.
            #axGif.vlines(x=self.Li2CO3peakPos, ymin=0, ymax=1, color='g', linestyles='--', linewidth=1, transform=trans)

            #Plot gaussian fit to As Ka peak

            # Gaussian fit As peak
            #AspeakMax = 10800
            #AspeakMin = 10250
            #spectrum = numpy.stack((datasetX[i], datasetY[i]), axis=1)  # Stack to have (eVs, Int) together
            #Aspeak = [x for x in spectrum if x[0] < AspeakMax and x[0] > AspeakMin]
            #Aspeak = numpy.array(Aspeak).transpose()
            #xdat = Aspeak[0]
            #ydat = Aspeak[1]
            #p0 = [100, 10525, 5]  # initial guess (height, mean, stdev)
            #popt, _ = optimize.curve_fit(self.gaussian, xdat, ydat, p0=p0)  # curve_fit(function, x, y)
            #axGif.plot(xdat, self.gaussian(xdat, *popt), 'b-') #plot

            #Add Custom Text
            #axGif.text(0.75, 0.9, 'Pos. ' + str(i), size=22, color='black', transform=axGif.transAxes)
            axGif.text(0.65, 0.9, 'Scan ' + str(i), size=22, color='black', transform=axGif.transAxes)
            #axGif.text(0.65, 0.9, 'Scan ' + str(int(i/self.multiplier)), size=22, color='black', transform=axGif.transAxes) #for echem

            #Snapshot
            camera.snap() #take a snapshot of the figure in the current state

        animation = camera.animate() #Animate these snapshots
        animation.save('celluloid_gif.gif', writer='Pillow', fps=1/duration)


m = Main()

#Echem
m.get_echem_files()
#m.echem_animation(55, 165, 206) # Total frames (add +1 to frames from XRF/XRD), start time, end time

#XRF
m.get_xrf_files()
#m.baseline_correct_XRF() #Slow and unescessary
m.XRF_XRD_time()  # Gets the XRF time array. Should be identical to the XRD one.
#m.plot_xrf_raw_spectra()
m.plot_xrf_linscans_manual_integral() #Use my integration. Also plots echem
#m.animate_scan_gif() #This animates and saves a GIF of a scan being made point by point (one scan, all positions)
#m.XRF_GIF_Pos() #Raw XRF spectrum plotted (GIF) at a single position
#m.XRF_GIF_Scan() #Raw XRF spectrum plotted (GIF) at all positions for a single scan. Or can make figure for the paper.

  
#XRD
#m.group_dat_files_by_position() #Only need to do once. Converts XRD files into .csv files by position
m.get_xrd_files() #Gets all XRD data from appropriate .csv files created with group_dat_files_by_position
#m.baseline_correct_XRD() #Baseline correct the raw XRDs. Will apply to all other functions as it changes the base file. Slow and unecessary.
#m.XRF_XRD_time() #Gets the XRD time array. Should be identical to the XRF one. Must run both get_xrd_files and get_xrf_files
m.reference_XRD() #Gets reference LFP and FP peaks. Also gets 2theta values from reference peaks in GSAS-2
#m.XRD_2d_plot() #Makes a 2d plot showing the measured intensity (colour) vs time (y-axis) at all angles (x-axis)
#m.makeGIFXRDatPosition() #Used to make GIF of XRD integrals vs time at a certain position
#m.animate_XRD_scan_gif()  # Integrates XRD peaks and plots point by point
#m.rietveld() #Plots a single spectrum for refinement (or publication)
m.plot_xrd_integrals()
#m.gradient_and_SoC_gif()

#Heatmap
m.heatmap_XRF() #Must be after XRD is ran because plots the Avg SoC

#Echem Plot
#m.plot_echem()
