import os
import numpy as np
import math
from scipy.io import wavfile

'''
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Please dont use this file. It is work in progress and may contain errors.
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

'''

def GaussianSpectrum(input, increment, winLength, samprate):


        # Enforce even winLength to have a symmetric window
        if winLength%2 == 1: winLength = winLength +1

        # Make input it into a row vector if it isn't
        if input.shape[0] > 1: input = input.transpose()

        # Padd the input with zeros
        pinput = np.zeros([1,input.shape[1]+winLength])
        pinput[winLength/2:winLength/2+input.shape[1]] = input
        inputLength = pinput.shape[1]

        # The number of time points in the spectrogram
        frameCount = math.floor((inputLength-winLength)/increment)+1

        # The window of the fft
        fftLen = winLength


        ########################
        # Guassian window 
        ########################
        nstd = 6                   # Number of standard deviations in one window.
        wx2 = (np.linspace(1, nstd, nstd)-((winLength+1)/2))**2
        wvar = (winLength/nstd)**2
        ws = math.exp(-0.5*(wx2/wvar))

        ##################################
        # Initialize output "s" 
        ##################################
        if fftLen%2==0:  s = np.zeros([(fftLen+1)/2+1, frameCount]) # winLength is odd
        else: s = np.zeros([fftLen/2+1, frameCount]) # winLength is even 
        

        pg = np.zeros([1, frameCount])
        
        for i in range(frameCount):
                start = (i-1)*increment + 1
                last = start + winLength - 1
                
                f = np.zeros([fftLen, 1])
                f[0:winLength] = ws*pinput[start:last]
                
                pg(i) = np.std(f[:winLength])
                
                #\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\\/\/\/\/\/\/\/\/\/
                specslice = np.fft.fft(f)
                
                s[:,i] = specslice[:((fftLen+1)/2+1)] if fftLen%2==0 else specslice[:(fftLen/2+1)]
        
        # Assign frequency_label
        select = np.linspace(1,(fftLen+1)/2, ((fftLen+1)/2)+1) if fftLen%2==0 else np.linspace(1,fftLen/2+1, (fftLen/2)+1+1)
        fo = np.matmul((select-1).transpose(),samprate/fftLen)

        # assign time_label
        to = np.linspace(1, s.shape[1]-1, s.shape[1]).transpose()*(increment/samprate)
        return s, to, fo, pg




def timefreq(audioWaveform, sampleRate, typeName, params):
        tfrep = params
        tfrep['params']['rawSampleRate'] = sampleRate
        
        if typeName == 'stft':
                #compute raw complex spectrogram
                twindow = tfrep['params']['nstd']/(tfrep['params']['fband']*2.0*math.pi)   # Window length
                winLength = int(twindow*sampleRate)  # Window length in number of points
                winLength = int(winLength/2)*2 # Enforce even window length
                
                #increment = fix(0.001*sampleRate) # Sampling rate of spectrogram in number of points - set at 1 kHz
                # increment = fix(sampleRate/5000) # Sampling rate of spectrogram in number of points - set at 5 kHz
                increment = int(sampleRate/10000) # Sampling rate of spectrogram in number of points - set at 10 kHz (same dt as Dynasim)
                
                #\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
                s, t0, f0, pg = GaussianSpectrum(audioWaveform, increment, winLength, sampleRate) 
                #/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
                
                #normalize the spectrogram within the specified frequency range
                maxIndx = np.where(f0 >= tfrep['params']['high_freq'])[0]
                minIndx = np.where(f0 < tfrep['params']['low_freq'])[-1]+1
                
                normedS = math.abs(s[minIndx:maxIndx, :]) #<<<<<<<<<<<<<<<< s is output spectrogram
                
                #set tfrep values
                fstep = f0[1]
                tfrep['t'] = t0
                tfrep['f'] = np.linspace(f0[minIndx], f0[maxIndx], fstep * (f0[maxIndx]-f0[minIndx]+1)) #f0(minIndx):fstep:f0(maxIndx)
                tfrep['spec'] = normedS #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< final output
        else:
                raise Exception(f'Unknown time-frequency representation type: {typeName}')
        
        return tfrep


def preprocSound(audioWaveforms, params):
        
        # TODO: implement 'wavelet' and 'lyons' tf representations if needed
        allowedTypes = ['stft'] 
        if params['tfType'] not in allowedTypes:
                raise Exception(f'Unknown time-frequency representation type: {params['tfType']}')

        stimStructs = {}
        maxPower = -1
        
        for idx, waveform in enumerate(audioWaveforms):    
        
                stim = {}        
                stim['tfrep'] = timefreq(waveform, params['rawSampleRate'], params['tfType'], params['tfParams'])  
                stim['stimLength'] = stim['tfrep']['spec'].shape[1] / params['stimSampleRate'] #size(stim.tfrep.spec, 2)
                stim['sampleRate'] = params['stimSampleRate']

                stimStructs[f'waveform_{idx}']=stim
                
                if params['numStimFeatures'] == -1: params['numStimFeatures'] = stim['tfrep']['spec'].shape[0] #size(stim.tfrep.spec, 1)
                if params['f'] == -1: params['f'] = stim['tfrep']['f']
                
                params['totalStimLength'] = params['totalStimLength'] + stim['tfrep']['spec'].shape[1] #size(stim.tfrep.spec, 2)
        
                maxPower = max([maxPower, max(max(stim['tfrep']['spec']))])        
        
        ## normalize spectrograms and take log if requested
        if params['tfParams']['log']:
                refpow = maxPower if params['tfParams']['refpow'] == 0 else params['tfParams']['refpow']
                for key in stimStructs.keys():          
                        stim = stimStructs[key]
                        stim['tfrep']['spec'] = max(0, 20*math.log10(stim['tfrep']['spec']/refpow)+params['tfParams']['dbnoise'])
                        stimStructs[key] = stim

        ## concatenate stims into big matrix and record info into struct
        stimInfo = {}
        stimInfo['stimLengths'] = np.zeros([1, len(audioWaveforms)]) #zeros(1, length(audioWaveforms))
        stimInfo['sampleRate'] = params['stimSampleRate']
        stimInfo['numStimFeatures'] = params['numStimFeatures']
        stimInfo['tfType'] = params['tfType']
        stimInfo['tfParams'] = params['tfParams']
        stimInfo['f'] = params['f']

        groupIndex = np.zeros([1, params['totalStimLength']])
        wholeStim = np.zeros([params['totalStimLength'], params['numStimFeatures']])
        
        cindx = 0
        for idx, key in enumerate(stimStructs.keys()):
        
                stim = stimStructs[key]
                slen = stim['tfrep']['spec'].shape[1] #size(stim.tfrep.spec, 2)
                tend = cindx + slen - 1
                
                wholeStim[cindx:tend, :] = stim['tfrep']['spec'].transpose()
                groupIndex[cindx:tend] = idx
                stimInfo['stimLengths'][idx] = slen / params['stimSampleRate']
                
                cindx = tend + 1

        return wholeStim, groupIndex, stimInfo, params

def STRFspectrogram(stim, fs):
        preprocStimParams = {}
        preprocStimParams['tfType'] = 'stft' #use short-time FT
        preprocStimParams['rawSampleRate']=fs
        preprocStimParams['f']=-1
        preprocStimParams['totalStimLength'] = 0
        preprocStimParams['numStimFeatures'] = -1
        preprocStimParams['stimSampleRate'] = 10000 # match dynasim sampling rate, or 1000, 5000
        
        
        tfParams = {}
        tfParams['high_freq'] = 8000       #specify max freq to analyze
        tfParams['low_freq'] = 500         #specify min freq to analyze
        tfParams['log'] = 1                #take log of spectrogram
        tfParams['dbnoise'] = 80           #cutoff in dB for log spectrogram, ignore anything below this
        tfParams['refpow'] = 0             #reference power for log spectrogram, set to zero for max of spectrograms across stimuli
        
        preprocStimParams['tfParams'] = tfParams


        preprocStimParams['outputDir'] = 'cache' #directory to save temp files
        os.makedirs(preprocStimParams['outputDir'], exist_ok=True)
        
        ## use preprocSound to generate spectrogram
        stim_spec, groupIndex, stimInfo, preprocStimParams = preprocSound(stim, preprocStimParams)
        
        
        tInc = 1 / stimInfo['sampleRate'] # generate corresponding timeline for spectrogram
        
        t = np.linspace(0, (stim_spec.shape[0]-1)*tInc, stim_spec.shape[0]/tInc)  # 0:tInc:(size(stim_spec, 1)-1)*tInc
        f=stimInfo['f']
        
        return stim_spec, t, f

sigma = 24 #60 for bird but 38 for mouse
tuning = 'mouse' #'bird' or 'mouse'
stimGain = 0.5
targetlvl = 0.01
maskerlvl = 0.01 #default is 0.01
maxWeight = 1 #maximum mixed tuning weight capped at this level.

paramH_alpha = 0.01 # time constant of temporal kernel [s] 0.0097
paramH_N1 = 5
paramH_N2 = 8
paramH_SC1 = 1 #Nominally 1
paramH_SC2 = 0.88  #increase -> more inhibition #0.88 in paper

strfGain = 0.1

# frequency parameters - static
paramG_BW = 2000 # Hz
paramG-_BSM = 5.00E-05 # 1/Hz=s best spectral modulation
paramGf0 = 4300 # ~strf.f(30)


data_path = r'D:\School_Stuff\Rotation_1_Sep_Nov_Kamal_Sen\Code\SenLabModeling\resampled-stimuli'

s1_sampling_rate, s1_audio_data = wavfile.read(os.path.join(data_path, '200k_target1.wav'))
s2_sampling_rate, s2_audio_data = wavfile.read(os.path.join(data_path, '200k_target2.wav'))


for trial in range(10):
        fs, masker = wavfile.read(f'200k_masker{str(trial)}.wav')
        spec,_,_ = STRFspectrogram(masker/rms(masker)*maskerlvl,fs)
        masker_specs{trial} = spec