import os
import numpy as np
import math
import sys


from scipy.io import wavfile
from tqdm import tqdm, trange

'''
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Please dont use this file. It is work in progress and may contain errors.
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

'''

def rms(data):
        """
        Calculates the Root Mean Square (RMS) of a NumPy array.

        Args:
        data (np.ndarray): The input_ array.

        Returns:
        float or np.ndarray: The RMS value(s).
        """
        return np.sqrt(np.mean(data**2))

def GaussianSpectrum(input_, increment, winLength, samprate, nstd = 6):


        # Enforce even winLength to have a symmetric window
        if winLength%2 == 1: winLength = winLength +1

        # Make input_ it into a row vector if it isn't
        if input_.shape[0] > 1: input_ = input_.transpose()

        # Padd the input_ with zeros
        pinput_ = np.zeros([1,input_.shape[1]+winLength])
        pinput_[:, winLength//2:winLength//2+input_.shape[1]] = input_
        input_Length = pinput_.shape[1]

        # The number of time points in the spectrogram
        frameCount = math.floor((input_Length-winLength)/increment)+1

        # The window of the fft
        fftLen = winLength


        ########################
        # Guassian window 
        ########################                 
        wx2 = (np.linspace(1, winLength, winLength)-((winLength+1)/2))**2
        wvar = (winLength/nstd)**2
        ws = np.exp(-0.5*(wx2/wvar))

        ##################################
        # Initialize output "s" 
        ##################################
        if fftLen%2!=0:  s = np.zeros([int((fftLen+1)/2)+1, frameCount]) # winLength is odd
        else: s = np.zeros([int(fftLen/2)+1, frameCount], dtype=complex) # winLength is even 
        

        pg = np.zeros([1, frameCount])
        
        for i in trange(frameCount):
                start = i*increment
                last = start + winLength
                
                f = ws*pinput_[:, start:last]
                pg[:,i] = np.std(f)
                
                specslice = np.fft.fft(f) #/fftLen ??
                s[:,i] = specslice[:,:(int((fftLen+1)/2)+1)] if fftLen%2!=0 else specslice[:,:(int(fftLen/2)+1)]
        
        # Assign frequency_label
        select = np.linspace(1,int((fftLen+1)/2), int((fftLen+1)/2)) if fftLen%2!=0 else np.linspace(1,int(fftLen/2)+1, int(fftLen/2)+1)
        fo = (select-1)*samprate/fftLen

        # assign time_label
        to = np.linspace(0, s.shape[1]-1, s.shape[1])*(increment/samprate)
        return s, to, fo, pg




def timefreq(audioWaveform, sampleRate, typeName, params):
        tfrep = {}
        tfrep['params'] = params
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
                s, t0, f0, pg = GaussianSpectrum(audioWaveform, increment, winLength, sampleRate, tfrep['params']['nstd']) 
                #/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
                
                #normalize the spectrogram within the specified frequency range
                maxIndx = np.where(f0 >= tfrep['params']['high_freq'])[0][0]
                minIndx = np.where(f0 < tfrep['params']['low_freq'])[0][-1]+1
                
                normedS = np.abs(s[minIndx:maxIndx+1, :]) #<<<<<<<<<<<<<<<< s is output spectrogram
                
                #set tfrep values
                fstep = f0[1]
                tfrep['t'] = t0
                tfrep['f'] = np.linspace(f0[minIndx], f0[maxIndx], 1 + int((f0[maxIndx]-f0[minIndx]+1)/fstep)) #f0(minIndx):fstep:f0(maxIndx)
                tfrep['spec'] = normedS #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< final output
        else:
                raise Exception(f'Unknown time-frequency representation type: {typeName}')
        
        return tfrep


def preprocSound(audioWaveform, params):
        
        # TODO: implement 'wavelet' and 'lyons' tf representations if needed
        allowedTypes = ['stft'] 
        if params['tfType'] not in allowedTypes:
                raise Exception(f'Unknown time-frequency representation type: {params['tfType']}')

        stimStructs = {}
        maxPower = -1
        
          
        
        stim = {}        
        stim['tfrep'] = timefreq(audioWaveform, params['rawSampleRate'], params['tfType'], params['tfParams'])  
        stim['stimLength'] = stim['tfrep']['spec'].shape[1] / params['stimSampleRate'] #size(stim.tfrep.spec, 2)
        stim['sampleRate'] = params['stimSampleRate']

        stimStructs[f'waveform']=stim
        
        if params['numStimFeatures'] == -1: params['numStimFeatures'] = stim['tfrep']['spec'].shape[0] #size(stim.tfrep.spec, 1)
        if params['f'] == -1: params['f'] = stim['tfrep']['f']
        
        params['totalStimLength'] = params['totalStimLength'] + stim['tfrep']['spec'].shape[1] #size(stim.tfrep.spec, 2)

        maxPower = np.max([maxPower, np.max(np.max(stim['tfrep']['spec'], axis=0))])        
        
        ## normalize spectrograms and take log if requested
        if params['tfParams']['log']:
                refpow = maxPower if params['tfParams']['refpow'] == 0 else params['tfParams']['refpow']
                for key in stimStructs.keys():          
                        stim = stimStructs[key]
                        stim['tfrep']['spec'] = 20*np.log10(stim['tfrep']['spec']/refpow)+params['tfParams']['dbnoise']
                        stim['tfrep']['spec'][stim['tfrep']['spec']<0]=0
                        stimStructs[key] = stim

        ## concatenate stims into big matrix and record info into struct
        stimInfo = {}
        stimInfo['stimLengths'] = np.zeros([1, audioWaveform.shape[1]]) #zeros(1, length(audioWaveforms))
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
                tend = cindx + slen
                
                wholeStim[cindx:tend, :] = stim['tfrep']['spec'].transpose()
                groupIndex[:, cindx:tend] = idx
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
        tfParams['fband'] = 125
        tfParams['nstd'] = 6
        preprocStimParams['tfParams'] = tfParams
        
        ## use preprocSound to generate spectrogram
        stim_spec, groupIndex, stimInfo, preprocStimParams = preprocSound(stim, preprocStimParams)
        
        
        tInc = 1 / stimInfo['sampleRate'] # generate corresponding timeline for spectrogram
        
        t = np.linspace(0, (stim_spec.shape[0]-1)*tInc, stim_spec.shape[0])  # 0:tInc:(size(stim_spec, 1)-1)*tInc
        f=stimInfo['f']
        
        return stim_spec, t, f

def  STRFgen_V2(paramH, paramG, f, dt, maxdelay = 2500, nIn=1, outputNL='linear', freqDom=0):

        strf = {}
        strf['type'] = 'lin'
        strf['nIn'] = nIn
        strf['t'] = np.linspace(0, maxdelay*dt, int(maxdelay/dt+1) ) #0:dt:maxdelay*dt
        strf['delays'] = np.linspace(0, maxdelay, int(maxdelay/1)+1  ) #0:maxdelay
        strf['nWts'] = (nIn*len(strf['delays']) + 1)

        # strf.w1=zeros(nIn,length(delays))
        strf['b1']=0

        nlSet=['linear','logistic','softmax','exponential']
        
        if outputNL in nlSet: strf['outputNL'] = outputNL
        else: raise Exception('linInit >< Unknown Output Nonlinearity!')

        strf['freqDomain'] = freqDom

        strf['internal'] = {}
        strf['internal']['compFwd']=1
        strf['internal']['prevResp'] = []
        strf['internal']['prevLinResp'] = []
        strf['internal']['dataHash'] = np.nan

        
        paramH['phase'] = np.pi/2

        ## Generate STRFs with specified parameters
        # Create STRF of [f] frequecy channels, and time delays of 40 dts
        # Parameters from Amin et al., 2010, J Neurophysiol
        # Temporal parameters from Adelson and Bergen 1985, J Opt Soc Am A   
        t = strf['t'] # time delay

        strf['H'] = np.exp(-t/paramH['alpha'])*(paramH['SC1']*((t/paramH['alpha'])**(paramH.N1/math.factorial(paramH.N1))) - \
        paramH['SC2'] * ((t/paramH['alpha'])**paramH.N2/math.factorial(paramH.N2)))
        
        strf['G'] = np.exp(-0.5*((f-paramG.f0)/paramG.BW)**2)* np.cos(2*np.pi*paramG['BSM']*(f-paramG['f0']))
        strf['w1']=strf['G'].transpose()*strf['H']
        strf['f']=f
        
        return strf




def linFwd_Junzi(strf, stim, nSample):

        eps = sys.float_info.epsilon
        realmin = sys.float_info.min
        realmax = sys.float_info.max
        
        samplesize = nSample

        a = np.zeros((samplesize, 1))
        
        for ti in range(len(strf['delays'])):
                at = np.matmul(stim, strf['w1'][:,ti]) 
                
                thisshift = strf['delays'][ti]
                
                if thisshift>=0: a[thisshift:-1] = a[thisshift:-1] + at[:-1-thisshift]
                else: offset = thisshift%samplesize
                
                a[:offset] = a[:offset] + at[-thisshift:-1]

        a = a + strf.b1

        
        if strf['outputNL'] == 'linear':   # Linear outputs 
                resp_strf = a

        if strf['outputNL'] == 'logistic':   # Logistic outputs
                # Prevent overflow and underflow: use same bounds as glmerr
                # Ensure that log(1-y) is computable: need exp(a) > eps
                maxcut = -np.log(eps)
                # Ensure that log(y) is computable
                mincut = -np.log(1/realmin - 1)
                a = min(a, maxcut)
                a = max(a, mincut)
                resp_strf = 1/(1 + np.exp(-a))

        if strf['outputNL'] ==  'softmax':        # Softmax outputs
                nout = a.shape[1]
                # Prevent overflow and underflow: use same bounds as glmerr
                # Ensure that sum(exp(a), 2) does not overflow
                maxcut = np.log(realmax) - np.log(nout)
                # Ensure that exp(a) > 0
                mincut = np.log(realmin)
                a = min(a, maxcut)
                a = max(a, mincut)
                temp = np.exp(a)
                resp_strf = temp/(np.sum(temp, 1)*np.ones((1,nout)))
                # Ensure that log(y) is computable
                resp_strf[resp_strf<realmin] = realmin

        if strf['outputNL'] ==   'exponential':
                resp_strf=np.exp(a)
        
        else:
                raise Exception('Unknown activation function ', strf['outputNL'])
        

        # mask for nonvalid frames
        nanmask = strf['delays']%(stim.shape[0]+1)
        nanmask = nanmask[nanmask!=0] #no mask for delay 0
        a[nanmask] = np.nan
        resp_strf[nanmask] = np.nan
        
        return strf, resp_strf, a


def STRFconvolve_V2(strf, stim_spec, mean_rate):
        
        t = strf['t']
        ## convolve STRF and stim
        # Initialize strflab global variables with our stim and responses
        
        # TODO: see if this is an important step
        # strfData(stim_spec, zeros(size(stim_spec)))
        
        
        _, frate,_ = linFwd_Junzi(strf, stim_spec) #strfFwd_Junzi(strf)
        
        frate = frate*mean_rate
        frate = np.nan_to_num(frate)
        
        # offset rate
        offset_rate = -frate + max(frate) #-frate + max(frate)*0.6;
        firstneg = np.where(offset_rate <= 0)[0] #find(offset_rate <= 0,1,'first')

        if firstneg > 5500: firstneg = 2501 # for AM stimuli
        
        # offset rate
        offset_rate[:firstneg] = 0
        offset_rate[offset_rate < 0] = 0

        # onset rate
        onset_rate = frate
        onset_rate[onset_rate < 0] = 0

        return onset_rate, offset_rate


sigma = 24 #60 for bird but 38 for mouse
tuning = 'mouse' #'bird' or 'mouse'
stimGain = 0.5
targetlvl = 0.01
maskerlvl = 0.01 #default is 0.01
maxWeight = 1 #maximum mixed tuning weight capped at this level.

paramH = {}
paramH['alpha'] = 0.01 # time constant of temporal kernel [s] 0.0097
paramH['N1'] = 5
paramH['N2'] = 8
paramH['SC1'] = 1 #Nominally 1
paramH['SC2'] = 0.88  #increase -> more inhibition #0.88 in paper

strfGain = 0.1

# frequency parameters - static
paramG = {}
paramG['BW'] = 2000 # Hz
paramG['BSM'] = 5.00E-05 # 1/Hz=s best spectral modulation
paramG['f0'] = 4300 # ~strf.f(30)


data_path = r'resampled-stimuli'



masker_specs = {}
for trial in range(1, 2):
        fs, masker = wavfile.read(os.path.join(data_path,f'200k_masker{str(trial)}.wav'))
        masker = masker[np.newaxis, :]
        spec,_,_ = STRFspectrogram(masker/rms(masker)*maskerlvl,fs)
        masker_specs[str(trial)] = spec

### tested above portion so far, need to check fft in gaussian as well
s1_sampling_rate, s1_audio_data = wavfile.read(os.path.join(data_path, '200k_target1.wav'))
s2_sampling_rate, s2_audio_data = wavfile.read(os.path.join(data_path, '200k_target2.wav'))

song1_spec,t,f = STRFspectrogram(s1_audio_data/rms(s1_audio_data)*targetlvl,fs)
song2_spec,_,_ = STRFspectrogram(s2_audio_data/rms(s2_audio_data)*targetlvl,fs)

specs = {}
specs[f'songs_{1}'] = song1_spec
specs[f'songs_{2}'] = song2_spec
specs['maskers'] = masker_specs
specs['dims'] = song1_spec.shape
specs['t'] = t
specs['f'] = f


# make STRF
strf = STRFgen_V2(paramH,paramG,specs.f,specs.t(2)-specs.t(1))
strf.w1 = strf.w1*strfGain

# sum of STRF with gain should be ~43.2;
# adjust STRF gain for spiking

paramSpk = {}
paramSpk['t_ref'] = 1.5
paramSpk['t_ref_rel'] = 0.5
paramSpk['rec'] = 4

## Run simulation script
mean_rate = 0.1

tuningParam = {}
tuningParam['strf'] = strf
tuningParam['type'] = tuning
tuningParam['sigma'] = sigma


fr_target_on_1, fr_target_off_1 = STRFconvolve_V2(strf,specs[f'songs_{1}']*stimGain,mean_rate)
fr_target_on_2, fr_target_off_2 = STRFconvolve_V2(strf,specs[f'songs_{2}']*stimGain,mean_rate)

fr_masker = {}
for m in range(1,11):
        _, fr_masker[str(m)] = STRFconvolve_V2(strf,specs['maskers'][str(m)]*stimGain, mean_rate)



