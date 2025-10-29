import os
import numpy as np
import math
import sys
import yaml

from scipy.io import wavfile
from tqdm import tqdm, trange
from argparse import ArgumentParser


def rms(data):
        """
        Calculates the Root Mean Square (RMS) of a NumPy array.

        Args:
        data (np.ndarray): The input_ array.

        Returns:
        float or np.ndarray: The RMS value(s).
        """
        return np.sqrt(np.mean(data**2))


def get_f(fftLen, samprate, tfParams):
        select = np.linspace(1,int((fftLen+1)/2), int((fftLen+1)/2)) if fftLen%2!=0 else np.linspace(1,int(fftLen/2)+1, int(fftLen/2)+1)
        f0 = (select-1)*samprate/fftLen
        
        maxIndx = np.where(f0 >= tfParams['high_freq'])[0][0]
        minIndx = np.where(f0 < tfParams['low_freq'])[0][-1]+1
        
        fstep = f0[1]
        f = np.linspace(f0[minIndx], f0[maxIndx], 1 + int((f0[maxIndx]-f0[minIndx]+1)/fstep)) #f0(minIndx):fstep:f0(maxIndx)
                
        return f


class GenSTRF(object):
        
        def __init__(self, args, config, init_stim_path):
                self.args = args
                self.strf_config = config
                self.paramH = self.strf_config['paramH']
                self.paramG = self.strf_config['paramG']
                
                # generating strfs with a reference stimlus
                self.strf = self.process_initial_stimulus(init_stim_path, self.strf_config['targetlvl'], self.strf_config['strfGain'])

        def process_initial_stimulus(self, stim_path, lvl, strfGain):
                fs, data = wavfile.read(stim_path)
                data = data[np.newaxis, :]
                spec, t, f = self.STRFspectrogram(data/rms(data)*lvl,fs)
                # sum of STRF with gain should be ~43.2;
                # adjust STRF gain for spiking
                strf = self.STRFgen(self.paramH, self.paramG, f, t[1]-t[0])
                strf['w1'] = strf['w1']*strfGain
                return strf
        
        def process_stimulus(self, stim_path, lvl, stimGain):
                fs, data = wavfile.read(stim_path)
                data = data[np.newaxis, :]
                spec,_,_ = self.STRFspectrogram(data/rms(data)*lvl,fs)
                fr_on, fr_off = self.STRFconvolve(self.strf,spec*stimGain, self.strf_config['mean_rate'])
                return fr_on, fr_off


        def GaussianSpectrum(self, input_, increment, winLength, samprate, nstd = 6):


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
                
                for i in trange(frameCount, desc='Computing fft - ', ncols=100):
                        start = i*increment
                        last = start + winLength
                        
                        f = ws*pinput_[:, start:last]
                        pg[:,i] = np.std(f)
                        
                        specslice = np.fft.fft(f) #/fftLen ??
                        s[:,i] = specslice[:,:(int((fftLen+1)/2)+1)] if fftLen%2!=0 else specslice[:,:(int(fftLen/2)+1)]
                
                # Assign frequency_label
                select = np.linspace(1,int((fftLen+1)/2), int((fftLen+1)/2)) if fftLen%2!=0 else np.linspace(1,int(fftLen/2)+1, int(fftLen/2)+1)
                f0 = (select-1)*samprate/fftLen

                # assign time_label
                t0 = np.linspace(0, s.shape[1]-1, s.shape[1])*(increment/samprate)
                return s, t0, f0, pg




        def timefreq(self, audioWaveform, sampleRate, typeName, params):
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
                        s, t0, f0, pg = self.GaussianSpectrum(audioWaveform, increment, winLength, sampleRate, tfrep['params']['nstd']) 
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


        def preprocSound(self, audioWaveform, params):
                
                # TODO: implement 'wavelet' and 'lyons' tf representations if needed
                allowedTypes = ['stft'] 
                if params['tfType'] not in allowedTypes:
                        raise Exception(f'Unknown time-frequency representation type: {params['tfType']}')

                stimStructs = {}
                maxPower = -1
                
                
                
                stim = {}        
                stim['tfrep'] = self.timefreq(audioWaveform, params['rawSampleRate'], params['tfType'], params['tfParams'])  
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

        def STRFspectrogram(self, stim, fs):
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
                stim_spec, groupIndex, stimInfo, preprocStimParams = self.preprocSound(stim, preprocStimParams)
                
                
                tInc = 1 / stimInfo['sampleRate'] # generate corresponding timeline for spectrogram
                
                t = np.linspace(0, (stim_spec.shape[0]-1)*tInc, 1 + int(((stim_spec.shape[0]-1)*tInc)/tInc))  # 0:tInc:(size(stim_spec, 1)-1)*tInc
                f=stimInfo['f']
                
                return stim_spec, t, f

        def  STRFgen(self, paramH, paramG, f, dt, maxdelay = 2500, nIn=1, outputNL='linear', freqDom=0):

                strf = {}
                strf['type'] = 'lin'
                strf['nIn'] = nIn
                strf['t'] = np.linspace(0, maxdelay*dt, int((maxdelay*dt)/dt)) #0:dt:maxdelay*dt
                strf['delays'] = np.linspace(0, maxdelay, maxdelay) #0:maxdelay
                strf['nWts'] = (nIn*len(strf['delays']) + 1)

                # strf.w1=zeros(nIn,length(delays))
                strf['b1']=0

                nlSet=['linear', 'logistic', 'softmax', 'exponential']
                
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

                strf['H'] = np.exp(-t/paramH['alpha'])*(paramH['SC1']*(t/paramH['alpha'])**paramH['N1']/math.factorial(paramH['N1']) - \
                paramH['SC2'] * (t/paramH['alpha'])**paramH['N2']/math.factorial(paramH['N2']))
                
                strf['G'] = np.exp(-0.5*((f-paramG['f0'])/paramG['BW'])**2)* np.cos(2*np.pi*paramG['BSM']*(f-paramG['f0']))
                strf['w1']=strf['G'][:, np.newaxis]*strf['H'][np.newaxis, :]
                strf['f']=f
                
                return strf




        def linFwd_Junzi(self, strf, stim):


                samplesize = stim.shape[0]

                a = np.zeros((samplesize, 1))
                
                for ti in range(len(strf['delays'])):
                        at = np.matmul(stim, strf['w1'][:,ti][:, np.newaxis]) 
                        
                        thisshift = int(strf['delays'][ti])
                        
                        if thisshift>=0: 
                                a[thisshift:] = a[thisshift:] + at[:len(at)-thisshift]
                        else: 
                                offset = thisshift%samplesize
                                a[:offset] = a[:offset] + at[-thisshift:]

                a = a + strf['b1']

                
                if strf['outputNL'] == 'linear':   # Linear outputs 
                        resp_strf = a

                # eps = sys.float_info.epsilon
                # realmin = sys.float_info.min
                # realmax = sys.float_info.max
                
                # if strf['outputNL'] == 'logistic':   # Logistic outputs
                #         # Prevent overflow and underflow: use same bounds as glmerr
                #         # Ensure that log(1-y) is computable: need exp(a) > eps
                #         maxcut = -np.log(eps)
                #         # Ensure that log(y) is computable
                #         mincut = -np.log(1/realmin - 1)
                #         a = np.min(a, maxcut)
                #         a = np.max(a, mincut)
                #         resp_strf = 1/(1 + np.exp(-a))

                # if strf['outputNL'] ==  'softmax':        # Softmax outputs
                #         nout = a.shape[1]
                #         # Prevent overflow and underflow: use same bounds as glmerr
                #         # Ensure that sum(exp(a), 2) does not overflow
                #         maxcut = np.log(realmax) - np.log(nout)
                #         # Ensure that exp(a) > 0
                #         mincut = np.log(realmin)
                #         a = min(a, maxcut)
                #         a = max(a, mincut)
                #         temp = np.exp(a)
                #         resp_strf = temp/(np.sum(temp, 1)*np.ones((1,nout)))
                #         # Ensure that log(y) is computable
                #         resp_strf[resp_strf<realmin] = realmin

                # if strf['outputNL'] ==   'exponential':
                #         resp_strf=np.exp(a)
                
                else:
                        raise Exception('Unknown activation function ', strf['outputNL'])
                

                # mask for nonvalid frames
                nanmask = strf['delays']%(stim.shape[0]+1)
                nanmask = nanmask[nanmask!=0].astype(np.int64) #no mask for delay 0
                a[nanmask] = 0
                resp_strf[nanmask] = 0
                
                return strf, resp_strf, a


        def STRFconvolve(self, strf, stim_spec, mean_rate):
                
                t = strf['t']      
                _, frate,_ = self.linFwd_Junzi(strf, stim_spec) #strfFwd_Junzi(strf)
                frate = frate*mean_rate
                
                # offset rate
                offset_rate = -frate + np.max(frate) #-frate + max(frate)*0.6;
                firstneg = np.where(offset_rate <= 0)[0][0] #find(offset_rate <= 0,1,'first')

                if firstneg > 5500: firstneg = 2501 # for AM stimuli
                
                # offset rate
                offset_rate[:firstneg] = 0
                offset_rate[offset_rate < 0] = 0

                # onset rate
                onset_rate = frate
                onset_rate[onset_rate < 0] = 0

                return onset_rate, offset_rate


if __name__ == "__main__":
        
        args = ArgumentParser()
        args.add_argument('--target_dir', type=str, default='resampled-stimuli/target', help='directory containing target stimuli')
        args.add_argument('--masker_dir', type=str, default='resampled-stimuli/masker', help='directory containing masker stimuli')
        parsed_args = args.parse_args()
        
        yaml_path = 'config/config.yaml'
        config = yaml.safe_load(open(yaml_path, 'r'))
        strf_config = config['strf_config']

        lst_target_stim = [os.path.join(parsed_args.target_dir, stim_path) for stim_path in os.listdir(parsed_args.target_dir)]
        lst_masker_stim = [os.path.join(parsed_args.masker_dir, stim_path) for stim_path in os.listdir(parsed_args.masker_dir)]
        
        gen_strfs = GenSTRF(parsed_args, strf_config, lst_target_stim[0])
        
        
        target_dict = {}
        for stim_path in lst_target_stim:
                
                fr_on, fr_off = gen_strfs.process_stimulus(stim_path, strf_config['targetlvl'], strf_config['stimGain'])

                stim_name = os.path.split(stim_path)[-1].split('.')[0]
                target_dict[stim_name] = {}
                target_dict[stim_name]['fr_on'] = fr_on
                target_dict[stim_name]['fr_off'] = fr_off
        
        masker_dict = {}
        for stim_path in lst_masker_stim:
                fr_on, fr_off = gen_strfs.process_stimulus(stim_path, strf_config['maskerlvl'], strf_config['stimGain'])

                stim_name = os.path.split(stim_path)[-1].split('.')[0]
                masker_dict[stim_name] = {}
                masker_dict[stim_name]['fr_on'] = fr_on
                masker_dict[stim_name]['fr_off'] = fr_off
        


