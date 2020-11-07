recObj = audiorecorder;
Fs = 44100 ; 
nBits = 16 ; 
nChannels = 2 ; 
ID = -1; % default audio input device 
recObj = audiorecorder(Fs,nBits,nChannels,ID);
disp('Start speaking.')
recordblocking(recObj,5);
disp('End of Recording.');
play(recObj);
%%  for continuous recordings.
fs = 11025;
recordData = audiorecorder(fs, 16, 1);
record(recordData);
stop(recordData);
voiceResponse{1}=getaudiodata(recordData);
play(recordData);
figure; plot(voiceResponse{1})

[y,Fs] = audioread('4_8cut.mov');
figure; plot(y)
