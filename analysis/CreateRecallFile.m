% this script makes a blank recall .mat file of the correct length
%  (by reading in the recall file) before we score it.
close all;clear;subs={'1','2','3','4','5','6','7','8','9','10',...
        '11','12','13','14','15','16','17','18','19','20'};
[base,root]=load_root();gs=0;tngs=9;recl=zeros(length(subs),tngs);
ph4=10:12;ph5=13:15;ph6=16:18;%recall event #s from output
for i=1:length(subs) %load data
    sub=char(subs(i));dr=[root 'data/nonfmri/' sub '/'];gom2fn=[dr 'g_o_m2.mat'];
    load(gom2fn);cd(dr);j=1;
    for ii=ph4(1):ph6(end) %read audio files
        fn=[root 'data/recall/' sub '_game-' num2str(g_o(j)) '_gtrial-' num2str(ii) '.wav'];
        [y,Fs] = audioread(fn);recl(i,j)=length(y)/Fs;
        fn=[root 'data/blank/' sub '_game-' num2str(g_o(j)) '_gtrial-' num2str(ii) '.mat'];
        recall=zeros(floor(recl(i,j)),5);%create 'recall' variable
        save(fn,'recall');j=j+1;% save data
    end
end
cd([root 'analysis/']);