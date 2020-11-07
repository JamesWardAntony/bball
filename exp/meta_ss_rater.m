%% meta-control script
%addpath('/Volumes/norman/jantony/surprisesuspense/analysis');
clear;[~,root]=load_root();
tngs=9;tnbgs=5;ngs=3;ngps=tngs;tnsgs=28;inp=1;et1=0;et2=0;rater=1;
sub='1015';suspy=1;cn=[root 'subs/' sub '/'];
if ~exist(cn,'file');mkdir(cn);end;sn=[cn 'g_o.mat'];
if ~exist(sn,'file');create_g_o;else;load(sn,'g_o','gb_o','gs_o','ml','gf');end
%% 
%addpath(genpath('/Applications/Psychtoolbox'))
phase=1;et=et1;fmri=0;inpo=inp;init_ss;%run first phase
%%
%phase=7;et=et2;fmri=0;inpo=inp;init_ss;