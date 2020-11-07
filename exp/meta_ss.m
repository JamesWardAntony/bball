%% meta-control script
addpath('/Volumes/norman/jantony/surprisesuspense/analysis');clear;[~,root]=load_root();
tngs=9;tnbgs=5;ngs=3;ngps=ngs;tnsgs=28;ngs=3;inp=2;et1=1;et2=0;sub='997';suspy=1;cn=[root 'subs/' sub '/'];
if ~exist(cn,'file');mkdir(cn);end;sn=[cn 'g_o.mat'];
if ~exist(sn,'file');create_g_o;else;load(sn,'g_o','gb_o','gs_o','ml','gf');end
%% test sound!!
%% in-scanner phases
phase=1;et=et1;fmri=1;inpo=inp;init_ss;%run first phase
%% ======
phase=4;et=et2;fmri=1;inpo=inp;init_ss;
%% ======
phase=2;et=et1;fmri=1;inpo=inp;init_ss;
%% 
phase=5;et=et2;fmri=1;inpo=inp;init_ss;
%% ====
phase=3;et=et1;fmri=1;inpo=inp;init_ss;
%% ====
phase=6;et=et2;fmri=1;inpo=inp;init_ss;
%% post-scanner phases==========
phase=7;et=et2;fmri=0;inpo=inp;init_ss;
phase=8;et=et2;fmri=0;inpo=inp;init_ss;