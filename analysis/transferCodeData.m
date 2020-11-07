%% transfer code and data to new directory
%% analysis folder
clear;close all;[base,root]=load_root();
base2='/Volumes/norman/jantony/shared/Antony2020/';
if ~exist(base2,'dir');mkdir(base2);end
%% experiment files!! 
% fn='experiment.zip';sn=[root fn];dn=[base2 fn];
% if ~exist(dn,'file');copyfile(sn,dn);end
%% transfer non-fmri scripts
%not in paper but possibly of interest:
%'decorrssfs','importVols','loadHistMargins','speech2text_stuff,'XSubAll'
%.m files
emd=[base2 'analysis/'];if ~exist(emd,'dir');mkdir(emd);end
md=[root 'analysis/'];%main directory
mf={'CamAng_Speech_Courtpos','cell2str','corrEZ','CreateRecallFile','fdr_bh',...
    'load_root','loadf0','loadGameFile','loadLogAndTR','loadValueTest',...
    'meancenter','NCAA','pimc','plotdefs','plotdefs2','ppos',...
    'ppos2','psc','pupilBetaTimeCoursePlot','scale01','smoothn','violin',...
    'vplot','XSubBehav','XSubHRET','XSubRecall','XSubSeg'};
for i=1:length(mf);fn=[char(mf(i)) '.m'];sn=[md fn];dn=[emd fn];copyfile(sn,dn);end
%all .mat files
a=dir('*.mat');for i=1:length(a);fn=[a(i).name];sn=[md fn];dn=[emd fn];copyfile(sn,dn);end
%all .sh files
a=dir('*.sh');for i=1:length(a);fn=[a(i).name];sn=[md fn];dn=[emd fn];copyfile(sn,dn);end
%all .csv files
a=dir('*.csv');for i=1:length(a);fn=[a(i).name];sn=[md fn];dn=[emd fn];copyfile(sn,dn);end
%from my custom directory / others ...
mf={'astats','astats2','plotSpread'};
md=[base 'plugins/jwacustom/'];
for i=1:length(mf);fn=[char(mf(i)) '.m'];sn=[md fn];dn=[emd fn];copyfile(sn,dn);end
%% transfer directories / non-fmri data
emd=[base2 'pics/'];if ~exist(emd,'dir');mkdir(emd);end
fn='pics.zip';sn=[root fn];dn=[base2 fn];
if ~exist(dn,'file');copyfile(sn,dn);end
emd=[base2 'RStuff/'];if ~exist(emd,'dir');mkdir(emd);end;md=[root 'RStuff/'];
mf={'LinMemRS','PAMemA','Event_HMM_XCond-ROI-Poss-Ind-R'};
for i=1:length(mf);fn=[char(mf(i)) '.csv'];sn=[md fn];dn=[emd fn];copyfile(sn,dn);end
mf={'ssR'};for i=1:length(mf);fn=[char(mf(i)) '.R'];sn=[md fn];dn=[emd fn];copyfile(sn,dn);end
mf={'RStuff'};for i=1:length(mf);fn=[char(mf(i)) '.Rproj'];sn=[md fn];dn=[emd fn];copyfile(sn,dn);end

emd=[base2 'exp/'];if ~exist(emd,'dir');mkdir(emd);end;md=[root 'exp/'];cd(md);
%all .m files
a=dir('*.m');for i=1:length(a);fn=[a(i).name];sn=[md fn];dn=[emd fn];copyfile(sn,dn);end
a=dir('*.mat');for i=1:length(a);fn=[a(i).name];sn=[md fn];dn=[emd fn];copyfile(sn,dn);end
a=dir('*.zip');for i=1:length(a);fn=[a(i).name];sn=[md fn];dn=[emd fn];copyfile(sn,dn);end

%all recall .mats
emd=[base2 'data/'];if ~exist(emd,'dir');mkdir(emd);end
emd=[base2 'data/recall/'];if ~exist(emd,'dir');mkdir(emd);end;md=[root 'data/recall/'];cd(md);
a=dir('*.mat');for i=1:length(a);fn=[a(i).name];sn=[md fn];dn=[emd fn];copyfile(sn,dn);end
%% nonfmri (behavioral) folders
emd=[base2 'data/nonfmri/'];if ~exist(emd,'dir');mkdir(emd);end;md=[root 'data/nonfmri/'];cd(md);
subs={'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'};
for id=1:length(subs)
    fill=['data/nonfmri/' char(subs(id)) '/'];emd=[base2 fill];if ~exist(emd,'dir');mkdir(emd);end
    md=[root fill];cd(md);
    %eyetracking
    a=dir('*.asc');for i=1:length(a);fn=[a(i).name];sn=[md fn];dn=[emd fn];copyfile(sn,dn);end
    %heart rate
    a=dir('*PULS.log');for i=1:length(a);fn=[a(i).name];sn=[md fn];dn=[emd fn];copyfile(sn,dn);end
    %behavior files
    for ii=1:8;fn=[char(subs(id)) '_phase' num2str(ii) '.mat'];sn=[md fn];dn=[emd fn];copyfile(sn,dn);end
    a=dir('confounds_hrf2_5*');for i=1:length(a);fn=[a(i).name];sn=[md fn];dn=[emd fn];copyfile(sn,dn);end
    a=dir('g*');for i=1:length(a);fn=[a(i).name];sn=[md fn];dn=[emd fn];copyfile(sn,dn);end
    a=dir('MSMEhr2*');for i=1:length(a);fn=[a(i).name];sn=[md fn];dn=[emd fn];copyfile(sn,dn);end
    a=dir('phys_m*');for i=1:length(a);fn=[a(i).name];sn=[md fn];dn=[emd fn];copyfile(sn,dn);end
    a=dir('ssr*');for i=1:length(a);fn=[a(i).name];sn=[md fn];dn=[emd fn];copyfile(sn,dn);end
end    
subs={'101','102','103','104','105','106','107','108','109','110','111','112','113','114','115'};
for id=1:length(subs)
    fill=['data/nonfmri/' char(subs(id)) '/'];emd=[base2 fill];if ~exist(emd,'dir');mkdir(emd);end
    md=[root fill];cd(md);
    for ii=1:6:7;fn=[char(subs(id)) '_phase' num2str(ii) '.mat'];sn=[md fn];dn=[emd fn];copyfile(sn,dn);end
    a=dir('g*');for i=1:length(a);fn=[a(i).name];sn=[md fn];dn=[emd fn];copyfile(sn,dn);end
end  
subs={'1009','1010','1011'};
for id=1:length(subs)
    fill=['data/nonfmri/' char(subs(id)) '/'];emd=[base2 fill];if ~exist(emd,'dir');mkdir(emd);end
    md=[root fill];cd(md);
    for ii=1:1;fn=[char(subs(id)) '_phase' num2str(ii) '.mat'];sn=[md fn];dn=[emd fn];copyfile(sn,dn);end
    a=dir('g*');for i=1:length(a);fn=[a(i).name];sn=[md fn];dn=[emd fn];copyfile(sn,dn);end
end  
%% bids fmri data
emd=[base2 'data/bids/'];if ~exist(emd,'dir');mkdir(emd);end
emd=[base2 'data/bids/Norman/'];if ~exist(emd,'dir');mkdir(emd);end
emd=[base2 'data/bids/Norman/Antony/'];if ~exist(emd,'dir');mkdir(emd);end
emd=[base2 'data/bids/Norman/Antony/ss/'];if ~exist(emd,'dir');mkdir(emd);end
md=[root 'data/bids/Norman/Antony/ss/'];cd(md);
%outer 'ss' folder
a=dir('*');for i=1:length(a);if a(i).isdir==0;fn=[a(i).name];
        sn=[md fn];dn=[emd fn];copyfile(sn,dn);end;end
%rois
emd=[base2 'data/bids/Norman/Antony/ss/rois/'];if ~exist(emd,'dir');mkdir(emd);end
md=[root 'data/bids/Norman/Antony/ss/rois/'];cd(md);
a=dir('*');for i=1:length(a);if a(i).isdir==0;fn=[a(i).name];
        sn=[md fn];dn=[emd fn];copyfile(sn,dn);end;end
%code
emd=[base2 'data/bids/Norman/Antony/ss/code/'];if ~exist(emd,'dir');mkdir(emd);end
md=[root 'data/bids/Norman/Antony/ss/code/'];cd(md);
a=dir('*.sh');for i=1:length(a);fn=[a(i).name];sn=[md fn];dn=[emd fn];copyfile(sn,dn);end
a=dir('*.ipynb');for i=1:length(a);fn=[a(i).name];sn=[md fn];dn=[emd fn];copyfile(sn,dn);end
a=dir('*.py');for i=1:length(a);fn=[a(i).name];sn=[md fn];dn=[emd fn];copyfile(sn,dn);end
a=dir('*.csv');for i=1:length(a);fn=[a(i).name];sn=[md fn];dn=[emd fn];copyfile(sn,dn);end
%derivatives
%fmriprep
emd=[base2 'data/bids/Norman/Antony/ss/derivatives/'];if ~exist(emd,'dir');mkdir(emd);end
emd=[base2 'data/bids/Norman/Antony/ss/derivatives/fmriprep/'];if ~exist(emd,'dir');mkdir(emd);end
md=[root 'data/bids/Norman/Antony/ss/derivatives/fmriprep/'];cd(md);
a=dir('dataset*');for i=1:length(a);fn=[a(i).name];sn=[md fn];dn=[emd fn];copyfile(sn,dn);end
subs={'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'};
for id=1:length(subs)
    if id<10;fill=['sub-0' char(subs(id)) '/'];else;fill=['sub-' char(subs(id)) '/'];end
    emd2=[emd fill];if ~exist(emd2,'dir');mkdir(emd2);end
    %anat - brain mask only
    sesn='anat/';emd2=[emd fill sesn];if ~exist(emd2,'dir');mkdir(emd2);end
    md2=[md fill sesn];cd(md2);
    fn=[fill(1:end-1) '_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'];
    sn=[md2 fn];dn=[emd2 fn];copyfile(sn,dn);
    %ses-01 - copy anat and select files from func
    sesn='ses-01/';emd2=[emd fill sesn];if ~exist(emd2,'dir');mkdir(emd2);end
    sesn='ses-01/anat/';emd2=[emd fill sesn];if ~exist(emd2,'dir');mkdir(emd2);end
    md2=[md fill sesn];cd(md2);copyfile(md2,emd2);
    sesn='ses-01/func/';emd2=[emd fill sesn];if ~exist(emd2,'dir');mkdir(emd2);end
    md2=[md fill sesn];cd(md2);
    a=dir('*desc-confounds_regressors.tsv');
    for i=1:length(a);fn=[a(i).name];sn=[md2 fn];dn=[emd2 fn];copyfile(sn,dn);end
    a=dir('*_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz');
    for i=1:length(a);fn=[a(i).name];sn=[md2 fn];dn=[emd2 fn];copyfile(sn,dn);end
end

%secondlevel
emd=[base2 'data/bids/Norman/Antony/ss/derivatives/secondlevel/'];if ~exist(emd,'dir');mkdir(emd);end
md=[root 'data/bids/Norman/Antony/ss/derivatives/secondlevel/'];cd(md);
a=dir('avg_brain*');for i=1:length(a);fn=[a(i).name];sn=[md fn];dn=[emd fn];copyfile(sn,dn);end
emd=[base2 'data/bids/Norman/Antony/ss/derivatives/secondlevel/HMM/'];if ~exist(emd,'dir');mkdir(emd);end
md=[root 'data/bids/Norman/Antony/ss/derivatives/secondlevel/HMM/'];cd(md);
a=dir('*-140.mat');for i=1:length(a);fn=[a(i).name];sn=[md fn];dn=[emd fn];copyfile(sn,dn);end
    
%firstlevel
emd=[base2 'data/bids/Norman/Antony/ss/derivatives/firstlevel/'];if ~exist(emd,'dir');mkdir(emd);end
md=[root 'data/bids/Norman/Antony/ss/derivatives/firstlevel/'];cd(md);
for id=1:length(subs)
    if id<10;fill=['sub-0' char(subs(id)) '/'];else;fill=['sub-' char(subs(id)) '/'];end
    emd2=[emd fill];if ~exist(emd2,'dir');mkdir(emd2);end
    md2=[md fill];cd(md2);
    %% main folder
    %kill extra niftis
    a=dir('*trim3.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*trim3_4.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*trim3_5.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_128.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_180.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_240.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_360.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_480.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_720.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*T1w_desc-preproc_bold_trim3_norm_event.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*T1w_desc-preproc_bold_trim3_norm_glm.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end    
    a=dir('*_128_norm_event.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_128_norm_glm.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_240_norm_event.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_240_norm_glm.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_360_norm_event.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_360_norm_glm.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_480_norm_event.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_480_norm_glm.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_720_norm_event.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_720_norm_glm.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    %kill extra ROIs
    a=dir('*_ac.mat');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_ACC.mat');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_amyg.mat');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_bilateral_frontal_inf-orbital.mat');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_bilateral_HC.mat');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_bilateral_NAcc.mat');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_bilateral_oc-temp.mat');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_PCC.mat');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_RPE_Cb.mat');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_vc.mat');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    %grab 
    %glm outputs
    fn=['glm_out-3-5-140-1-1.mat'];sn=[md2 fn];dn=[emd2 fn];copyfile(sn,dn);
    fn2=['glm_out-3-5-140-1-2.mat'];sn=[md2 fn2];dn=[emd2 fn2];copyfile(sn,dn);
    a=dir('glm_out*');
    for i=1:length(a);fn3=[a(i).name];
        if ~strcmpi(fn,fn3);if ~strcmpi(fn2,fn3);delete(fn3);end;end
    end
    %all other files
    fi=[fill(1:end-1) '_task-view*'];
    a=dir(fi);
    for i=1:length(a);fn=[a(i).name];sn=[md2 fn];dn=[emd2 fn];copyfile(sn,dn);end
    
    %% ses-01 folder
    sesn='ses-01/';emd2=[emd fill sesn];if ~exist(emd2,'dir');mkdir(emd2);end
    md2=[md fill sesn];cd(md2);
    %kill extra niis
    a=dir('*_trim6_norm.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_trim6TRs.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_trim0TRs.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    %grab confounds and smooth5 nii
    fi=[fill(1:end-1) '_ses-01_task-view*confounds_selected.txt'];
    a=dir(fi');
    for i=1:length(a);fn=[a(i).name];sn=[md2 fn];dn=[emd2 fn];copyfile(sn,dn);end
    fi=[fill(1:end-1) '_ses-01_task-view*smooth5.nii.gz'];
    a=dir(fi);
    for i=1:length(a);fn=[a(i).name];sn=[md2 fn];dn=[emd2 fn];copyfile(sn,dn);end
    
    %% masks folder
    sesn='masks/';emd2=[emd fill sesn];if ~exist(emd2,'dir');mkdir(emd2);end
    md2=[md fill sesn];cd(md2);
    
    a=dir('*_ac.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_ACC.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_amyg.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*frontal_inf-orbital.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_bilateral_HC.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_bilateral_NAcc.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*pole.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*temp_lat-fusifor.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*temp_lat.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*and_Lingual.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*-Parahip.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*temp.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*DMN2.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*HC2.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*STG.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*left_ang.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*left_cun.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*right_ang.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*right_cun.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*lNAcc.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*rNAcc.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*VMPFC.gz.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*PMC2.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_sup_gt.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_sup_l.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_sup_pp.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_sup_pt.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*V1_V2.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*V2.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*V1z.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_RPE_Cb.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_VMPFC2.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    a=dir('*_vc.nii.gz');for i=1:length(a);fn3=[a(i).name];delete(fn3);end
    %transfer the remaining
    fi=[fill(1:end-1) '_*'];
    a=dir(fi');
    for i=1:length(a);fn=[a(i).name];sn=[md2 fn];dn=[emd2 fn];copyfile(sn,dn);end
end

    
    


