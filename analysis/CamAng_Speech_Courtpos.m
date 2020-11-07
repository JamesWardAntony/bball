%% calculates the number of camera angles, queries speech on/off, or left/right half of court
% to do this, I created fake subject, 1009-11, and I watched the videos,
% clicking the mouse each time there was a camera change / any speech / lr
% change in the court position. This script just reads in that file as if I
% were a subject and computes the clicks. Preference/enjoyment ratings are
% meaningless. It only needs to be run once, and it's already been run, but
% it's here in case it's useful. Didn't end up using 'speech' - went with
% the prosody measure instead.
close all;clear;[base,root]=load_root();addpath(genpath([root 'analysis/']));
CA_S_LR=3;%1=camera angles,2=speech,3=lr
if CA_S_LR==1;subs={'1009'};elseif CA_S_LR==2;subs={'1010'};
elseif CA_S_LR==3;subs={'1011'};end
gs=0;tngs=9;nS=length(subs);load g;load d_event_mat1;
ph1=1:9;g_o_m2=cell(ph1(end),1);clicks_a=zeros(1,tngs);
for i=1:nS %load data
    sub=char(subs(i));
    clear g_o_m2 g_o gb_o gs_o gf pet bt spt;
    dr=[root 'data/nonfmri/' sub '/'];cd(dr);
    load([sub '_phase1.mat'],'g_o_m','g_o','gb_o','gs_o','gf');g_o_m2(ph1)=g_o_m(ph1);
    % pref/enjoyment
    for ii=ph1(1):ph1(end)
        g_id=char(gf{g_o(ii)}.gf);
        %segmentation
        c=g_o_m2{ii}.clickTimes;clicks2=[];
        if ~isempty(c)
            if CA_S_LR==1;clicks2=c(diff(c)>0.5);c1=[c(1) clicks2];
                clicks{g_o(ii)}.clicks=c1;l=length(find(clicks_a(:,g_o(ii))>0));
                clicks_a(l+1:l+length(c1),g_o(ii))=c1;
                n_camang(i,g_o(ii))=length(c1);
            elseif CA_S_LR==2;clicks2=c(diff(c)>0.05);c1=[c(1) clicks2];
                speech{g_o(ii)}.speech=round(c1);speech{g_o(ii)}.gf=g_id;
            elseif CA_S_LR==3;clicks2=c(diff(c)>0.05);c1=[c(1) clicks2];c1=c1';
                for iii=1:size(c1,1);el=c1(iii,1);marker=floor(el);
                    f=find(abs(el-c1)<0.4);
                    if numel(f)>1;c1(iii,2)=2;else;c1(iii,2)=1;end
                end
                lrpos=zeros(ceil(g_o_m2{ii}.movieLength),1);jj=0;prevlr=0;
                for iii=1:ceil(g_o_m2{ii}.movieLength)
                    f=find(and(c1(:,1)-(iii-1)>0,c1(:,1)-iii<0));
                    if numel(f)>0;lrpos(iii)=c1(f(end),2);prevlr=c1(f(end),2);
                        if jj==0;firstlr=prevlr;jj=1;end
                    else;lrpos(iii)=prevlr;
                    end
                end
                f=find(lrpos==0);
                if numel(f)>0;lrpos(f)=firstlr;end %fix if nothing @ beginning
                courtpos{g_o(ii)}.courtpos=lrpos;courtpos{g_o(ii)}.gf=g_id;
            end
        end
    end
end
cd([root 'analysis/']);
%save outputs
if CA_S_LR==1;clearvars -except n_camang;load susgamebinmean;save susgamebinmean;
elseif CA_S_LR==2;clearvars -except speech;save speech;
elseif CA_S_LR==3;clearvars -except courtpos;save courtpos;
end