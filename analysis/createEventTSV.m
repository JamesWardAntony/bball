%% create event TSV files
clear;close all;clc;[~,root]=load_root();
subs={'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'};
nS=length(subs);tasks={'view','recall'};runs=3;gpr=3;n_trunc=3;
onsets=[];onsetsR=[];subm=[];g_os=[];%games/run
for s=1:nS
    %load mstr
    sn=char(subs(s));if s<10;st=['sub-0' sn];else;st=['sub-' sn];end
    fn=[root 'data/nonfmri/' sn '/MSMEhr2.mat'];load(fn,'MSTR','METR','RSTR','RETR');
    onset=MSTR(:,1)+n_trunc;dur=METR(:,1)-MSTR(:,1);onsetR=RSTR(:,1)+n_trunc;durR=RETR(:,1)-RSTR(:,1);
    fn=[root 'data/nonfmri/' sn '/g_o.mat'];load(fn,'g_o');%load game order
    %load event directory / file
    %ed=[root 'data/bids/Norman/Antony/ss/' st '/ses-01/func/'];%bids dir
    ed=[root 'data/extra/bball/' st '/ses-01/func/'];%OpenNeuro dir
    %note, subjects 4/5/6 had changes in their scans that caused odd/early
    %initial timings, but this is accounted for 
    subm=[subm;repmat(s,length(onset),1)];g_os=[g_os;g_o'];
    onsets=[onsets;onset];onsetsR=[onsetsR;onsetR];
    for t=1:length(tasks)
        tn=char(tasks(t));
        for r=1:runs
            ind=(r-1)*3+1:(r-1)*3+3;
%             efn=[ed st '_ses-01_task-' tn '_run-0' num2str(r) '_events.tsv'];
%             fid=fopen(efn,'w');fprintf(fid,'onset\tduration\ttrial_type\n');
%             for gn=1:gpr
%                 switch t;case 1;fprintf(fid,'%d\t%d\t%d\n',onset(ind(gn)),dur(ind(gn)),g_o(ind(gn)));
%                     case 2;fprintf(fid,'%d\t%d\t%d\n',onsetR(ind(gn)),durR(ind(gn)),g_o(ind(gn)));end
%             end
%             fclose(fid);
        end
    end
    %exceptions for subjects 4/5 - done manually on 7.6.20
end