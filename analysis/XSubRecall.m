%% across-subject recall analysis-Fig 6A-B, analysis set up for 6C, Fig S5
close all;clear;plotdefs2;[base,root]=load_root();addpath(genpath([root 'analysis/']));
subs={'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'};
gs=0;tngs=9;fact4y=1;load questionnaire;nS=length(subs);
switch fact4y;case 0;load g0;load d_event_mat0;load gamebinmeans0;
    case 1;load g1;load d_event_mat1;load gamebinmeans1;end
load memsurmed;%median surprise of recalled posses for each sub - ran previously to work
nPoss=length(p_fl);origpflc=size(p_fl,2);%original p_fl columns
game_poss=zeros(tngs,1);for i=1:tngs;game_poss(i)=length(find(p_fl(:,1)==i));end % # game possessions
rec_all_poss_xsub=zeros(length(subs),nPoss);%matrix for each sub, each poss
rec_all_poss_order=rec_all_poss_xsub;
sur_order=rec_all_poss_order;mcols=[4 11 5:7];%mcols=[4 8 5 7];
precise_recall=zeros(length(subs),1);false_recall=precise_recall;
recl=zeros(length(subs),tngs);
ph4=10:12;ph5=13:15;ph6=16:18;%recall phase #s
lag=20;lags_x=linspace(-lag,lag,lag*2+1);%lags for recall data
poss_length=-diff(p_fl(:,4));poss_length(poss_length<0)=p_fl(poss_length<0,4);
poss_length(length(poss_length)+1)=p_fl(end,4);p_fl=[p_fl poss_length];
%presets
Sub_num=[];Mem=[];PrevMem=[];Sub_num2=[];Mem2=[];Sub_numG=[];
MemG=[];FalseMemG=[];SurG=[];Block=[];PossInGame=[];Expert=[];
Engage=[];BeliefPerf=[];PrefAny=[];PrefWin=[];Enjoy=[];recall_a=[];
prevgame=0;prevposs=0;jjj=0;
for i=1:nS %load data
    sub=char(subs(i));dr=[root 'data/nonfmri/' sub '/'];
    gom2fn=[dr 'g_o_m2.mat'];load(gom2fn);cd(dr);%load
    lags=zeros(length(lags_x),1);lags(lag+1)=nan;vert=[];pref_any2=[];%preset
    pref_win2=[];enjoy_2=[];j=1;kk=0;ll=0;
    lags_ls=lags;lags_ls(lag+1)=nan;lags_hs=lags;lags_hs(lag+1)=nan;numpossorder=zeros(tngs,0);
    lags_nr_ls=lags;lags_nr_ls(lag+1)=nan;lags_nr_hs=lags;lags_nr_hs(lag+1)=nan;poss_e=0;
    for ii=ph4(1):ph6(end) %read audio files
        fn=[root 'data/recall/' sub '_game-' num2str(g_o(j)) '_gtrial-' num2str(ii) '.wav'];
        [y,Fs] = audioread(fn);recl(i,j)=length(y)/Fs;%load audio file
        fn=[root 'data/recall/' sub '_game-' num2str(g_o(j)) '_gtrial-' num2str(ii) '.mat'];
        clear recall;load(fn);%load recall data
        fill=find(p_fl(:,1)==g_o(j));%find curr game
        pref_any(fill,1)=pet.prefAny(g_o(j));% preference in game?
        pref_win(fill,1)=pet.prefWin(g_o(j));% preferred team won?
        enjoy_(fill,1)=pet.enjoy(g_o(j));%level of enjoyment
        recall_a=[recall_a;recall];currnumposs=game_poss(g_o(j));
        curr_poss=zeros(currnumposs,1);rectime=nan(nPoss,1);%rectime=curr_poss+1000;
        for jj=1:currnumposs % all possessions
            %games 1 and 6 cut off the first possession due to 10 s cutoff, 
            %which could be remembered, so there is 1 less possession in 
            %this big matrix than in the fmri matrix. adjust by doing this.
            if g_o(j)==1 || g_o(j)==6;actposs=jj+1;else;actposs=jj;end
            %must be recalled w/ precision (1st column == 1), be the correct game and possession 
            rows=find(and(and(recall(:,1)==1,recall(:,2)==g_o(j)),recall(:,3)==actposs),1);
            fill=find(and(p_fl(:,1)==g_o(j),p_fl(:,3)==actposs));
            if ~isempty(fill);sur_order(i,jj+poss_e)=p_fl(fill,7);end
            if ~isempty(rows)
                curr_poss(jj)=1;
                % find the possession in our aligned matrix that fits
                rec_all_poss_xsub(i,fill)=1;% for Python analyses
                rec_all_poss_order(i,jj+poss_e)=1;
                %sometimes remembered possessions are cut out of the
                %initial 10 s so we need to qualify this here
                if fill>0;kk=kk+1;allingames(kk)=p_fl(fill,7);end 
                rectime(fill)=rows(1);
            end
        end
        %confabulation
        rows=find(and(recall(:,1)==3,recall(:,2)==g_o(j)));
        rows_d=diff(rows);fms=(length(find(rows_d~=1))+1);tms=sum(curr_poss);
        % # separate false memories 
        false_recall(i,g_o(j))=fms/(tms+fms);
        
        poss_s=poss_e+1;poss_e=poss_e+currnumposs;
        blk(poss_s:poss_e)=ceil(find(g_o==j)/3);
        P_IG(poss_s:poss_e)=1:currnumposs;%possession within game
        numpossorder(ii-tngs)=poss_e+0.5;
        precise_recall(i,g_o(j))=tms/currnumposs;
        j=j+1;
    end
    j=1;
    for ii=ph4(1):ph6(end) %read in data
        fn=[root 'data/recall/' sub '_game-' num2str(g_o(j)) '_gtrial-' num2str(ii) '.mat'];
        load(fn);prevgame=0;prevposs=0;jjjj=0;
        for jj=1:game_poss(g_o(j)) % all possessions
            %if this possession recalled
            if ~isempty(find(and(and(recall(:,1)==1,recall(:,2)==g_o(j)),recall(:,3)==jj),1))
                fill=find(and(p_fl(:,1)==g_o(j),p_fl(:,3)==jj));%find possession row
                if prevgame==g_o(j) %if same game
                    psur=p_fl(prevfill,7);psurN=p_fl(prevfill+1,7);
                    ll=jj-prevposs+(lag+1);lags(ll)=lags(ll)+1;
                    if ~isempty(psur);jjj=jjj+1;lagsur(jjj,1)=ll;
                        lagsur(jjj,2)=psur;lagsur(jjj,3)=psurN;end
                    if psur<=memsurmed(i);lags_ls(ll)=lags_ls(ll)+1;
                    elseif memsurmed(i)<psur;lags_hs(ll)=lags_hs(ll)+1;end 
                    if psurN<=memsurmed(i);lags_nr_ls(ll)=lags_nr_ls(ll)+1;
                    elseif memsurmed(i)<psurN;lags_nr_hs(ll)=lags_nr_hs(ll)+1;end 
                end
                prevgame=g_o(j);prevposs=jj;prevfill=fill;
                jjjj=jjjj+1;allposses(i,g_o(j),jjjj)=jj;
            end
        end
        j=j+1;
    end
    %normalize and log the lags
    q=lags;lags_xs(i,:)=q./nansum(q);
    q=lags_ls;lags_ls_xs(i,:)=q./nansum(q);
    q=lags_hs;lags_hs_xs(i,:)=q./nansum(q);
    q=lags_nr_ls;lags_nr_ls_xs(i,:)=q./nansum(q);
    q=lags_nr_hs;lags_nr_hs_xs(i,:)=q./nansum(q);
        
    [~,aa]=sort(recall_a(:,2));recall_a=recall_a(aa,:);
    X=p_fl(:,mcols);
    y=rec_all_poss_xsub(i,:)';
    if i==1;for ii=1:size(p_fl,2);p_fl2(:,ii)=p_fl(:,ii)-nanmean(p_fl(:,ii));end;end
    %simple outputs to python
    save ssrecall precise_recall rec_all_poss_xsub recall_a
    Sub_num=[Sub_num;repmat(str2double(sub),nPoss,1)];Mem=[Mem;y];
    Expert=[Expert;repmat(expert(i,3),nPoss,1)];Engage=[Engage;repmat(engage(i,2),nPoss,1)];
    PrefAny=[PrefAny;pref_any];PrefWin=[PrefWin;pref_win];Enjoy=[Enjoy;enjoy_];
    BeliefPerf=[BeliefPerf;repmat(bt.r,nPoss,1)];
    prevmat=nan(nPoss,1);prevmat(2:size(p_fl,1))=y(1:end-1);
    firsts=find(diff(p_fl(:,3))<0);prevmat(firsts+1)=nan;PrevMem=[PrevMem;prevmat];
    fill2=find(sum(p_fl2(:,[8:10]),2)==0);
    Sub_num2=[Sub_num2;repmat(str2double(sub),length(fill2),1)];Mem2=[Mem2;y(fill2)];
    Sub_numG=[Sub_numG;repmat(str2double(sub),tngs,1)];
    SurG=[SurG;gamebinsurmean'];
    MemG=[MemG;precise_recall(i,:)'];Block=[Block;blk'];PossInGame=[PossInGame;P_IG'];
    FalseMemG=[FalseMemG;false_recall(i,:)'];
end
cd([root 'analysis/']);

%% visualize all transitions for sample subject(s) - Fig S5
for i=6 %1:nS;
    figure;imagesc(squeeze(allposses(i,:,:)));h=colorbar;h.FontSize=25;
    set(get(h,'label'),'string','Possession number');xlabel('Recall order');
    ylabel('Game #');gg=gca;gg.FontSize=25;gg.LineWidth=3;
    sn=[root 'pics/SampRecallTransitions-' num2str(i)];print(sn,'-dpdf','-bestfit');
end

%median sur of actually recalled - must be run once before for this to save
%/ load the next time
q=rec_all_poss_xsub'.*p_fl(:,7);
for i=1:nS;memsurmed(i,1)=median(q(q(:,i)>0,i));end
%save memsurmed memsurmed; %saved previously

%% individual differences histogram - Fig 6A
pool=sum(rec_all_poss_xsub,2);
figure;hh=histogram(pool,10);hh.LineWidth=2;hh.FaceColor=[0 0 0];hold on;
gg=gca;gg.XLim=[0 60];gg.XTick=[0 10 20 30 40 50 60];
box off;gg.YLim(2)=7;gg.YTick=[0 3.5 7];gg.FontSize=25;gg.LineWidth=3;
xlabel('Possessions remembered');ylabel('# subjects');
plot([mean(pool) mean(pool)],[0 gg.YLim(2)],'--k','LineWidth',3);
sn=[root 'pics/IndDiffs'];print(sn,'-dpdf','-bestfit');%x-sub differences
disp(['mean recalled=' num2str(mean(sum(rec_all_poss_xsub,2))) ... %print mean / SEM
    'sem recalled=' num2str(std(sum(rec_all_poss_xsub,2))/sqrt(nS))])

%% scatterplot of possession memorability & correlate with surprise measures
row=7;splitbyfinal=0;load evSeg evs evs_w;evs_pre=nan(nPoss,1);
for i=1:tngs;fill=find(p_fl(:,1)==i);fill=fill(1:end-1);
    evs_pre(fill(1))=nan;evs_pre(fill+1)=evs(fill);end %create _pre metric
for i=1:size(rec_all_poss_xsub,2);possm(i)=mean(rec_all_poss_xsub(:,i));end %possession memorability
if splitbyfinal;fposs=find(diff(p_fl(:,3))<0);fposs(tngs)=nPoss;
    fill=1:nPoss;nfposs=find(diff(p_fl(:,3))==1);
    figure;scatter(p_fl(fposs,row),possm(fposs),'b');hold on;lsline;
    scatter(p_fl(nfposs,row),possm(nfposs),'r');
    r=corrcoef(p_fl(fposs,row),possm(fposs),'rows','complete');possm_r_f=r(2);
    r=corrcoef(p_fl(nfposs,row),possm(nfposs),'rows','complete');possm_r_nf=r(2);
    ylabel('Recall proportion');lsline;
else;figure;scatter(p_fl(:,row),possm,'b');hold on;
    ylabel('Recall proportion');%ind=p_fl(:,9)==1;%label oddness
    gg=gca;gg.FontSize=25;gg.LineWidth=3;lsl=lsline;lsl.Color=[0 0 0];lsl.LineStyle='--';
    %scatter(p_fl(ind,row),possm(ind),'r');%mark oddness in red
end
if row==7;xlabel('Surprise @ end of possession');
elseif row==12;xlabel('Entropy @ end of possession');end
r=corrcoef(p_fl(:,row),possm,'rows','complete');possm_r=r(2);
r=corr(p_fl(:,row),possm','Type','Spearman');possm_rS=r;
r=corrcoef(evs,possm,'rows','complete');possm_evs_r=r(2);
r=corrcoef(evs_pre,possm,'rows','complete');possm_evs_pre_r=r(2);
sn=[root 'pics/IndPossess-' num2str(row)];print(sn,'-dpdf','-bestfit');
title(['r = ' num2str(possm_r)]);
length(find(p_fl(:,row)>25))%find % of rows above 20% surprise
iters=10000;%scramble, preserving game structure
scram_meth=3;%1=scramble all,2=scramble game order,keep poss order,3=circ shift w/in game
for i=1:iters
    sur_i=[];sur_i2=[];sur_i_part=[];
    switch scram_meth;case 1;sur_i=p_fl(randperm(nPoss),7);
        case 2;fill=randperm(tngs);sur_i=[];
            for ii=1:tngs;f2=find(fill(ii)==p_fl(:,1));sur_i=[sur_i;p_fl(f2,row)];end
        case 3
            fill=randperm(tngs);
            for ii=1:tngs
                f2=find(ii==p_fl(:,1));
                proc=0;
                while proc==0;proc=1;seed=randperm(length(f2));seed=seed(1);
                    if seed<3;proc=0;elseif seed==length(f2);proc=0;end
                end
                f2=[f2(seed:end);f2(1:seed-1)];sur_i=[sur_i;p_fl(f2,row)];
            end
    end
    r=corrcoef(sur_i,possm,'rows','complete');possm_rs(i)=r(2);
    r=corr(sur_i,possm','Type','Spearman');possm_rsS(i)=r;
end
possm_r
possm_r_p=1-length(find(possm_r>possm_rs))/iters
possm_rS %spearman
possm_rS_p=1-length(find(possm_rS>possm_rsS))/iters

%same, but for each game
poss_e=0;maxsur=max(p_fl(:,7));maxmem=max(possm);possm_space=[];
for h=1:tngs;poss_s=poss_e+1;poss_e=poss_e+game_poss(h);
    possm_g=mean(rec_all_poss_xsub(:,poss_s:poss_e));
    figure;yyaxis left;plot(possm_g);xlabel(['Possession within game' num2str(h)]);g=gca;
    g.YLim=[0 maxmem];ylabel('Proportion of time remembered');yyaxis right;
    plot(p_fl(poss_s:poss_e,7));g=gca;g.YLim=[0 maxsur];
    ylabel('Surprise @ end of possession');sn=[root 'pics/IndPossessGame-' num2str(h)];
    print(sn,'-dpdf','-bestfit');
    poss_space=linspace(0,100,game_poss(h));possm_space=[possm_space;poss_space' possm_g'];
end

%% bootstrapping analysis - not in the paper but for reviewer point
iters=1000;d=1:nS;dl=length(d);bs_dist=zeros(iters,1);x2=p_fl(:,7);
scram_meth=3;%1=scramble all,2=scramble game order,keep poss order,3=circ shift w/in game
for h=1:iters
	subset=datasample(d,dl); %subset of subjects
    y2=[];
    %concatenate dataset using these subjects
    for ii=1:dl
        %find subject rows
        fill=Sub_num==subset(ii);
        y2=[y2 Mem(fill)];
    end
    y2=mean(y2,2);
    true_r=corrcoef(x2,y2); %this is the ground truth value for this bootstrap!
    
    for i=1:iters
        sur_i=[];
        switch scram_meth;case 1;sur_i=p_fl(randperm(nPoss),7);
            case 2;fill=randperm(tngs);sur_i=[];
                for ii=1:tngs;f2=find(fill(ii)==p_fl(:,1));sur_i=[sur_i;p_fl(f2,row)];end
            case 3
                fill=randperm(tngs);
                for ii=1:tngs
                    f2=find(ii==p_fl(:,1));
                    proc=0;
                    while proc==0;proc=1;seed=randperm(length(f2));seed=seed(1);
                        if seed<3;proc=0;elseif seed==length(f2);proc=0;end
                    end
                    f2=[f2(seed:end);f2(1:seed-1)];sur_i=[sur_i;p_fl(f2,row)];
                end
        end
        %run correlation for this bootstrap's particular shift
        r=corrcoef(sur_i,y2,'rows','complete');y2_rs(i)=r(2);
    end
    %find z-score of the true vs null distribution
    tog=[true_r(2) y2_rs];togz=zscore(tog);
    bs_dist(h)=togz(1); %assign to bootstrap distribution
end

%plot histogram of z-scores
figure;histogram(bs_dist);ylabel('# instances');xlabel('z (Mem and Sur)');
mean(bs_dist)
std(bs_dist)
pval=((iters-length(find(bs_dist>0)))/iters)*2 %p value

%% plot space from 0-100
mwin=5;sz=101;m_avg=zeros(sz,1);
for i=1:sz
    if i<mwin;win=1:i+10;elseif i>sz-mwin;win=i-10:sz;else;win=i-10:i+10;end
    fill=and(possm_space(:,1)>=win(1),possm_space(:,1)<win(end));
    m_avg(i)=mean(possm_space(fill,2));
end
figure;scatter(possm_space(:,1),possm_space(:,2));hold on;plot(0:sz-1,m_avg,'r');
xlabel('Proportion into game that a possession occurs');ylabel('Proportion of time remembered');
sn=[root 'pics/CollapseAcrossGamesSerialPosition'];print(sn,'-dpdf','-bestfit');

%% temporal contiguity - Fig 6B
[r,p]=corrcoef(lagsur(:,1),lagsur(:,2),'rows','complete');r(2) %is size of jump correlated to current surprise?
[r,p]=corrcoef(lagsur(:,1),lagsur(:,3),'rows','complete');r(2) 
mm=nanmean(lags_xs);for i=1:size(lags_xs,2);semm(i)=nanstd(lags_xs(:,i))/sqrt(nS);end
figure;errorbar(lags_x,mm,semm,'k','LineWidth',2);%hold on;plot(lags_x,mm,'k','LineWidth',3);
gg=gca;gg.FontSize=25;gg.LineWidth=3;
xlabel('Possession lag');ylabel({'Proportion of transitions','during recall'});box off;
sn=[root 'pics/AllLagsTogether'];print(sn,'-dpdf','-bestfit');
figure;%errorbar(lags_x,mm,semm);
xlabel('Possession lag');ylabel('Proportion of transitions');hold on;
mm=nanmean(lags_hs_xs);for i=1:size(lags_hs_xs,2);semm(i)=nanstd(lags_hs_xs(:,i))/sqrt(nS);end
errorbar(lags_x,mm,semm);box off;
mm=nanmean(lags_ls_xs);
for i=1:size(lags_ls_xs,2);semm(i)=nanstd(lags_ls_xs(:,i))/sqrt(nS);
    good=and(~isnan(lags_ls_xs(:,i)),~isnan(lags_hs_xs(:,i)));
    [~,p1(i)]=ttest(lags_ls_xs(good,i),lags_hs_xs(good,i));end
errorbar(lags_x,mm,semm);box off;
legend('Hi Sur','Lo sur');%
figure;hold on;
mm=nanmean(lags_nr_hs_xs);for i=1:size(lags_nr_hs_xs,2);semm(i)=nanstd(lags_nr_hs_xs(:,i))/sqrt(nS);end
errorbar(lags_x,mm,semm);box off;
mm=nanmean(lags_nr_ls_xs);
for i=1:size(lags_nr_ls_xs,2);semm(i)=nanstd(lags_nr_ls_xs(:,i))/sqrt(nS);
    good=and(~isnan(lags_ls_xs(:,i)),~isnan(lags_hs_xs(:,i)));
    [~,p2(i)]=ttest(lags_nr_ls_xs(:,i),lags_nr_hs_xs(:,i));end
errorbar(lags_x,mm,semm);
legend('Hi SurN','Lo surN');%'All',

figure;plot(lags_x,lags./nansum(lags));xlabel('Possession lag');ylabel('Proportion of transitions');
hold on;plot(lags_x,lags_hs./nansum(lags_hs));plot(lags_x,lags_ls./nansum(lags_ls));
plot(lags_x,lags_nr_hs./nansum(lags_nr_hs));plot(lags_x,lags_nr_ls./nansum(lags_nr_ls));
legend('All','Hi Sur','Lo sur','Hi SurN','Lo surN');
sn=[root 'pics/AllLags'];print(sn,'-dpdf','-bestfit');

%% export to R
%game level
T=table(Sub_numG,SurG,MemG,FalseMemG);
writetable(T,[root 'Rstuff/MemGame.csv'])%
%possession level - collapse across subjects
MemPoss=possm';
Sur=p_fl(:,7);
T=table(MemPoss,Sur);
writetable(T,[root 'Rstuff/PossMem.csv'])%
% possession level for each subject - rescaled version (below) = Fig 6C
%PossInGame=repmat(PossInGame,nS,1);
PrevMem=PrevMem-nanmean(PrevMem);
P_Num=repmat((1:nPoss)',nS,1);
GR=repmat(p_fl(:,4),nS,1);
PL=repmat(p_fl(:,origpflc+1),nS,1);
Sus=repmat(p_fl(:,5),nS,1);
Sur_pre=repmat(p_fl(:,6),nS,1);
Sur=repmat(p_fl(:,7),nS,1);
Ent_d_pre=repmat(p_fl(:,11),nS,1);
Ent_d=repmat(p_fl(:,12),nS,1);
Dunk=repmat(p_fl(:,8),nS,1);
Odd=repmat(p_fl(:,9),nS,1);
Turn=repmat(p_fl(:,10),nS,1);
Ent_dXSur=Ent_d+Sur;
if size(p_fl,2)>16;T=table(Sub_num,Expert,Engage,PrefAny,PrefWin,...
        Enjoy,P_Num,PossInGame,Block,GR,PL,Dunk,Odd,Turn,Sur,Sur_pre,Sus,Ent_d_pre,Ent_d,Ent_d+Sur,PrevMem,Mem);
else;T=table(Sub_num,Expert,Engage,PrefAny,PrefWin,Enjoy,P_Num,PossInGame,...
    Block,GR,PL,Dunk,Odd,Turn,Sur,Sur_pre,Sus,Ent_d_pre,Ent_d,PrevMem,Mem);end
writetable(T,[root 'Rstuff/LinMem.csv'])%
Expert=Expert-nanmean(Expert);Engage=Engage-nanmean(Engage);BeliefPerf=BeliefPerf-nanmean(BeliefPerf);
PrefAny=PrefAny-nanmean(PrefAny);PrefWin=PrefWin-nanmean(PrefWin);Enjoy=Enjoy-nanmean(Enjoy);
GR=repmat(p_fl2(:,4),nS,1);
PL=repmat(p_fl2(:,origpflc+1),nS,1);
Dunk=repmat(p_fl2(:,8),nS,1);
Odd=repmat(p_fl2(:,9),nS,1);
Turn=repmat(p_fl2(:,10),nS,1);
Sur=repmat(p_fl2(:,7),nS,1);
Sur_pre=repmat(p_fl2(:,6),nS,1);
Sus=repmat(p_fl2(:,5),nS,1);
Ent_d_pre=repmat(p_fl2(:,11),nS,1);
Ent_d=repmat(p_fl2(:,12),nS,1);
ent_d_nolast=p_fl2(:,12);fill=find(1==diff(p_fl(:,1)));ent_d_nolast(fill)=nan;
Ent_d_nolast=repmat(ent_d_nolast,nS,1);
T=table(Sub_num,Expert,Engage,PrefAny,PrefWin,...
        Enjoy,P_Num,PossInGame,Block,GR,PL,Dunk,Odd,Turn,Sur,Sur_pre,Sus,Ent_d_pre,Ent_d,PrevMem,Mem);
writetable(T,[root 'Rstuff/LinMemRS.csv'])%

save recall_vars rec_all_poss_xsub possm;
