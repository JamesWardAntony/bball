%% behavioral event segmentation on non-fMRI cohort - Fig 2
close all;clear;[~,root]=load_root();addpath(genpath([root 'analysis/']));
subs={'101','102','103','104','105','106','107','108','109','110',...
    '111','112','113','114','115'};
set(0,'DefaultFigurePosition',[150 150 1600 1200]);%defaults for plotting
set(0,'DefaultLineLinewidth',5);set(0,'DefaultAxesFontSize',30);
set(0,'DefaultLineMarkerSize',10);
set(groot,{'DefaultAxesXColor','DefaultAxesYColor'},{'k','k'})
gs=0;tngs=9;nS=length(subs);fact4y=1;%defaults
switch fact4y;case 0;load g0;load d_event_mat0;load gamebinmeans1
    case 1;load g1;load d_event_mat1;load gamebinmeans1;end
rs_a=zeros(nS,1);phs_a=zeros(nS,3);b_a=zeros(nS,4);%pre-assign
rb_a=rs_a;res_a=rs_a;rer_a=rs_a;resd_a=rs_a;rsq_a=rs_a;
repw_a=rs_a;rcw_a=rs_a;bsd_a=rs_a;bpw_a=rs_a;bsdpw_a=rs_a;
ph1=1:9;ph7=19:23;g_o_m2=cell(ph7(end),1);nPoss=length(p_fl);
prefhist=[];enjoy_mat=[];pred_mat=[];nclicks=zeros(nS,tngs);
clicks_a=zeros(1,tngs);clicks_a_w=zeros(1,tngs);
for i=1:nS %load data
    sub=char(subs(i));
    %fix data oddity for '101'
    if strcmpi(char(sub),'101');ph7=21:23;end 
    clear g_o_m2 g_o gb_o gs_o gf pet bt spt;phs=zeros(1,3);prefpython=[];
    dr=[root 'data/nonfmri/' sub '/'];
    cd(dr);
    load([sub '_phase1.mat'],'g_o_m','g_o','gb_o','gs_o','gf');
    g_o_m2(ph1)=g_o_m(ph1);
    load([sub '_phase7.mat'],'g_o_m');g_o_m2(ph7)=g_o_m(ph7);
    % pref/enjoyment - not in use
    for ii=ph1(1):ph1(end)
        enjoy_g_o(g_o(ii))=g_o_m2{ii}.enjoy(1);
        g_id=char(gf{g_o(ii)}.gf);j=1;
        for iii=1:length(g)
            if isfield(g{iii},'gf')
                if ii==1
                    summary_vars.Mean_Suspense(j)=g{iii}.Mean_Suspense;
                    Mean_Surprise(j)=g{iii}.Mean_Surprise;
                    summary_vars.Mean_Surprise_h(j)=g{iii}.Mean_Surprise_h;
                    summary_vars.Mean_ScoreDiff(j)=g{iii}.Mean_ScoreDiff;
                    summary_vars.seedDiff(j)=g{iii}.SeedDiff;
                    summary_vars.HomeWin(j)=g{iii}.HomeWin;
                    summary_vars.duration(j)=g{iii}.duration;j=j+1;
                end
                if strcmpi(g{iii}.gf,g_id)
                    fmrig=g{iii}.fmrig;
                    pet.trial(fmrig)=ii;pet.g_id(fmrig)=g_o(ii);
                    pet.pref(fmrig)=g_o_m2{ii}.pref(1);
                    pet.enjoy(fmrig)=g_o_m2{ii}.enjoy(1);
                    pet.Mean_Suspense(fmrig)=g{iii}.Mean_Suspense;
                    pet.Mean_Surprise(fmrig)=g{iii}.Mean_Surprise;
                    pet.Mean_Surprise_h(fmrig)=g{iii}.Mean_Surprise_h;
                    pet.Mean_ScoreDiff(fmrig)=g{iii}.Mean_ScoreDiff;
                    pet.seedDiff(fmrig)=g{iii}.SeedDiff;
                    pet.HomeWin(fmrig)=g{iii}.HomeWin;
                end
            end
        end
        if pet.pref(fmrig)==2;pet.prefWin(fmrig)=0;pet.prefAny(fmrig)=0;phs(2)=phs(2)+1;
        elseif pet.pref(fmrig)==1 && pet.HomeWin(fmrig)==1;pet.prefWin(fmrig)=1;pet.prefAny(fmrig)=1;phs(3)=phs(3)+1;
        elseif pet.pref(fmrig)==3 && pet.HomeWin(fmrig)==0;pet.prefWin(fmrig)=1;pet.prefAny(fmrig)=1;phs(3)=phs(3)+1;
        else;pet.prefWin(fmrig)=-1;pet.prefAny(fmrig)=1;phs(1)=phs(1)+1;
        end
        prefhist=[prefhist;pet.prefWin(fmrig) pet.seedDiff(fmrig) pet.pref(fmrig) pet.enjoy(fmrig)];
        phs_a(i,:)=phs;
        prefpython(fmrig,1)=pet.prefAny(fmrig);
        prefpython(fmrig,2)=pet.prefWin(fmrig);
        %segmentation performance
        c=g_o_m2{ii}.clickTimes;clicks2=[];
        if ~isempty(c)
            %don't keep counting when space bar pressed down
            clicks2=c(diff(c)>0.5);
            c1=[c(1) clicks2];clicks{g_o(ii)}.clicks=c1;%congregate
            cn=length(c1);nclicks(i,g_o(ii))=cn;%clicks / game
            l=length(find(clicks_a(:,g_o(ii))>0));
            subs_a(l+1:l+cn,g_o(ii))=i;
            clicks_a(l+1:l+cn,g_o(ii))=c1;
            clicks_a_w(l+1:l+cn,g_o(ii))=1/cn;%weighted by # clicks - not in use
        end
    end
    % prediction test - not in use
    bt.gr=[];bt.hwp=[];bt.position=[];
    for ii=ph7(1):ph7(end)
        if strcmpi(char(sub),'107') && ii==22 %sub 107 knew duke outcome
        elseif strcmpi(char(sub),'104') %104 no data
        elseif strcmpi(char(sub),'106') && ii==22
            qqqq=[];
        else;bt.gr=[bt.gr;g_o_m2{ii}.belief(:,1)];
            bt.hwp=[bt.hwp;g_o_m2{ii}.belief(1:end-1,2)];%cut 0/100% end event
            bt.position=[bt.position;g_o_m2{ii}.belief(1:end-1,3)];
        end
    end
    % correlations - not in use
    a=[pet.Mean_Suspense;pet.enjoy]';res=corrcoef(a(:,1),a(:,2));res=res(2);r=res;
    a=[pet.Mean_Surprise;pet.enjoy]';rer=corrcoef(a(:,1),a(:,2));rer=rer(2);r=rer;
    a=[pet.seedDiff;pet.enjoy]';resd=corrcoef(a(:,1),a(:,2));resd=resd(2);r=res;
    a=[pet.prefWin;pet.enjoy]';repw=corrcoef(a(:,1),a(:,2));repw=repw(2);r=repw;
    a=[pet.Mean_Surprise;nclicks(i,:)]';rcw=corrcoef(a(:,1),a(:,2));rcw=rcw(2);r=rcw;        
    if isempty(bt.hwp);rb=nan;
        else;a=[bt.hwp bt.position];rb=corrcoef(a(:,1),a(:,2));rb=rb(2);r=rb;bt.r=rb;end
    if gs;xl='True win prob (home)';yl='Guessed win prob (home)';corrEZ;end
    %simple outputs to python
    pref=pet.pref;enjoy=pet.enjoy;
    save g_o_m2 g_o_m2 g_o gb_o gs_o gf pet bt res rer rb pref prefpython enjoy enjoy_g_o phs;
    res_a(i)=res;rer_a(i)=rer;rb_a(i)=rb;
    resd_a(i)=resd;repw_a(i)=repw;rcw_a(i)=rcw;
end
cd([root 'analysis/']);
endpts=find(diff(p_fl(:,1))==1);endpts(tngs,1)=nPoss;%don't subtract - these are final posses
nendpts=find(diff(p_fl(:,1))==0);%create index that omits final poss - not in use
%% find best moving window size by correlating with true event boundaries
%note: didn't end up doing it this way (went with a priori mw=1)
mws=8;mw=1;cg=0;startcut=10;
nclicks_rg=sum(nclicks,2);
for iii=1:length(g)
    if isfield(g{iii},'gf')
        cg=cg+1;curr=clicks_a(clicks_a(:,cg)>0,cg);
        vsecs=ceil(g{iii}.duration);cps=nan(vsecs,1);cpsS=cps;
        binary=zeros(vsecs,1);binary(g{iii}.TRsur>0)=1;
        for ii=0:mws;for i=1:vsecs;cps(i)=length(find(and(curr>i-mw,curr<i+mw+ii)));
            r=corrcoef(cps,g{iii}.TRsur,'rows','complete');c_r(ii+1,cg)=r(2);
            r=corrcoef(cps,binary,'rows','complete');binc_r(ii+1,cg)=r(2);end;end
    end
end
[maxcorr,mw]=max(nanmean(c_r,2));%find max
[binmaxcorr,~]=max(nanmean(binc_r,2)) %strongest correlation with just the time course
mw=1;%override with a priori
%% explore x-sub data a little
figure;imagesc(nclicks);xlabel('games');ylabel('subs');%all subs x games
cl=colorbar;cl.Label.String='Clicks';
cm=sum(nclicks,2);
disp(['Max=' num2str(max(cm)) ',min=' num2str(min(cm)) ',mean=' ...
    num2str(mean(cm)) ',ste=' num2str(std(cm)/sqrt(nS))]) %print relevant summary stats
figure;h=histogram(cm,100);xlabel('# clicks, all games');ylabel('# subs');
for i=1:nS;r=corrcoef(gamebinsurmean,nclicks(i,:));sur_ev_rs(i)=r(2);end
cm2=sum(nclicks,1);%x-game segmentations
figure;scatter(gamebinsurmean,cm2);r=corrcoef(gamebinsurmean,cm2);sur_ev_xg_r=r(2)
[~,p]=ttest(sur_ev_rs',zeros(nS,1)) %sig diff x-sub
iters=10000;%scramble, preserving game structure
for i=1:iters
    sur_i=[];fill=randperm(tngs);
    r=corrcoef(gamebinsurmean(fill),cm2,'rows','complete');sur_ev_xg_rs(i)=r(2);
end
sur_ev_xg_r
sur_ev_xg_r_perc=1-length(find(sur_ev_xg_r>sur_ev_xg_rs))/iters

%split into clusters of subjects w/ different strategies - not in use but 
%could be informative sometime with a larger dataset
[kmid,C]=kmeans(cm,3);hold on;plot([C C],[0 max(h.Values)]);
runvers=1;%1=all subs (in use),2-4 different K-means clusters of subjects
switch runvers;case 1;subset=1:nS;case 2;subset=find(kmid==1);
    case 3;subset=find(kmid==2);case 4;subset=find(kmid==3);end
figure;cg=0;wps=[];surs=[];evs=[];evs_w=[];fs=18;lw=3;
curr_set=zeros(size(clicks_a,1),size(clicks_a,2));
for i=1:size(clicks_a,1)
    for ii=1:size(clicks_a,2)
        for iii=1:length(subset)
            if subs_a(i,ii)==subset(iii)
                curr_set(i,ii)=1;%set=1 for relevant data points
            end
        end
    end
end
%% create game-by-game plots of all subjects w/ true boundaries - Fig 2A
for iii=1:length(g)
    if isfield(g{iii},'gf')
        cg=cg+1;
        curr=clicks_a(clicks_a(:,cg)>0,cg);
        curr_w=clicks_a_w(clicks_a_w(:,cg)>0,cg);%weighted by # clicks
        vsecs=ceil(g{iii}.duration);cps=nan(vsecs,1);cps_w=cps;
        for ij=1:length(subset);for i=1:vsecs;ii=subset(ij);
                cpsS(i,ii)=length(find(and(and(curr>i-mw,... %x-sub!
                    curr<=i+mw),ii==subs_a(1:length(curr),cg))));end;end
        cpsS(cpsS>1)=1;
        %plot ticks & boundary agreement
        figure;subplot(211);imagesc(abs(1-cpsS)');xlabel('TRs');ylabel('Subject');colormap gray;ppos;
        set(gca,'YTick',5:5:length(subset),'XTick',100:100:400,'XLim',[0 length(cps)],'YColor',[0 0 0],'FontName',...
            'Helvetica','XColor',[0 0 0],'FontSize',fs,'box','off','LineWidth',lw);
        gg=gca;gg.XAxis.Visible='off';gg.XTick=[];
        subplot(212);plot(mean(cpsS,2),'k','LineWidth',lw);ylabel('Agreement');ppos;xlabel('TRs');
        set(gca,'YTick',0:0.25:1,'YLim',[0 0.75],'XLim',[0 length(cps)],'XTick',100:100:400,'YColor',[0 0 0],...
            'XColor',[0 0 0],'FontName','Helvetica','FontSize',fs,'box','off','LineWidth',lw);
        sn=[root 'pics/EventSegAllsubs-' num2str(cg)];print(sn,'-dpdf','-bestfit');
        %plot single row of boundaries
        bd=diff(g{iii}.TRpn);bd(2:end+1)=bd;bd(1)=0;bd(isnan(bd))=0;bds=find(bd>0);
        for i=1:vsecs;bd2(i)=length(find(and(bds>i-mw,bds<=i+mw)));end
        figure;imagesc(abs(1-bd2));colormap gray;
        h=gcf;set(h,'Position',[50 50 1200 100]);set(h,'PaperOrientation','landscape');
        set(gca,'XTick',100:100:400,'FontName','Helvetica','XLim',[0 length(cps)],...
            'FontSize',fs,'box','off','LineWidth',lw);
        gg=gca;gg.XAxis.Visible='off';gg.XTick=[];gg.YAxis.Visible='off';gg.YTick=[];
        sn=[root 'pics/EventSegBounds-' num2str(cg)];print(sn,'-dpdf','-bestfit');
        %adjust curr by subset to compute 'ev'
        curr=clicks_a(clicks_a(curr_set(:,cg)==1,cg)>0,cg);
        for i=1:vsecs 
            fill=find(and(curr>i-mw,curr<=i+mw));
            doubles=length(find(diff(fill)==1));
            cps(i)=(length(fill)-doubles)/length(subset);
            cps_w(i)=sum(curr_w(fill));%'weighted' version
        end
        dm=diff(g{iii}.TRpn)~=0;
        fill=find(and(dm,~isnan(g{iii}.TRpn(1:end-1))));fill=fill+1;%adj for diff
        nanyes=find(isnan(g{iii}.TRpn));if isempty(nanyes);fill=[fill;vsecs];end%if no nans
        fill2=find(fill<=startcut);
        if ~isempty(fill2);fill=fill(length(fill2)+1:end);end%adjust for startcut
        %log across-possession values for later correlations
        wp=g{iii}.TRwph(fill);wps=[wps;wp];
        sur=g{iii}.TRsur(fill);surs=[surs;sur];
        ev=cps(fill);evs=[evs;ev];evs_w=[evs_w;cps_w(fill)];
        bcsur=(g{iii}.TRsur(fill)-g{iii}.TRent(fill))/2;
        bicsur=(g{iii}.TRsur(fill)+g{iii}.TRent(fill))/2;
        if cg==4 %couple sample plots for viz - Fig 2A
            figure;yyaxis left;plot(ev,'LineWidth',lw);ylabel('Agreement');ppos2;xlabel('Possession');
            set(gca,'YTick',0:0.25:0.75,'YLim',[0 0.75],'XTick',0:5:25,'XLim',[0 length(cps(fill))],'FontName',...
                'Helvetica','XColor',[0 0 0],'FontSize',fs,'box','off','LineWidth',lw);
            yyaxis right;plot(sur,'LineWidth',lw);ylabel('Surprise');
            set(gca,'YTick',0:10:30,'YLim',[0 30],'FontName',...
                'Helvetica','XColor',[0 0 0],'FontSize',fs,'box','off','LineWidth',lw);
            sn=[root 'pics/EventSegVSur-' num2str(cg)];print(sn,'-dpdf','-bestfit');
            figure;yyaxis left;plot(ev,'LineWidth',lw);ylabel('Agreement');ppos2;xlabel('Possession');
            set(gca,'YTick',0:0.25:0.75,'YLim',[0 0.75],'XTick',0:5:25,'XLim',[0 length(cps(fill))],'FontName',...
                'Helvetica','XColor',[0 0 0],'FontSize',fs,'box','off','LineWidth',lw);
            yyaxis right;plot(bcsur,'LineWidth',lw);ylabel({'Belief-consistent';'surprise'});
            set(gca,'YTick',0:10:30,'YLim',[0 30],'FontName',...
                'Helvetica','XColor',[0 0 0],'FontSize',fs,'box','off','LineWidth',lw);
            sn=[root 'pics/EventSegVBCSur-' num2str(cg)];print(sn,'-dpdf','-bestfit');
            figure;yyaxis left;plot(ev,'LineWidth',lw);ylabel('Agreement');ppos2;xlabel('Possession');
            set(gca,'YTick',0:0.25:0.75,'YLim',[0 0.75],'XTick',0:5:25,'XLim',[0 length(cps(fill))],'FontName',...
                'Helvetica','XColor',[0 0 0],'FontSize',fs,'box','off','LineWidth',lw);
            yyaxis right;plot(bicsur,'LineWidth',lw);ylabel({'Belief-inconsistent';'surprise'});
            set(gca,'YTick',0:10:30,'YLim',[0 30],'FontName',...
                'Helvetica','XColor',[0 0 0],'FontSize',fs,'box','off','LineWidth',lw);
            sn=[root 'pics/EventSegVBICSur-' num2str(cg)];print(sn,'-dpdf','-bestfit');
        end
        if cg==tngs %if final game, make scatterplot 
            r=corrcoef(evs,surs);possev_r=r(2);r=corrcoef(evs_w,surs);possev_w_r=r(2);
            r=corrcoef(evs(nendpts),surs(nendpts));possev_r_nend=r(2);maxsur=max(surs);
            figure;yyaxis left;plot(evs,'b');gg=gca;gg.YLim=[0 1];
            ylabel('Event boundary agreement');
            hold on;yyaxis right;plot(surs,'r');gg=gca;gg.YLim=[0 maxsur];
            ylabel('Surprise');xlabel('Possession number');
            sn=[root 'pics/EventSegVSurAll'];print(sn,'-dpdf','-bestfit');
            figure;scatter(surs(endpts),evs(endpts),'r');hold on;
            scatter(surs(nendpts),evs(nendpts),'b');
            gg=gca;gg.YLim=[0 1];gg.XLim=[0 maxsur];
            ylabel('Event boundary agreement');xlabel('Surprise');lsline;
            sn=[root 'pics/EventSegVSurAllScat'];print(sn,'-dpdf','-bestfit');
        end
    end
end

%% surprise correlations
col=77;%7=surprise,12=entropy,71=belief-consistent sur,77="-inconsistent sur
switch col;case 7;col_d=p_fl(:,col);col_part=p_fl(:,12);
    case 12;col_d=p_fl(:,col);col_part=p_fl(:,7);
    case 71;col_d=p_fl(:,7)-p_fl(:,12);col_part=p_fl(:,7);
    case 77;col_d=p_fl(:,7)+p_fl(:,12);col_part=p_fl(:,7);end
r=corrcoef(evs,col_d);possev_r=r(2); %true correlation
if col==77;col_d2=p_fl(:,7)-p_fl(:,12);%if bic, also find bc sur to contrast
    r2=corrcoef(evs,col_d2);bic_bc_r=r(2)-r2(2);end
r=partialcorr(evs,col_d,col_part);possev_r_part=r;%partial correlation, not in use
r=corrcoef(evs(nendpts),col_d(nendpts));possev_r_nend=r(2);%if nan-ing out the final possession
iters=10000;%num scrambles
scram_meth=3;%1=scramble all, 2=scramble w/in game, 3=circ shift w/in game
for i=1:iters
    sur_i=[];sur_i2=[];sur_i_part=[];sur_i_nend=[];sur_i2_nend=[];
    switch scram_meth;case 1;sur_i=col_d(randperm(nPoss));sur_i_part=col_part(randperm(nPoss));
        if col==77;sur_i2=col_d2(randperm(nPoss));end
    case 2;fill=randperm(tngs);
        for ii=1:tngs;f2=find(fill(ii)==p_fl(:,1));
            sur_i=[sur_i;col_d(f2)];sur_i_part=[sur_i_part;col_part(f2)];
            sur_i_nend=[sur_i_nend;col_d(f2(1:end-1))];
            if col==77;sur_i2=[sur_i2;col_d2(f2)];
                sur_i2_nend=[sur_i2_nend;col_d2(f2(1:end-1))];end
        end
    case 3;fill=randperm(tngs);
        for ii=1:tngs;f2=find(ii==p_fl(:,1));
            %f2=find(fill(ii)==p_fl(:,1));
            proc=0;
            while proc==0;proc=1;seed=randperm(length(f2));seed=seed(1);
                if seed<3;proc=0;elseif seed==length(f2);proc=0;end
            end
            f3=[f2(seed:end);f2(1:seed-1)];
            f4=[f2(seed:end-1);f2(1:seed-1)];%for nanend
            sur_i=[sur_i;col_d(f3)];sur_i_part=[sur_i_part;col_part(f3)];
            sur_i_nend=[sur_i_nend;col_d(f4)];
            if col==77;sur_i2=[sur_i2;col_d2(f3)];
                sur_i2_nend=[sur_i2_nend;col_d2(f4)];end
        end
    end
    r=corrcoef(sur_i,evs,'rows','complete');possev_rs(i)=r(2);%iteration value
    r=partialcorr(sur_i,evs,sur_i_part,'rows','complete');possev_r_parts(i)=r;
    r=corrcoef(sur_i_nend,evs(nendpts),'rows','complete');possev_r_nends(i)=r(2);
    if col==77;r=corrcoef(sur_i2,evs,'rows','complete');bic_bc_rs(i)=r(2)-r2(2);
        r=corrcoef(sur_i2_nend,evs(nendpts),'rows','complete');bic_bc_rs(i)=r(2)-r2(2);end
end
possev_r %true
possev_r_perc=1-length(find(possev_r>possev_rs))/iters; %rank against permutation tests
possev_r_p=(0.5-abs(0.5-possev_r_perc))*2 %p val
possev_r_part
possev_r_part_perc=1-length(find(possev_r_part>possev_r_parts))/iters
possev_r_nend
possev_r_nend_perc=1-length(find(possev_r_nend>possev_r_nends))/iters
if col==77;bic_bc_r
    bic_bc_r_perc=1-length(find(bic_bc_r>bic_bc_rs))/iters
end
close all;fs=64;lw=9;
%sn=[root 'pics/EventSegRDist-' num2str(col) '-' num2str(runvers)];
sn=[root 'pics/EventSegRDist-' num2str(col)];
vplot(sn,possev_rs,possev_r,fs,lw); %create violin plot of results - Fig 2C
save evSeg evs evs_w;
