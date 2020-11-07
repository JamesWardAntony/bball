%% loadLogAndTR
% loads in game information and assigns variables for each TR
% also plots win probabilities and surprise, as in Fig 1B and Supp Fig 1
%% read in game info from annotated Excel file
fmrig=fmrig+1;
[~,~,raw]=xlsread([root 'analysis/IngameVars.xlsx'],homestring);%read data
raw=raw(2:end,1:12);raw(cellfun(@(x) ~isempty(x) && isnumeric(x) && isnan(x),raw)) = {''};
R=cellfun(@(x) ~isnumeric(x) && ~islogical(x),raw);raw(R) = {NaN};
data = reshape([raw{:}],size(raw));gl = table;
gl.SecondsInVideo = data(:,1);gl.GameRemaining = data(:,2);
gl.EventCode = data(:,3);gl.NBAeFG = data(:,4);
gl.HomeScoreChange = data(:,5);gl.VidWinProb = data(:,6);
gl.HomeScoreDiff = data(:,7);gl.InferredWinProb = data(:,8);
gl.HomePoss = data(:,9);gl.PossNum = data(:,10);gl.BSur=data(:,12);
ftperc=76;% average free throw percentage (old value: %69.05;) (https://www.ncaa.com/stats/basketball-men/d1/current/team/150)
proc=0;switch fact4y;case 1;proc=1;end
if proc;exp_v=exp_vict_h(game);%adjust by exp_v based on difference between teams
    currpred=zeros(length(gl.GameRemaining),1);%calculate using our model
    for i=1:length(gl.GameRemaining)
        if gl.GameRemaining(i)==0;currpred(i)=gl.InferredWinProb(i);
        elseif isnan(gl.GameRemaining(i));currpred(i)=gl.InferredWinProb(i);
        else;[~,SLbin]=min(abs(gl.GameRemaining(i)-SecB2));
            if SLbin>size(f4,1);SLbin=size(f4,1);end
            hp=gl.HomePoss(i);if hp==0;hp=2;end %assign home possession value
            AdjScoreDiff=gl.HomeScoreDiff(i)+gl.GameRemaining(i)*exp_v;
            diffs=abs(AdjScoreDiff-ScoreDiffB);hsd=find(diffs<1);%ScoreDiff for no adj
            if numel(hsd)==2;hsdw=1-diffs(hsd);%weights for interpolating between values!
            elseif numel(hsd)==0;[~,hsd]=min(diffs);%if outside range
            end
            if find(hsd>size(f4,2));hsd=size(f4,2);hsdw=1;end
            if gl.HomePoss(i)==0.5;possh=sum((f4(SLbin,hsd,1).*hsdw))*100;possv=sum((f4(SLbin,hsd,2).*hsdw))*100;
                    currpred(i)=mean([possh possv]);
            else;currpred(i)=sum((f4(SLbin,hsd,hp).*hsdw))*100;end
        end
    end
    gflined=[gflined;currpred gl.InferredWinProb gl.EventCode];%aligned variables
    if gs;figure;scatter(currpred,gl.InferredWinProb);%plot if desired
        xlabel('3-factor win probability');ylabel('Ken Pomeroy win probability');
        [r,p]=corrcoef(currpred,gl.InferredWinProb,'rows','complete');r(2)
        sn=[root 'pics/WinProbModelComp-' num2str(game)];print(sn,'-dpdf','-bestfit');
    end
    %re-assign to our model instead of KP!
    switch fact4y;case 1;gl.InferredWinProb=currpred;gl.VidWinProb=currpred;end
end
gl.VidSuspense=NaN(size(data,1),1);gl.VidSurprise=gl.VidSuspense;gl.VidSurprise_h=gl.VidSuspense;
gl.VidSurprise(2:end)=abs(diff(gl.InferredWinProb));
gl.VidSurprise_h(2:end)=diff(gl.InferredWinProb);
glkeep=~isnan(gl.SecondsInVideo);gl=gl(glkeep,:);
HomeRight=data(1,11);% ball going right for home team or not
%i annotated, for every shot, the likelihood of it going in using the below
%shot chart (used for the NBA) and converting it to NCAA using broad %s.
%this allows one to compute eFG% for each shot. NOTE: not used.
%momentary effective field-goal percentage, eFG
%shot chart source: http://www.sloansportsconference.com/wp-content/uploads/2014/02/2014-SSAC-Quantifying-Shot-Quality-in-the-NBA.pdf
%NBA 2014 (year of shot chart): https://www.teamrankings.com/nba/stat/effective-field-goal-pct?date=2014-06-15
%NCAA 2012 (year of games): https://www.teamrankings.com/ncaa-basketball/stat/effective-field-goal-pct?date=2012-04-02
nba2col=48.88/50.05;%conversion from nba to ncaa shot
gl.eFG=(gl.NBAeFG*nba2col)/100;%adjust nba to ncaa
ftcols=find(gl.EventCode==1);%free throws
gl.eFG(ftcols)=(ftperc/100)/2;%adjust by half b/c only 1 pt
clearvars data raw R;
%% match suspense up to possessions
endrows=find(and(g{game}.GameTime>EndTimes(1),g{game}.GameTime<=EndTimes(2)));
endrows(2:length(endrows)+1)=endrows;endrows(1)=endrows(2)-1;%add preceding time point for start of clip
prob_end=g{game}.WinProbHome(endrows);%win prob for end of game
seed_diff=((9-g{game}.HomeSeed)-1)*2+1;
gr=1200-g{game}.GameTime(endrows);
hsdt=g{game}.ScoreDiff(endrows);
hpt=g{game}.HomePoss(endrows);
WinProbWinning=abs(gl.InferredWinProb-50)+50;%win prob for winning team
for i=1:length(gl.VidSuspense)
    sbin=SecB-gl.GameRemaining(i);sbin=find(sbin>0);sbin=sbin(end);
    wbin=WinB-WinProbWinning(i);wbin=find(wbin<0);wbin=wbin(end);
    gl.VidSuspense(i)=suspense_smooth(sbin,wbin);
end
oldway=0;%was previously not re-indexing each time
if oldway
    for i=1:length(endrows)
        %find row of that matches the video
        fill=find(and(and(abs(gr(i)-gl.GameRemaining)<=2,hsdt(i)==gl.HomeScoreDiff),hpt(i)==gl.HomePoss));
        %fill=find(abs(prob_end(i)-gl.VidWinProb)<0.001);
        %fill in suspense with that row...
        gl.VidSuspense(fill)=g{game}.i_sus(i);
    end
    %fix weird error in Gonzaga game
    if game==27;gl.VidSuspense(1)=gl.VidSuspense(2);end 
    %if nan, assign to previous suspense level
    prevsus=gl.VidSuspense(1);
    for i=1:length(gl.VidSuspense)
        if isnan(gl.VidSuspense(i));gl.VidSuspense(i)=prevsus;
        else;prevsus=gl.VidSuspense(i);
        end
    end
end
gl.VidSuspense(end)=0;%no suspense at game end
%momentary suspense - essentially mean score change likely for a shot X 
%suspense. this would only really matter for the < 2 seconds as shot is
%attempted and not yet hit the rim...NOTE: not in use
gl.susXeFG=gl.eFG.*gl.VidSuspense;
%% assign variables for each TR
clear TRsecs TRsus TRsur TRsur_h TRwph TRsusXeFG
% calculate sensory regressors by loading in files - only calculate once!
TR=1;vdims=[720 1280];fn=[root 'analysis/upper_lum-' num2str(game) '.mat'];fsec=43;lsec=48;
if ~exist(fn,'file') % if we haven't already calculated this...
    tic; %read in video below
    v=VideoReader([root 'exp/vids/720res/' gf 'cut.mov']);lum=zeros(floor(v.Duration*v.FrameRate),1);
    lum_temp=zeros(ceil(v.FrameRate),vdims(1),vdims(2));%temporary luminance for each second
    lum_f=zeros(floor(v.Duration),vdims(1),vdims(2));%stores mean of that second for x/y
    mot=lum;
    mot_temp=zeros(ceil(v.FrameRate),vdims(1),vdims(2));%temporary motion for each second
    mot_f=zeros(floor(v.Duration),vdims(1),vdims(2));%stores mean of that second for x/y
    j=1;jj=1;jjj=1;
    while hasFrame(v);video=readFrame(v);if j==1;prevvideo=video;end
        rgbv=rgb2gray(video);lum(j)=mean2(rgbv);lum_temp(jj,:,:)=rgbv;
        %from https://www.mathworks.com/matlabcentral/answers/53581-key-frames-extraction-in-a-video
        rgbpv=rgb2gray(prevvideo);imdiff=imabsdiff(rgbv,rgbpv);mot(j)=sum(sum(imdiff));%global motion
        mot_temp(jj,:,:)=imdiff;%frame-by-frame motion
        j=j+1;jj=jj+1;
        if jj>v.FrameRate;lum_f(jjj,:,:)=mean(lum_temp,1);jjj %print to update progress
            lum_temp=zeros(floor(v.Duration),vdims(1),vdims(2));
            mot_f(jjj,:,:)=mean(mot_temp,1);
            mot_temp=zeros(ceil(v.FrameRate),vdims(1),vdims(2));jj=1;jjj=jjj+1;
        end
        prevvideo=video;
        %if run through 2x, show first imdiff
        %figure;imagesc(imdiff);sn=[root 'pics/ImdiffEx'];print(sn,'-dpdf','-bestfit');
    end
    mot(1)=0;%no motion on first frame b/c no difference yet.
    if jj>1;lum_f(jjj,:,:)=mean(lum_temp,1);mot_f(jjj,:,:)=mean(mot_temp,1);end%final (partial) second
    clear lum_temp mot_temp;secmax=ceil(v.Duration);%ceil(max(gl.SecondsInVideo)/TR);%corr for last few seconds of vid!!
    [aud,Fs] = audioread([root 'exp/vids/720res/' gf 'cut.mov']);%read in audio!
    [upper,lower]=envelope(aud,Fs/100);upper=upper(:,1);lower=lower(:,1);%find envelope
    fn=[root 'analysis/upper_lum-' num2str(game) '.mat'];
    save(fn,'upper','Fs','lum','lum_f','mot','mot_f','v','secmax','-v7.3');toc;%save!
    %plot some examples (see Fig 4B)
    if game==5
        fpt=Fs*fsec;lpt=Fs*lsec;c1=upper(fpt:Fs/50:lpt);c2=lower(fpt:Fs/50:lpt);x=1:length(c1);x=x';x2=[x,fliplr(x)];bet=[c1,fliplr(c2)];
        figure;plot(c1,'k');hold on;plot(c2,'k');clear fill;fill(x2',bet','k');%plot Norfolk St example
        gg=gca;gg.XAxis.Visible='off';gg.YAxis.Visible='off';gg.XTick=[];gg.YTick=[];
        sn=[root 'pics/AudEx-' num2str(game)];print(sn,'-dpdf','-bestfit');
        fpt=v.FrameRate*fsec;lpt=v.FrameRate*lsec;c1=lum(fpt:lpt);
        figure;plot(fpt:lpt,c1,'k');%plot Norfolk St example
        gg=gca;gg.XAxis.Visible='off';gg.YAxis.Visible='off';gg.XTick=[];gg.YTick=[];
        sn=[root 'pics/LumEx-' num2str(game)];print(sn,'-dpdf','-bestfit');
        fpt=v.FrameRate*fsec;lpt=v.FrameRate*lsec;c1=mot(fpt:lpt);
        figure;plot(fpt:lpt,c1,'k');
        gg=gca;gg.XAxis.Visible='off';gg.YAxis.Visible='off';gg.XTick=[];gg.YTick=[];
        sn=[root 'pics/MotEx-' num2str(game)];print(sn,'-dpdf','-bestfit');
        fpt=1;lpt=v.FrameRate*120;c1=mot(fpt:lpt);
        figure;plot((fpt:lpt)/v.FrameRate,c1,'k');
        xlabel('Time (s)');ylabel('Global video motion (AU)');
        sn=[root 'pics/MotExTwoMins-' num2str(game)];print(sn,'-dpdf','-bestfit');
    end
else;load(fn,'upper','Fs','lum','mot','v','secmax');% if we've done this, then just load
end

TRsecs(:,1)=linspace(0,secmax*TR,secmax+1);
fn=[root 'analysis/f0proc.mat'];%speech variables
if ~exist(fn,'file');loadf0;TRspeech=f0m;f0ms{fmrig}.f0m=f0m;
    if fmrig==9;save(fn,'f0ms');end %save if last game
else;load(fn);TRspeech=f0ms{fmrig}.f0m;TRspeech(TRspeech==0)=nan;%else load
end

%save # of camera angle changes 
maxthresh=40000000;%~ threshold of total screen refresh
n_camang_mot(fmrig)=length(find(mot>maxthresh));%# of putative camera shifts
tot_mot(fmrig)=sum(mot);%total motion from whole video
if fmrig==9;save n_camang_mot n_camang_mot tot_mot;end %save if last game

% pre-assign a bunch of variables
TRsus=zeros(length(TRsecs)-1,1);
TRsur=TRsus;TRBsur=TRsus;TRsur_h=TRsus;TRwph=TRsus;TRent=TRsus;TRhp=TRsus;
TRsusXeFG=TRsus;TRaud=TRsus;TRodd=TRsus;TReFG=TRsus;TRlum=TRsus;
TRmot=TRsus;TRgr=TRsus;TRpn=TRsus;TRto=TRsus;TRft=TRsus;TRft_up=TRsus;
TRtwopt=TRsus;TRthreept=TRsus;TRsc=TRsus;TRsc_h=TRsus;TRwph_b=TRsus;
TRwtp=TRsus;TRrl=TRsus;TRdunk=TRsus;TRodd=TRsus;%TRspeech=TRsus;
maxfill=size(gl,1);prevwph=gl.InferredWinProb(1);
for i=1:length(TRsecs)-1 %cycle through all TRs and assign values
    fill=find(and(gl.SecondsInVideo>=TRsecs(i),gl.SecondsInVideo<TRsecs(i+1)));
    tb=(TRsecs(i)*Fs+1:TRsecs(i+1)*Fs);
    TRaud(i,1)=mean(upper(tb));
    if i<length(TRsecs)-1;tb=(floor(TRsecs(i)*v.FrameRate)+1:floor(TRsecs(i+1)*v.FrameRate));
    else;tb=(floor(TRsecs(i)*v.FrameRate)+1:length(lum));
    end
    TRlum(i,1)=mean(lum(tb));TRmot(i,1)=mean(mot(tb));%find avg luminance/motion for each second
    %IF at a changepoint
    if ~isempty(fill)
        TRsus(i,1)=mean(gl.VidSuspense(fill));
        TRsusXeFG(i,1)=mean(gl.susXeFG(fill));
        TReFG(i,1)=mean(gl.eFG(fill));
        TRsc(i,1)=abs(mean(gl.HomeScoreChange(fill)));
        TRsc_h(i,1)=mean(gl.HomeScoreChange(fill));
        TRsur(i,1)=mean(gl.VidSurprise(fill));
        TRBsur(i,1)=mean(gl.BSur(fill));
        TRgr(i,1)=mean(gl.GameRemaining(fill));
        TRsur_h(i,1)=mean(gl.VidSurprise_h(fill));
        TRwph(i,1)=mean(gl.InferredWinProb(fill));
        wphdiff=TRwph(i,1)-prevwph;%with or against the current predominant belief?
        if prevwph>50;TRent(i,1)=-wphdiff;else;TRent(i,1)=wphdiff;end
        %TRent(i,1)=(50-abs(TRwph(i,1)-50))-(50-abs(prevwph-50));%old way
        %where crossing 50 actually made a difference - pure entropy rather
        %than belief change
        TRhp(i,1)=mean(gl.HomePoss(fill));
        TRpn(i,1)=mean(gl.PossNum(fill));
        prevsus=TRsus(i,1);prevwph=TRwph(i,1);prevhp=TRhp(i,1);
        prevgr=TRgr(i,1);prevpn=TRpn(i,1);prevfill=fill(1);
        if gl.EventCode(fill(1))==11;TRdunk(i,1)=1;end %dunks
        if gl.EventCode(fill(1))==12;TRodd(i,1)=1;end %odd possessions
        if gl.EventCode(fill(1))==1;TRft(i,1)=1; %free throws
        elseif gl.EventCode(fill(1))==2 || gl.EventCode(fill(1))==5 ...
                || gl.EventCode(fill(1))==8 || gl.EventCode(fill(1))==11;TRtwopt(i,1)=1;%two pt shot
        elseif gl.EventCode(fill(1))==3 || gl.EventCode(fill(1))==6 ...
                 || gl.EventCode(fill(1))==9;TRthreept(i,1)=1;%three pt shot
        elseif gl.EventCode(fill(1))==4 || gl.EventCode(fill(1))==12;TRto(i,1)=1;%turnover
        end
    else %if NOT at changepoint, keep suspense, win probability, home possession, game remainint, possession #, freethrow upcoming updated
        TRsus(i,1)=prevsus;TRwph(i,1)=prevwph;
        TRhp(i,1)=prevhp;TRgr(i,1)=prevgr;TRpn(i,1)=prevpn;
        if prevfill<maxfill
            if gl.EventCode(prevfill+1)==1
                TRft_up(i,1)=1;% free-throw upcoming
            end
        end
    end
end
%final adjustments
TRsecs=TRsecs(1:end-1);
TRcp=courtpos{fmrig}.courtpos;%assign court position based on manual annotations
if numel(TRcp)<numel(TRsecs);TRcp(numel(f0m)+1:numel(TRsecs))=TRcp(end);
elseif numel(TRcp)>numel(TRsecs);TRcp=TRcp(1:numel(TRsecs));end
TRwph_b(TRwph>50,1)=1;%home likely to win, binarized
TRwtp(TRwph_b+TRhp~=1)=1;%winning team has ball
TRrl(HomeRight+TRhp==2)=1;TRrl(HomeRight+TRhp==0)=1;% ball going to the right
%% assign variables to 'g' and plot factors over time
g{game}.gf=gf;g{game}.duration=v.Duration;g{game}.fmrig=fmrig;
g{game}.TRsecs=TRsecs;g{game}.TRwph=TRwph;g{game}.TRwph_b=TRwph_b;
g{game}.TRhp=TRhp;g{game}.TRwtp=TRwtp;g{game}.TRrl=TRrl;
g{game}.TRsur=TRsur;g{game}.TRent=TRent;g{game}.TRBsur=TRBsur;
g{game}.TRsur_h=TRsur_h;g{game}.TRsus=TRsus;
g{game}.TRsusXeFG=TRsusXeFG;g{game}.TReFG=TReFG;g{game}.TRsc=TRsc;
g{game}.TRsc_h=TRsc_h;g{game}.TRaud=TRaud;g{game}.TRlum=TRlum;
g{game}.TRmot=TRmot;g{game}.TRspeech=TRspeech;g{game}.TRcp=TRcp;%TRpercp
g{game}.TRgr=TRgr;g{game}.TRpn=TRpn;g{game}.NumPoss=nanmax(TRpn);
g{game}.TRto=TRto;g{game}.TRft=TRft;g{game}.TRft_up=TRft_up;
g{game}.TRtwopt=TRtwopt;g{game}.TRthreept=TRthreept;g{game}.TRdunk=TRdunk;
g{game}.TRodd=TRodd;g{game}.SeedDiff=seed_diff;
if gs % plot a bunch of variables across TRs
    %top two here go in Fig S1
    lw=3;set(0,'DefaultAxesFontSize',25);
    figure;plot(TRsecs,TRwph,'b','LineWidth',3);xlabel('TR');ylabel({'Win probability,';homestring});%['Win probability,\newline' ]
    gg=gca;gg.XAxis.Visible='off';gg.XTick=[];gg.YLim=[0 100];gg.LineWidth=lw;gg.YTick=[0 25 50 75 100];box off;%
    sn=[root 'pics/TRXWinProbOnlyOneGame-' num2str(game)];print(sn,'-dpdf','-bestfit');
    figure;plot(TRsecs,TRsur,'r','LineWidth',3);xlabel('TR');ylabel({'Surprise';'(win probability change)'});gc=gca;gc.YLim=[0 35];%\newline(win probability change)
    gg=gca;gg.XTick=[];gg.YLim=[0 40];gg.LineWidth=lw;gg.YTick=[0 10 20 30 40];box off;%gg.XAxis.Visible='off';
    sn=[root 'pics/TRXSurpriseOnlyOneGame-' num2str(game)];print(sn,'-dpdf','-bestfit');
    %win prob vs surprise, same plot
    figure;yyaxis left;plot(TRsecs,TRpn);xlabel('TR');ylabel('Possession #');
    yyaxis right;plot(TRcp-1);ylabel('Court position (left vs right)');gc=gca;gc.YLim=[-0.1 1.1];gc.YTick=[0 1];
    sn=[root 'pics/TRXTRpnXTRcp-' num2str(game)];print(sn,'-dpdf','-bestfit');
    figure;yyaxis left;plot(TRsecs,TRwph);xlabel('TR');ylabel('Home win prob');gc=gca;gc.YLim=[0 100];
    yyaxis right;plot(TRsecs,TRsur);ylabel('Surprise (unsigned change in win prob)');gc=gca;gc.YLim=[0 35];
    sn=[root 'pics/TRXSurpriseOneGame-' num2str(game)];print(sn,'-dpdf','-bestfit');
    figure;yyaxis left;plot(TRsecs,TRwph);xlabel('TR');ylabel('Home win prob');gc=gca;gc.YLim=[0 100];
    yyaxis right;plot(TRsecs,TRBsur);ylabel('Bayesian surprise (unsigned change in win prob)');gc=gca;gc.YLim=[0 35];
    sn=[root 'pics/TRXBSurpriseOneGame-' num2str(game)];print(sn,'-dpdf','-bestfit');
    figure;yyaxis left;plot(TRsecs,TRwph);xlabel('TR');ylabel('Win prob home');gc=gca;gc.YLim=[0 100];
    yyaxis right;plot(TRsecs,TRsur_h);ylabel('Surprise (change in home win prob)');
    sn=[root 'pics/TRXSurpriseSignedOneGame-' num2str(game)];print(sn,'-dpdf','-bestfit');
    figure;yyaxis left;plot(TRsecs,TRwph);xlabel('TR');ylabel('Home win prob');gc=gca;gc.YLim=[0 100];
    yyaxis right;plot(TRsecs,TRsus);ylabel('Suspense');
    sn=[root 'pics/TRXSuspenseOneGame-' num2str(game)];print(sn,'-dpdf','-bestfit');
    figure;yyaxis left;plot(TRsecs,TRsus);xlabel('TR');ylabel('Suspense');
    yyaxis right;plot(TRsecs,TRsusXeFG);ylabel('Momentary suspense');
    sn=[root 'pics/TRXMomentarySuspenseOneGame-' num2str(game)];print(sn,'-dpdf','-bestfit');
    figure;yyaxis left;plot(TRsecs,TRsus);xlabel('TR');ylabel('Suspense');
    yyaxis right;plot(TRsecs,TRaud);ylabel('Auditory envelope');
    sn=[root 'pics/TRXSuspenseAudioOneGame-' num2str(game)];print(sn,'-dpdf','-bestfit');
    figure;plot(TRsecs,TRaud);gg=gca;gg.XAxis.Visible='off';gg.XTick=[];gg.YTick=[];ylabel('Auditory envelope');%xlabel('TR');
    sn=[root 'pics/TRXAudioOnlyOneGame-' num2str(game)];print(sn,'-dpdf','-bestfit');
    TRlum2=TRlum+(TRlum/8).*(randn(length(TRlum),1));mx=max([TRlum;TRlum2]);mn=min([TRlum;TRlum2]);
    figure;plot(TRsecs,TRlum);gg=gca;gg.XAxis.Visible='off';gg.YLim=[mn mx];gg.XTick=[];gg.YTick=[];ylabel('Global video luminance');%xlabel('TR');
    sn=[root 'pics/TRXLumOnlyOneGame-' num2str(game)];print(sn,'-dpdf','-bestfit');
    figure;plot(TRsecs,TRlum2);gg=gca;gg.XAxis.Visible='off';gg.YLim=[mn mx];gg.XTick=[];gg.YTick=[];ylabel('Local video luminance');%xlabel('TR');
    sn=[root 'pics/TRXLocLumOnlyOneGame-' num2str(game)];print(sn,'-dpdf','-bestfit');
    TRmot2=TRmot+(TRmot/8).*(randn(length(TRmot),1));mx=max([TRmot;TRmot2]);mn=min([TRmot;TRmot2]);
    figure;plot(TRsecs,TRmot);gg=gca;gg.XAxis.Visible='off';gg.YLim=[mn mx];gg.XTick=[];gg.YTick=[];ylabel('Global video motion');%xlabel('TR');
    sn=[root 'pics/TRXMotOnlyOneGame-' num2str(game)];print(sn,'-dpdf','-bestfit');
    figure;plot(TRsecs,TRmot2);gg=gca;gg.XAxis.Visible='off';gg.YLim=[mn mx];gg.XTick=[];gg.YTick=[];ylabel('Local video motion');%xlabel('TR');
    sn=[root 'pics/TRXLocMotOnlyOneGame-' num2str(game)];print(sn,'-dpdf','-bestfit');
    mx=nanmax([TRspeech;TRspeech]);mn=nanmin([TRspeech;TRspeech]);
    figure;plot(TRsecs,TRspeech,'k');gg=gca;gg.XAxis.Visible='off';gg.YLim=[mn mx];gg.XTick=[];gg.YTick=[];ylabel('Speech');%xlabel('TR');
    sn=[root 'pics/TRXSpeechOnlyOneGame-' num2str(game)];print(sn,'-dpdf','-bestfit');
end
%% create design matrices for both GLM and event-based analysis
if bayes;d_in=[repmat(seed_diff,secmax,1) TRgr TRhp TRsc TRsc_h TRBsur TRsur_h TRsus TRto TRft ...
    TRtwopt TRthreept TRaud TRlum TRmot TRspeech TRcp TRdunk TRodd TRent];
else;d_in=[repmat(seed_diff,secmax,1) TRgr TRhp TRsc TRsc_h TRsur TRsur_h TRsus TRto TRft ...
    TRtwopt TRthreept TRaud TRlum TRmot TRspeech TRcp TRdunk TRodd TRent];
end
%also create positive only, negative only, and unscaled models
d_inUS=d_in;indd=d_in(:,4)>0;d_inUS(indd,4)=1;
indd=d_in(:,5)>0;d_inUS(indd,5)=1;%all scores the same
indd=d_in(:,5)<0;d_inUS(indd,5)=-1;
indd=d_in(:,6)>0;d_inUS(indd,6)=1;
indd=d_in(:,7)>0;d_inUS(indd,7)=1;%all surprises the same
indd=d_in(:,7)<0;d_inUS(indd,7)=-1;
% convolve with canonical HRF
clear d_in_hrf d_inUS_hrf;
for i=1:size(d_in,2);d_in_hrf(:,i)=conv(d_in(:,i),hrf);d_inUS_hrf(:,i)=conv(d_inUS(:,i),hrf);end
if bayes;event_in=[repmat(fmrig,secmax,1) repmat(seed_diff,secmax,1) TRgr TRpn TRhp TRwph TRwtp TRrl TRft_up TRBsur TRsus TRdunk TRodd TRto TRent TRsur_h TRaud TRlum TRmot TRspeech]; 
else;event_in=[repmat(fmrig,secmax,1) repmat(seed_diff,secmax,1) TRgr TRpn TRhp TRwph TRwtp TRrl TRft_up TRsur TRsus TRdunk TRodd TRto TRent TRsur_h TRaud TRlum TRmot TRspeech]; 
end
%sus_mod_in=[repmat(fmrig,secmax,1) repmat(seed_diff,secmax,1) TRgr TRsus];
%% assign to variables that concatenate all games 
%cut off opening TRs b/c of influence from previous screen
event_in=event_in(startcut+1:end,:);d_in_hrf=d_in_hrf(startcut+1:end,:);d_in=d_in(startcut+1:end,:);
d_mat_in=[d_mat_in;d_in;zeros(postgametrs,size(d_in,2))];%no conv for a correlation
d_mat_hrf=[d_mat_hrf;d_in_hrf(1:size(d_in,1)+postgametrs,:)];
d_inUS_hrf=d_inUS_hrf(startcut+1:end,:);d_inUS=d_inUS(startcut+1:end,:);
d_matUS_hrf=[d_matUS_hrf;d_inUS_hrf(1:size(d_inUS,1)+postgametrs,:)];
event_mat=[event_mat;event_in];
%sus_mod_mat=[sus_mod_mat;sus_mod_in];
%calculate total # of game updates (including free throws, etc.)
g_updates=[g_updates;size(gl,1)];
susav=[susav;TRsus TRaud TRlum TRmot];
cumu_poss=[cumu_poss;cumu_poss(end)+max(TRpn)];
TRBCsur=(TRsur-TRent)/2;TRBICsur=(TRsur+TRent)/2;
gamebinsurtotal=[gamebinsurtotal nansum(TRsur)];
gamebindurmean=[gamebindurmean v.Duration];
gamebinsurmean=[gamebinsurmean g{game}.Mean_Surprise];
gamebinsurstdev=[gamebinsurstdev g{game}.StDev_Surprise];
gamebinBCsurtotal=[gamebinBCsurtotal nansum(TRBCsur)];
gamebinBICsurtotal=[gamebinBICsurtotal nansum(TRBICsur)];
gamebinsusmean2=[gamebinsusmean2 g{game}.Mean_Suspense];
gamebinsusmean=[gamebinsusmean nanmean(TRsus)];
gamebinaudmean=[gamebinaudmean nanmean(TRaud)];
gamebinlummean=[gamebinlummean nanmean(TRlum)];
gamebinspeechmean=[gamebinspeechmean nanmean(TRspeech)];
gamebinsusmedian=[gamebinsusmedian nanmedian(TRsus)];
close all;