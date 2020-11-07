%% NCAA
%this is the primary organizing script of the project. it imports game
%data, computes or applies the trained win probability model, loads data
%for each game and outputs most of the major data that will be input for
%behavioral (XSubBehav/Recall), physiological (XSubHR), or fMRI analyses 
%(see Python scripts).
plotdefs2;clear;close all;[base,root]=load_root();
gs=0;% plot graphs or not
bayes=0;%1=use bayesian surprise,0=point estimate of surprise (0 in paper)
%important switch variable - do we want to train the model, apply trained
%model to the viewed (playoff) games, or apply to regular season (for
%example, to train suspense), or something else
mode=1;%0=train model,1=apply trained model,2=apply trained to reg season,3=custom
switch mode;case 0;fact4y=1;%apply 0=KP's model, 1=4-factor model,2=avg
    trainfact4=1;%if we want to train the model
    regplay=1;%1=regular season for corpus data, 2=playoffs
    case 1;fact4y=1;trainfact4=0;regplay=2;%apply
    case 2;fact4y=1;trainfact4=0;regplay=1;%apply to reg
    case 3;fact4y=1;trainfact4=0;regplay=1;%custom
end
%important switch variable - which suspense model do we use? - not in use
%for Antony 2020 paper
sus_model=3;%1=L1,2=L2,3=uncertainty,4=IG,5=KL
qlgf=1;%quickly load game file from .mat
addpath('/Volumes/norman/jantony/plugins/jwacustom') %add custom functions
%% Import data from files
%load suspense data
if regplay==2;ssn=['SSSpace-' num2str(sus_model)];load(ssn);
    SecLeft1=SecLeft;WinProbHome1=WinProbHome;end
%load game files - see loadGameFile to see input
if regplay==1;if qlgf;load lgf;else;loadGameFile;end;else;loadGameFile;end
%% log and plot some playoffs games
% must run 'mode=0' above first to be able to load 'fact4'.
% create variable that groups data by individual game. we can improve the 
% model by finding the expected margin of victory, from which we can calc 
% expected points / sec for home team and adjust ScoreDiff accordingly.
proc=0;
%if fact4y, can plot and compare viewed games to Ken Pomeroy model
if fact4y==1;proc=1;end 
if proc
    load fact4;f4=fact4_sm2R;%load trained model
    HomePoss=zeros(length(TeamPoss),1);%home possession variable
    for i=1:length(TeamPoss);if strcmpi(TeamPoss(i),Home(i));HomePoss(i,1)=1;end;end
    ScoreDiff=HomeScore-VisScore;%difference in score
    currpred_all=zeros(length(WinProbHome),1);
    view_game_ind=1:32;
    %here I enter expected score difference, aligned to each game, in order
    exp_vict_h0=[11 -1 1 1 23 -3 17 2 6 -2 ... 
        2 -1 4 6 7 7 1 9 4 3 ...
        6 5 6 10 24 4 1 18 3 3 24 5];
    exp_vict_h=exp_vict_h0/(40*60);
    for i=1:length(WinProbHome) %all possessions
        id=GameID(i);lookup=find(id==games(view_game_ind)); %find game id
        if ~isempty(lookup);exp_v=exp_vict_h(lookup);else;exp_v=0;end
        if SecLeft(i)==0;currpred_all(i)=HomeWin(i)*100;%if final possession, assign 0/100
        else;[~,SLbin]=min(abs(SecLeft(i)-SecB2));hsdw=1;%find second bin
            if SLbin>size(f4,1);SLbin=size(f4,1);end
            hp=HomePoss(i);if hp==0;hp=2;end %find team in possession
            %adjust expected final score by multipling points / sec difference
            %* seconds remaining, then interpolate between values using
            %hsd/hsdw
            if regplay==2;AdjScoreDiff(i)=ScoreDiff(i)+SecLeft(i)*exp_v;
                diffs=abs(AdjScoreDiff(i)-ScoreDiffB);hsd=find(diffs<1);%ScoreDiff for no adj
                if numel(hsd)==2;hsdw=1-diffs(hsd);%weights
                elseif numel(hsd)==0;[~,hsd]=min(diffs);%if outside range
                end
            else;[~,hsd]=min(abs(ScoreDiff(i)-ScoreDiffB));
            end
            if find(hsd>size(f4,2));hsd=size(f4,2);hsdw=1;end
            currpred_all(i)=sum((f4(SLbin,hsd,hp).*hsdw))*100; %model prediction!
        end
    end
    under5=SecLeft<60*5;%grab scores with under 5 min remaining
    currpred_u5=currpred_all(under5);WinProbHome_u5=WinProbHome(under5);
    find(and(WinProbHome_u5>68,currpred_u5<41))
    find(and(WinProbHome_u5<40,currpred_u5>58))
    %print correspondence w/ KP model in terminal
    [r,p]=corrcoef(currpred_u5,WinProbHome_u5,'rows','complete');r(2) 
    if gs %plot comparison between studies for tournament games, Fig 1C, bottom
        set(0,'DefaultAxesFontSize',25);
        figure;scatter(currpred_u5,WinProbHome_u5);gg=gca;gg.LineWidth=2;
        xlabel('4-factor model win probability');ylabel('Expert win probability');
        sn=[root 'pics/WinProbModelCompAll-' num2str(regplay) '-' num2str(SecSteps2)];print(sn,'-dpdf','-bestfit');
    end
    %re-assign from KP model to 4-factor model prediction! henceforth,
    %everything will be 4-factor model if proc>0
    WinProbHome=currpred_all;
end

%load up all game info
GameTime=20*60-SecLeft;%flip seconds left to time into game
EndTimes=[(20-5)*60-40 20*60];%account for possession starting before 5:00 left
g=cell(length(games),1);g{length(games),1}=[];
HomePoss=zeros(length(TeamPoss),1);
for i=1:length(TeamPoss);if strcmpi(TeamPoss(i),Home(i));HomePoss(i,1)=1;end;end
for i=1:length(games)
    rows=find(GameID==games(i));
    %re-assign some variables
    g{i}.Home=Home(rows);g{i}.HomeScore=HomeScore(rows);
    g{i}.GameTime=GameTime(rows);g{i}.Visitor=Visitor(rows);
    g{i}.VisScore=VisScore(rows);g{i}.WinProbHome=WinProbHome(rows);
    g{i}.ScoreDiff=HomeScore(rows)-VisScore(rows);
    g{i}.ScoreDiffAbs=abs(HomeScore(rows)-VisScore(rows));
    g{i}.Entropy=50-abs(WinProbHome(rows)-50);
    g{i}.TeamPoss=TeamPoss(rows);g{i}.HomePoss=HomePoss(rows);
    if regplay==2
        g{i}.HomeSeed=seeds(i);
        %fix Ken Pom Purdue messup
        if strcmpi(g{i}.Home{1,1},'Purdue')
            tp1=[];
            fill=g{i}.VisScore;fill2=g{i}.HomeScore;
            g{i}.HomeScore=fill;g{i}.VisScore=fill2;
            g{i}.ScoreDiff=-g{i}.ScoreDiff;
            g{i}.WinProbHome=100-g{i}.WinProbHome;
            fill=g{i}.Visitor{1};fill2=g{i}.Home{1};
            for j=1:length(g{i}.Home)
                g{i}.Home{j}=fill;g{i}.Visitor{j}=fill2;
                %if strcmpi(g{i}.TeamPoss(j),'Purdue')
                %    g{i}.TeamPoss(j)={'Saint Mary''s'};
                %elseif strcmpi(g{i}.TeamPoss(j),'Saint Mary''s')
                %    g{i}.TeamPoss(j)={'Purdue'};
                %end
            end
        end
    end 
end

%% create win prob metric and compare to KP - Fig 1B,C
%for every win prob rating, how often does that actually lead to a win
if regplay==1 %only use corpus data
    if trainfact4 %only if we want to train
        ScoreDiff=HomeScore-VisScore;
        %train on the first half of games in corpus and test on second half
        %against Ken Pom
        xsd=20;%range of score differences from - to + this number
        nsd=-xsd;
        ScoreDiffSteps=1+xsd-nsd;%
        ScoreDiffB=linspace(nsd,xsd,ScoreDiffSteps); %create win prob space
        SecSteps2=200;%number of different seconds bins for the 2nd half of games
        SecB2=linspace(20*60,0,SecSteps2+1);%second space
        fact4i=nan(SecSteps2,ScoreDiffSteps,2);clear fact4_sm;
        fact4full=cell(SecSteps2,ScoreDiffSteps,2);
        tr_half=0;%0=train all,1=train half, using ID of game ~ halfway
        if tr_half;gthr=1:169069;else;gthr=1:length(SecLeft);end
        nn=[SecLeft(gthr) ScoreDiff(gthr) HomePoss(gthr) HomeWin(gthr)];
        for i=1:SecSteps2 %all second bins
            for j=1:ScoreDiffSteps %all score bins
                for k=1:2 %possession (home / visitor)
                    kk=mod(k,2);
                    %find all games w/ game state and calculate how often
                    %they won
                    q=find(and(and(and(SecLeft(gthr)<=SecB2(i),SecLeft(gthr)>SecB2(i+1)),...
                        ScoreDiff(gthr)==ScoreDiffB(j)),HomePoss(gthr)==kk));
                    fact4i(i,j,k)=nanmean(HomeWin(q));
                    fact4full{i,j,k}.z=HomeWin(q);
                end
            end
        end
        fact4=fact4i;
        %add up to 0.5% of noise on 0/100 to allow for the rarest of events 
        %outside scope of training games
        fact4(fact4==0)=fact4(fact4==0)+rand(length(find(fact4==0)),1)/200;
        fact4(fact4==1)=fact4(fact4==1)-rand(length(find(fact4==1)),1)/200;
        smf=2;%smoothing factor
        fact4b=nan(SecSteps2+smf*2,ScoreDiffSteps+smf*2,2);
        fact4b(smf+1:end-smf,smf+1:end-smf,:)=fact4;fact4_sm=fact4;fact4_smR=fact4;
        if smf>0
            for i=1+smf:SecSteps2+smf
                for j=1+smf:ScoreDiffSteps+smf
                    for k=1:2 %possession
                        mat=fact4b(i-smf:i+smf,j-smf:j+smf,k);
                        leng=sum(~isnan(mat),'all');
                        newmat=mat(:);
                        fact4_sm(i-smf,j-smf,k)=nanmean(newmat,'all');
                        %stop smoothing in last couple second bins where 
                        %it is very well-defined
                        if i<SecSteps2+smf-2;fact4_smR(i-smf,j-smf,k)=nanmean(newmat,'all');
                        else;newmat=[mat(:);repmat(fact4(i-smf,j-smf,k),ceil(leng*3/4),1)];
                            fact4_smR(i-smf,j-smf,k)=nanmean(newmat,'all');
                        end
                    end
                end
            end
        else;fact4_sm(:,:,1)=smoothn(fact4(:,:,1),'robust');fact4_sm(:,:,2)=smoothn(fact4(:,:,2),'robust');
            fact4_sm(fact4_sm>1)=1;fact4_sm(fact4_sm<0)=0;%limit to 0-1 after smoothing distortion
        end
        fact4_sm2=fact4_sm;fact4_sm2R=fact4_smR;
        if tr_half==0 %plot and save - Fig 1B
            %plot model state space!
            xx=find(SecB2<5*60);xx=xx(1:end-1);tn='';
            set(0,'DefaultAxesFontSize',25);
            figure;set(gcf,'position',[0 0 2000 1200]);%300-fact4_sm
            pimc(SecB2(xx),ScoreDiffB,100*squeeze(fact4_smR(xx,:,1))',tn,'Win probability');%SecB2(1:end-1)
            gg=gca;gg.LineWidth=2;sn=[root 'pics/WinProbSpaceHome-' num2str(smf) '-' num2str(SecSteps2)];print(sn,'-dpdf','-bestfit');
            pimc(SecB2(xx),ScoreDiffB,100*squeeze(fact4_smR(xx,:,2))',tn,'Win probability');
            gg=gca;gg.LineWidth=2;sn=[root 'pics/WinProbSpaceVis-' num2str(smf) '-' num2str(SecSteps2)];print(sn,'-dpdf','-bestfit');
            pimc(SecB2(xx),ScoreDiffB,100*(squeeze(fact4_smR(xx,:,1))-squeeze(fact4_smR(xx,:,2)))',...
                tn,'Win probability difference');
            gg=gca;gg.LineWidth=2;sn=[root 'pics/WinProbSpaceDiff-' num2str(smf) '-' num2str(SecSteps2)];print(sn,'-dpdf','-bestfit');
            %save once, not each time, because there's some randomization
%            save fact4full fact4full SecSteps2;
%            save fact4 fact4 fact4_sm2 fact4_sm2R SecB2 ScoreDiffB SecSteps2;
        end
        
        %if training on half (above) and testing on other half to validate
        if tr_half
            gthr=gthr(2)+1:length(SecLeft);%possessions to validate on
            clear actualO_sm KPpred_sm
            KPpred=nan(SecSteps2,ScoreDiffSteps,2);
            actualO=KPpred;count=KPpred;
            for i=1:SecSteps2
                for j=1:ScoreDiffSteps
                    for k=1:2 %possession
                        kk=mod(k,2);
                        q=find(and(and(and(SecLeft(gthr)<=SecB2(i),SecLeft(gthr)>SecB2(i+1)),...
                            ScoreDiff(gthr)==ScoreDiffB(j)),HomePoss(gthr)==kk));
                        KPpred(i,j,k)=mean(WinProbHome(gthr(q))/100);
                        actualO(i,j,k)=mean(HomeWin(gthr(q)));
                        count(i,j,k)=length(q);
                    end
                end
            end
            
            actualOb=nan(SecSteps2+smf*2,ScoreDiffSteps+smf*2,2);KPpredb=actualOb;
            actualOb(smf+1:end-smf,smf+1:end-smf,:)=actualO;
            actualO_sm=fact4;actualO_smR=fact4;
            KPpredb(smf+1:end-smf,smf+1:end-smf,:)=KPpred;
            KPpred_sm=fact4;KPpred_smR=fact4;
            if smf>0
                for i=1+smf:SecSteps2+smf
                    for j=1+smf:ScoreDiffSteps+smf
                        for k=1:2 %possession
                            mat=actualOb(i-smf:i+smf,j-smf:j+smf,k);
                            leng=sum(~isnan(mat),'all');newmat=mat(:);
                            actualO_sm(i-smf,j-smf,k)=nanmean(newmat,'all');
                            matKP=KPpredb(i-smf:i+smf,j-smf:j+smf,k);
                            lengKP=sum(~isnan(matKP),'all');newmatKP=matKP(:);
                            KPpred_sm(i-smf,j-smf,k)=nanmean(newmatKP,'all');
                            %stop smoothing in last couple second bins where 
                            %it is very well-defined
                            if i<SecSteps2+smf-2;actualO_smR(i-smf,j-smf,k)=nanmean(newmat,'all');
                                KPpred_smR(i-smf,j-smf,k)=nanmean(newmatKP,'all');
                            else;newmat=[mat(:);repmat(actualO(i-smf,j-smf,k),ceil(leng*3/4),1)];
                                actualO_smR(i-smf,j-smf,k)=nanmean(newmat,'all');
                                newmatKP=[matKP(:);repmat(KPpred(i-smf,j-smf,k),ceil(lengKP*3/4),1)];
                                KPpred_smR(i-smf,j-smf,k)=nanmean(newmatKP,'all');
                            end
                        end
                    end
                end
                %if we choose the stop smoothing route, switch over
                actualO_sm=actualO_smR;KPpred_sm=KPpred_smR;
            else %if we didn't want to smooth the above way and just smooth all
                actualO_sm(:,:,1)=smoothn(actualO(:,:,1),'robust');actualO_sm(:,:,2)=smoothn(actualO(:,:,2),'robust');
                KPpred_sm(:,:,1)=smoothn(KPpred(:,:,1),'robust');KPpred_sm(:,:,2)=smoothn(KPpred(:,:,2),'robust');
                actualO_sm(actualO_sm>1)=1;actualO_sm(actualO_sm<0)=0;%limit to 0-1 after smoothing distortion
                KPpred_sm(KPpred_sm>1)=1;KPpred_sm(KPpred_sm<0)=0;
            end
            %find errors
            KPerr_sm=abs(KPpred_sm-actualO_sm);KPerr=abs(KPpred-actualO);
            fact4err_sm=abs(fact4_sm-actualO_sm);fact4err=abs(fact4-actualO);
            fact4err_smR=abs(fact4_smR-actualO);
            %mean errors and stds for both models, + comparisons
            nanmean(KPerr_sm(:))
            nanmean(fact4err_sm(:))
            nanstd(fact4err_sm(:))
            nanstd(fact4err(:))
            astats(KPerr(:),fact4err(:))
            astats(KPerr_sm(:),fact4err_sm(:))
            %correlations between model predictions + actual
            [r,p]=corrcoef(fact4_smR(:),actualO_sm(:),'rows','complete');r(2)
            [r,p]=corrcoef(KPpred_sm(:),actualO_sm(:),'rows','complete');r(2)
            [r,p]=corrcoef(fact4_smR(:),KPpred_sm(:),'rows','complete');r(2)
            %plot comparison between studies - Fig 1C, top
            set(0,'DefaultAxesFontSize',25);
            figure;scatter(fact4_smR(:)*100,KPpred_sm(:)*100);gg=gca;gg.LineWidth=2;
            xlabel('4-factor model win probability');ylabel('Expert win probability');
            sn=[root 'pics/fact4vsKPModelComp-' num2str(smf) '-' num2str(SecSteps2)];print(sn,'-dpdf','-bestfit');
            %histogram of errors in our / KP model
            figure;histogram(KPerr(:),'BinWidth',0.005);hold on;
            histogram(fact4err(:),'BinWidth',0.005);%histogram(PMerr_sm(:),'BinWidth',0.005);
            gg=gca;gg.XLim=[0 0.2];
            if gs;tn='Prediction difference, Ken Pom - actual';
                figure;subplot(211);pimc(SecB2(1:end-1),ScoreDiffB,mean(KPerr_sm,3)',tn,'Error');tn=[tn '- smoothed'];
                tn='Prediction difference, 3-factor model - actual';
                subplot(212);pimc(SecB2(1:end-1),ScoreDiffB,mean(fact4err_smR,3)',tn,'Error');
                sn=[root 'pics/ModelErrors-' num2str(smf) '-' num2str(SecSteps2)];print(sn,'-dpdf','-bestfit');
            end
        end
    end
end

%% calculate suspense using a variety of metrics - not in use
%only train on regular season corpus
%we can later apply the 'suspense' variable to playoff games
WinProbHomeDiff=diff(WinProbHome);%difference since last probability
%4_15_19 note - keep this so a swing that crosses the 50% prob mark is 
%registered as such (and not reduced by having swung to other side), so a
%swing from 55 to 40 = 15, not 5.
SecSteps=200;%time bin steps %300 on 8/9/20, %200 on 9/2/20
SecB=linspace(20*60,0,SecSteps+1);twin=(SecB(1)-SecB(2))/2;
WinSteps=10;%win probability steps %10
WinB=linspace(50,100,WinSteps+1);%win probability space
WinStepsFull=WinSteps*2;WinBFull=linspace(0,100,WinStepsFull+1);
pwin=(WinBFull(2)-WinBFull(1))/2;%full space
WinProbWinning=abs(WinProbHome-50)+50;%win prob for winning team
%gs=1;
if regplay==1 %only train on regular season games
    suspense=nan(SecSteps,WinSteps);
    suspensedist=cell(SecSteps,WinStepsFull);%pre-allocate
    % for explanatory figure, use i = 76, j = 1
    for i=1:SecSteps %all seconds bins
        for j=1:WinSteps %all win probability bins
            %find points that fit 1) time left and 2) win probability
            q=find(and(and(and(SecLeft<=SecB(i),SecLeft>SecB(i+1)),...
                WinProbWinning>=WinB(j)),WinProbWinning<WinB(j+1)));
            %for explanatory figures only
            if i==SecSteps-15&&j==WinSteps;spread1=WinProbHomeDiff(q);
            std1=std(WinProbHomeDiff(q));end
            if i==SecSteps-20&&j==1;spread2=WinProbHomeDiff(q);
            std2=std(WinProbHomeDiff(q));end
            if ~isempty(q) %if anything is in that time / win prob bin
                q2=q>1;
                q=q(q2);clear q2;%clear last trial
                if length(q)>1 %need at least 2 values to get stdev
                    %'diff', but don't need the -1 b/c talking about upcoming state
                    switch sus_model;case 1;suspense(i,j)=std(WinProbHomeDiff(q)); %L1
                    case 2;suspense(i,j)=var(WinProbHomeDiff(q)); %find suspense, L2
                    case 3;suspense(i,j)=100-mean(WinProbWinning(q));%diff from 100%
                    case 4;%curr_inf=WinProbWinning(q);up_inf=WinProbWinning(q+1);
                        %suspense(i,j)=mean((curr_inf*log2(1/curr_inf))-(up_inf*log2(1/up_inf)));%? Nelson, 2005
                        curr_inf=mean(WinProbWinning(q))-50;up_inf=mean(WinProbWinning(q+1))-50;
                        suspense(i,j)=(curr_inf*log2(1/curr_inf))-(up_inf*log2(1/up_inf));%? Nelson, 2005
                    case 5;curr=WinProbWinning(q);upc=WinProbWinning(q+1);
                        KLdiv=curr.*log(curr./upc);
                        KLdiv=KLdiv(~isinf(KLdiv));KLdiv=KLdiv(~isnan(KLdiv));
                        suspense(i,j)=nanmean(abs(KLdiv));%suspense(i,j)=abs(nanmean(KLdiv));
                    end
                end
            end
        end
    end
    
    for i=1:SecSteps %grab full distribution within each - not using atm
        for j=1:WinStepsFull
            %find points that fit time left and win probability
            q=find(and(and(and(SecLeft<=SecB(i),SecLeft>SecB(i+1)),...
                WinProbHome>=WinBFull(j)),WinProbHome<WinBFull(j+1)));
            if ~isempty(q) %if anything in that time / win prob bin
                q2=q>1;%q2=and(q<length(DateOfGame),q>1); 9_14_19
                q=q(q2);clear q2;%clear last trial
                if length(q)>1 %need at least 2 to get stdev
                    suspensedist{i,j}.d=WinProbHome(q);
                end
            end
        end
    end
    %explanatory figures (see special demo cases above)
    cats={'Current win probability (zeroed)','Upcoming win probability'};
    if gs;sp=30;figure;plot([zeros(length(spread1),1) spread1]','b');ylabel('Win probability change');
        ggg=gca;ggg.XTick=1:length(cats);ggg.XTickLabel=cats;ggg.YLim=[-sp sp];title(std(spread1));
        sn=[root 'pics/LowSuspenseSpread'];print(sn,'-dpdf','-bestfit');
        figure;plot([zeros(length(spread2),1) spread2]','b');ylabel('Win probability change');
        ggg=gca;ggg.XTick=1:length(cats);ggg.XTickLabel=cats;ggg.YLim=[-sp sp];title(std(spread2));
        sn=[root 'pics/HighSuspenseSpread'];print(sn,'-dpdf','-bestfit');end
    
    %for plotting each suspense model
    switch sus_model;case 1;tn='L1 as a function of time and win prob';
    case 2;tn='L2 as a function of time and win prob';
    case 3;tn='Uncertainty as a function of time and win prob';
    case 4;tn='IG as a function of time and win prob';
    case 5;tn='KL as a function of time and win prob';
    end
    
    suspense_smooth=smoothn(suspense,2);%apply smoothing
    %smooth by row for uncertainty to avoid too much blurring
    if sus_model==3;for i=1:size(suspense,2);suspense_smooth(:,i)=smoothn(suspense(:,i),2);end;end
    if gs;figure;subplot(211);pimc(SecB(1:end-1),WinB,suspense',tn,'Suspense');tn=[tn '- smoothed'];
        subplot(212);pimc(SecB(1:end-1),WinB,suspense_smooth',tn,'Suspense');
        sn=[root 'pics/SuspenseSpace-' num2str(sus_model)];print(sn,'-dpdf','-bestfit');end
end

%% plot suspense against one game
if regplay==1;wg=8;elseif regplay==2;wg=5;end %pick game manually
if gs;figure;
    tn='Suspense as a function of time and win prob';tn=[tn '- smoothed'];
    pimc(SecB(1:end-1),WinB,suspense_smooth',tn,'Suspense');hold on;
    for i=1:length(wg)
        igt=max(g{wg(i)}.GameTime)-g{wg(i)}.GameTime;%inverse game time
        plot(igt,abs(g{wg(i)}.WinProbHome-50)+50,'-w','LineWidth',0.5);
    end
    sn=[root 'pics/SuspenseSpaceWithGames-' num2str(regplay) '-' num2str(sus_model)];
    print(sn,'-dpdf','-bestfit');
end
close all;
%% assign latent variables for individual games
%some of this is jargon for finding Bayesian surprise, which we are not
%using atm. Other aspects grab cumulative amount of surprise within each
%game, etc.
ind_sur=[];ind_sur_b=[];ind_sur_t=[];ind_fsur=[];ind_sus=[];ind_prob=[];
ind_prob_win=[];ind_ids=[];sum_sur=zeros(length(games),1);sum_sus=sum_sur;
init_sur=sum_sur;ind_poss=[];currdist=0;prevdist=0;init_sus=sum_sur;
mean_sus=sum_sur;mean_sur=sum_sur;mean_sur_b=sum_sur;median_sus=sum_sur;
median_sur=sum_sur;val5m=zeros(length(games),2);hsp=linspace(0,100,21);rf=5;
for i=1:length(games)
    endrows=find(and(g{i}.GameTime>EndTimes(1),g{i}.GameTime<=EndTimes(2)));%9/2/20
    %endrows=find(and(g{i}.GameTime>=EndTimes(1),g{i}.GameTime<=EndTimes(2)));
    endrows(2:length(endrows)+1)=endrows;endrows(1)=endrows(2)-1;%add preceding time point for start of clip
    igt=max(g{i}.GameTime(endrows))-g{i}.GameTime(endrows);%inverse game time
    prob_end=g{i}.WinProbHome(endrows);%win prob for end of game
    sds=g{i}.ScoreDiff(endrows);
    for ii=endrows(1):endrows(end);idx=ii-endrows(1)+1;
        if strcmpi(g{i}.TeamPoss(ii),g{i}.Home(ii));hp(idx,1)=1;else;hp(idx,1)=2;end;end
    prob_end_win=abs(g{i}.WinProbHome(endrows)-50)+50;
    i_sus=zeros(length(igt),1);i_sur=i_sus;i_tbin=i_sus;
    i_sur_h=i_sus;i_prob=i_sus;i_prob_win=i_sus;f_sur=i_sus;i_sur_b=i_sus;
    i_sur_t=i_sus;
    for ii=1:length(igt)
        tbin=find(igt(ii)<SecB);tbin=tbin(end);%find suspense time bin
        pbin=find(prob_end_win(ii)>=WinB(1:end-1));pbin=pbin(end);%find suspense win prob bin
        i_tbin(ii)=tbin;%instantaneous time
        i_prob(ii)=prob_end(ii);%instantaneous win prob
        i_prob_win(ii)=prob_end_win(ii);%instantaneous win prob for winning team
        if ii<length(igt)
            i_sus(ii)=suspense_smooth(tbin,pbin);%instantaneous suspense
            f_sur(ii)=abs(prob_end(ii+1)-prob_end(ii));%NEXT (future) surprise
        end
        if ii>1 
            i_sur(ii)=abs(prob_end(ii)-prob_end(ii-1));%instantaneous surprise
            %surprise positively signed for the home team
            i_sur_h(ii)=prob_end(ii)-prob_end(ii-1);
            
            %bayesian surprise stuff for the next ~30 lines - not in use
            tbinprev=find(igt(ii-1)<SecB);tbinprev=tbinprev(end);%find suspense time bin
            if regplay==1;secl=SecLeft;winph=WinProbHome;else;secl=SecLeft1;winph=WinProbHome1;end
            pbinprev=find(prob_end(ii-1)>=WinBFull(1:end-1));pbinprev=pbinprev(end);%find suspense win prob bin
            q=find(and(and(and(secl<=igt(ii-1)+twin,secl>igt(ii-1)-twin),...
                winph>=prob_end(ii-1)-pwin),winph<prob_end(ii-1)+pwin));
            if ~isempty(q);if q(end)==length(winph);q=q(1:end-1);end
            tidx=secl(q+1)<=igt(ii);%same game
            q=q(tidx);q2=q>1;q=q(q2);clear q2;%clear last trial
                if length(q)>1;prevdist=winph(q+1);end;end %add +1 6/4/2020
            pc=numel(q);
            q=find(and(and(and(secl<=igt(ii)+twin,secl>igt(ii)-twin),...
                winph>=prob_end(ii)-pwin),winph<prob_end(ii)+pwin));
            if ~isempty(q);if q(end)==length(winph);q=q(1:end-1);end
            tidx=secl(q+1)<=igt(ii);q=q(tidx);q2=q>1;q=q(q2);clear q2;%clear last trial
                if length(q)>1
                    if ii==length(igt);currdist=repmat(prob_end(ii),pc,1);
                    else;currdist=winph(q+1);end;end;end 
            nc=numel(q);
            currdist=currdist+randn(numel(currdist),1)*rf;%add random noise
            currdist(currdist>100)=100;currdist(currdist<0)=0;%curb >100,<0
            prevdist=prevdist+randn(numel(prevdist),1)*rf;
            prevdist(prevdist>100)=100;prevdist(prevdist<0)=0;
            currdist=histcounts(currdist,hsp);prevdist=histcounts(prevdist,hsp);
            currdist(currdist==0)=1/nc;prevdist(prevdist==0)=1/pc;
            currdist=currdist./(sum(currdist));prevdist=prevdist./(sum(prevdist));
            KLdiv=prevdist.*log(prevdist./currdist);
            KLdiv=KLdiv(~isinf(KLdiv));KLdiv=KLdiv(~isnan(KLdiv));
            if and(nc>3,pc>3);i_sur_b(ii)=abs(nanmean(KLdiv));
            else;i_sur_b(ii)=nan;
            end
        end
    end
    
    %cumulative variables
    ind_sur=[ind_sur;i_sur];ind_sus=[ind_sus;i_sus];
    ind_fsur=[ind_fsur;f_sur];
    ind_prob=[ind_prob;i_prob];ind_prob_win=[ind_prob_win;i_prob_win];
    ind_ids=[ind_ids;repmat(i,length(i_sur),1)];
    poss(1:length(i_sur),1)=1:length(i_sur);ind_poss=[ind_poss;poss];
    sum_sur(i)=sum(i_sur);sum_sus(i)=sum(i_sus);
    mean_sur(i)=mean(i_sur);mean_sus(i)=mean(i_sus);
    median_sur(i)=median(i_sur);median_sus(i)=median(i_sus);
    init_sur(i)=i_sur(1);init_sus(i)=i_sus(1);
    g{i}.i_prob=i_prob;g{i}.i_sur=i_sur;g{i}.i_sur_h=i_sur_h;
    g{i}.i_sus=i_sus;g{i}.f_sur=f_sur;
    %accumulated variables across the game
    g{i}.Mean_ScoreDiff=mean(abs(diff(g{i}.ScoreDiff(endrows))));
    g{i}.Mean_Surprise=mean(i_sur);
    g{i}.StDev_Surprise=std(i_sur);
    g{i}.Mean_Surprise_h=mean(i_sur_h);
    g{i}.Mean_Suspense=mean(i_sus);
    g{i}.Median_Suspense=median(i_sus);
    %for value test, grab initial probability and mean surprise
    %grab expected surprise values with 5 minutes remaining
    val5m(i,1)=abs(i_prob(1)-50)+50;val5m(i,2)=mean(i_sur);%eventually read in for future surprise test
    mean_sur_b(i)=mean(i_sur_b);r=corrcoef(i_sur,i_sur_b);rs(i)=r(2);
    ind_sur_b=[ind_sur_b;i_sur_b];%ind_sur_t=[ind_sur_t;i_sur_t];
    g{i}.i_sur_b=i_sur_b;g{i}.Mean_Surprise_b=mean(i_sur_b);
    if g{i}.HomeScore(end)>g{i}.VisScore(end);g{i}.HomeWin=1;else;g{i}.HomeWin=0;end
end
%save suspense space variable
if regplay==1;fn=['SSSpace-' num2str(sus_model) '.mat'];
    save(fn,'SecB','WinB','suspense_smooth','SecLeft','WinProbHome','val5m');end
%add pca for surprise & suspense - not in use
a=[mean_sur mean_sus];[coeff,score,~,~,var_exp]=pca(a);
if gs;figure;biplot(coeff,'scores',score,'varlabels',{'Surprise','Suspense'});
    sn=[root 'pics/PCAMeanSusMeanSur-' num2str(regplay)];print(sn,'-dpdf','-bestfit');
    figure;scatter(val5m(:,1),val5m(:,2),3);
    xlabel('Win probability,\newlinewinning team');ylabel('Future surprise per possession');
    gg=gca;gg.YTick=[0 4 8 12 16 20];gg.XTick=[50 60 70 80 90 100];
    sn=[root 'pics/FutureSurprise-' num2str(regplay)];print(sn,'-dpdf','-bestfit');
    [r,p]=corrcoef(val5m(:,1),val5m(:,2));end
for i=1:length(games);g{i}.PCs=score(i,:);end
%% plots
if gs;indd=and(~isnan(ind_sur_b),~isinf(ind_sur_b));
    %plot surprise vs Bayesian surprise
    x=ind_sur(indd);y=ind_sur_b(indd);
    figure;scatter(x,y);h=lsline;r=corrcoef(x,y,'rows','complete');r=r(2);title(['r= ' num2str(r)]);
    xlabel('All instant surprise');ylabel('All instant Bayesian surprise');
    sn=[root 'pics/SurpriseByBSurpriseAllMoments-' num2str(regplay)];print(sn,'-dpdf','-bestfit');
    %plot surprise vs suspense
    indd=ind_sus>0;x=ind_sur(indd);y=ind_sus(indd);
    figure;scatter(x,y);h=lsline;r=corrcoef(x,y);r=r(2);title(['r= ' num2str(r)]);
    xlabel('All instant surprise');ylabel('All instant suspense');
    sn=[root 'pics/SuspenseBySurpriseAllMoments-' num2str(regplay)];print(sn,'-dpdf','-bestfit');
    %plot future surprise (actual next outcome) vs suspense
    x=ind_fsur(indd);y=ind_sus(indd);
    figure;scatter(x,y);h=lsline;r=corrcoef(x,y);r=r(2);title(['r= ' num2str(r)]);
    xlabel('All future surprise');ylabel('All instant suspense');
    sn=[root 'pics/SuspenseByFutureSurpriseAllMoments-' num2str(regplay)];print(sn,'-dpdf','-bestfit');
    %plot future surprise (actual next outcome) vs current surprise
    x=ind_sur;y=ind_fsur;
    figure;scatter(x,y);h=lsline;r=corrcoef(x,y);r=r(2);title(['r= ' num2str(r)]);
    xlabel('All instant surprise');ylabel('All future surprise');
    sn=[root 'pics/SurpriseByFutureSurpriseAllMoments-' num2str(regplay)];print(sn,'-dpdf','-bestfit');
    %plot suspense vs win prob difference for that possession
    indd=ind_sus>0;x=ind_sus(indd);y=abs(ind_prob-50);y=y(indd);
    figure;scatter(x,y);h=lsline;r=corrcoef(x,y);r=r(2);title(['r= ' num2str(r)]);
    xlabel('All instant suspense');ylabel('All win probability difference');
    sn=[root 'pics/SuspenseByWinProbAllMoments-' num2str(regplay)];print(sn,'-dpdf','-bestfit');
    %plot mean surprise across game vs initial suspense at game outset
    x=mean_sur;y=init_sus;
    figure;scatter(x,y);h=lsline;r=corrcoef(x,y);r=r(2);title(['r= ' num2str(r)]);
    xlabel('Mean surprise');ylabel('Initial suspense');
    sn=[root 'pics/InitSuspenseBySurpriseAllGames-' num2str(regplay)];print(sn,'-dpdf','-bestfit');
    %"" for games above a modest level of initial suspense
    x=mean_sur;y=init_sus;
    if regplay==1;sel=y>5;elseif regplay==2;sel=y>3;end
    figure;scatter(x(sel),y(sel));h=lsline;r=corrcoef(x(sel),y(sel));r=r(2);title(['r= ' num2str(r)]);
    xlabel('Mean surprise');ylabel('Initial suspense');
    sn=[root 'pics/InitSuspenseBySurpriseCloseGames-' num2str(regplay)];print(sn,'-dpdf','-bestfit');
    %plot sum across game, surprise vs suspense
    x=sum_sur;y=sum_sus;
    figure;scatter(x,y);h=lsline;r=corrcoef(x,y);r=r(2);title(['r= ' num2str(r)]);
    xlabel('Sum surprise');ylabel('Sum suspense'); hold on;
    a=[1:length(games)]';b=num2str(a);c=cellstr(b);dx=0.5;dy=dx;text(x+dx,y+dy,c,'Fontsize',15);
    sn=[root 'pics/SuspenseBySurpriseAllGames-' num2str(regplay)];print(sn,'-dpdf','-bestfit');
    %plot mean ""
    x=mean_sur;y=mean_sus;
    figure;scatter(x,y);h=lsline;r=corrcoef(x,y);r=r(2);title(['r= ' num2str(r)]);
    xlabel('Mean surprise');ylabel('Mean suspense'); hold on;
    a=[1:length(games)]';b=num2str(a);c=cellstr(b);dx=0.2;dy=dx;text(x+dx,y+dy,c,'Fontsize',15);
    sn=[root 'pics/SuspenseBySurpriseAllGamesMean-' num2str(regplay)];print(sn,'-dpdf','-bestfit');
end

%% tournament games for scanner
%code below will produce most outputs we use for the analyses in the paper!
if regplay==2
    %games manually selected out of 32 available (in order in which they
    %were stored, see below for their game labels, etc.)
    selc=[2 3 5 6 13 15 18 19 21];
    ind_sur2=[];ind_sur_b2=[];ind_fsur2=[];ind_sus2=[];ind_prob2=[];
    for i=1:length(selc) %assign variables for each selected game
        ids(i,1)=g{selc(i)}.Home(1);ids(i,2)=g{selc(i)}.Visitor(1);
        ind_sur2=[ind_sur2;g{selc(i)}.i_sur];
        ind_sur_b2=[ind_sur_b2;g{selc(i)}.i_sur_b];
        ind_fsur2=[ind_fsur2;g{selc(i)}.f_sur];
        ind_sus2=[ind_sus2;g{selc(i)}.i_sus];
        ind_prob2=[ind_prob2;g{selc(i)}.i_prob];seeds2(i,1)=g{selc(i)}.HomeSeed;
    end
    if gs %plot relationships for selected games
        figure;scatter(mean_sur(selc),mean_sus(selc));h=lsline;
        r=corrcoef(mean_sur(selc),mean_sus(selc));r=r(2);title(['r= ' num2str(r)]);
        xlabel('Mean surprise');ylabel('Mean suspense');
        a=[1:length(selc)]';b=num2str(a);c=cellstr(b);dx=0.5;dy=dx;text(mean_sur(selc)+dx,mean_sus(selc)+dy,c,'Fontsize',15);
        sn=[root 'pics/SuspenseBySurpriseScannerGames-' num2str(regplay)];print(sn,'-dpdf','-bestfit');
        %plot surprise vs suspense, all possessions
        figure;scatter(ind_sur2,ind_sus2);h=lsline;
        r=corrcoef(ind_sur2,ind_sus2);r=r(2);title(['r= ' num2str(r)]);
        xlabel('All instant surprise');ylabel('All instant suspense');
        sn=[root 'pics/SuspenseBySurpriseScannerMoments-' num2str(regplay)];print(sn,'-dpdf','-bestfit');
        %plot # instances of win probabilities, binned by 10% - Fig S1B
        set(0,'DefaultAxesFontSize',25);figure;h=histogram(ind_prob2,10);h.LineWidth=2;
        ylabel('# of possessions');gg=gca;gg.LineWidth=2;
        xlabel('Home team win probability');box off;
        sn=[root 'pics/ScannerWinProbabilityDistribution-' num2str(regplay)];print(sn,'-dpdf','-bestfit');
    end
    %% Games for prediction test
    selc=[1 8 12 14 22];%games manually selected for prediction test
    ind_sur3=[];ind_sus3=[];ind_prob3=[];
    for i=1:length(selc) %assign variables
        ids2(i,1)=g{selc(i)}.Home(1);ids2(i,2)=g{selc(i)}.Visitor(1);
        ind_sur3=[ind_sur3;g{selc(i)}.i_sur];ind_sus3=[ind_sus3;g{selc(i)}.i_sus];
        ind_prob3=[ind_prob3;g{selc(i)}.i_prob];
    end
    if gs %plot # instances of win probabilities, binned by 10% - Fig S1C
        set(0,'DefaultAxesFontSize',25);
        figure;h=histogram(ind_prob3,10);h.LineWidth=2;ylabel('# of possessions');
        xlabel('Home team win probability');gg=gca;gg.LineWidth=2;box off;
        sn=[root 'pics/PredictionTestWinProbabilityDistribution-' num2str(regplay)];print(sn,'-dpdf','-bestfit');
    end
    %% run game logs for prediction test
    %this will apply our 4-factor model to the situations in these
    %prediction test to find true win probabilities
    game=selc(1);homestring='Duke';loadPredictionTest;
    game=selc(2);homestring='Memphis';loadPredictionTest;
    game=selc(3);homestring='Cincinnati';loadPredictionTest;
    game=selc(4);homestring='Michigan';loadPredictionTest;
    game=selc(5);homestring='NewMexico';loadPredictionTest;
    %configure 'predictions' variable
    for i=1:length(selc);predictions{i}=predictions{selc(i)};end 
    predictions(6:end)=[];
    if fact4y==1;save predictiontest1 predictions;
        elseif fact4y==0;save predictiontest0 predictions;end
    %% load games for future surprise / value test
    loadValueTest; %not in use atm
    %% run game logs for scanner & organize variables
    %here, we will run 'loadLogAndTR' to co-align matrices for GLM and
    %other regressions
    susav=[];d_mat=[];d_mat_hrf=[];d_matUS_hrf=[];event_mat=[];%pre-assign
    d_mat_in=[];fmrig=0;cumu_poss=0;g_updates=[];
    gamebinsurtotal=[];gamebindurmean=[];gamebinsurmean=[];gamebinsurstdev=[];
    gamebinBCsurtotal=[];gamebinBICsurtotal=[];gamebinsusmean=[];
    gamebinsusmean2=[];gamebinsusmedian=[];gflined=[];
    gamebinaudmean=[];gamebinlummean=[];gamebinspeechmean=[];
    load courtpos;%load court position data (left v right side of court)
    hrfn=[root 'analysis/in_hrf.mat'];load(hrfn);%loads canonical hrf
    %load speech;%speech present/absent for each TR - not in use, using envelope / prosody instead!
    startcut=10;% remove this number of TRs from clip onset (Nastase, 2019)
    postgametrs=2;% keep this number of TRs after clip offset to avoid super abrupt cutoff
    %within each of the below, it will produce some # of plots, some of which
    %include what are in Fig S1A (win prob / surprise over time, each game)
    game=2;homestring='Creighton';gf='4_8';loadLogAndTR;%magic happening in <-
    game=3;homestring='NotreDame';gf='1_7';loadLogAndTR;
    game=5;homestring='Missouri';gf='2_2';loadLogAndTR;
    game=6;homestring='StMarys';gf='4_7';loadLogAndTR;
    game=13;homestring='FloridaSt';gf='3_3';loadLogAndTR;
    game=15;homestring='WichitaSt';gf='1_5';loadLogAndTR;
    game=18;homestring='Indiana';gf='1_4';loadLogAndTR;
    game=19;homestring='UNLV';gf='1_6';loadLogAndTR;
    game=27;homestring='Gonzaga';gf='3_7';loadLogAndTR;gs=0;
    cumu_poss=cumu_poss(1:end-1);
    %cutting off the beginning of the video loses a few possessions, so we
    %need to put them in new order and investigate the remaining 157!
    newposs=zeros(size(event_mat,1),1);oldposs=newposs;
    for i=1:size(event_mat,1)
        newposs(i)=cumu_poss(event_mat(i,1))+event_mat(i,4);
        oldposs(i)=event_mat(i,4);
    end
    for i=1:max(newposs)
        fill=find(newposs==i);
        if isempty(fill)
            above=find(newposs>i);
            newposs(above)=newposs(above)-1;
            above=find(oldposs>i);
            oldposs(above)=oldposs(above)-1;
        end
    end
    if gs;figure;plot(newposs);figure;plot(oldposs);end
    
    %bin suspense for each poss into bins for ISC analysis - not in use
    bins=32;%num suspense bins
    susbins=prctile(event_mat(:,11),linspace(0,100,bins+1));%the +1 is so the highest suspense bin is included
    TRsusbin=zeros(size(event_mat,1),1);susbinmean=zeros(bins,1);
    audbinmean=susbinmean;lumbinmean=susbinmean;motbinmean=susbinmean;
    speechbinmean=susbinmean;susbinmedian=susbinmean;nTRs_susbin=susbinmean;
    for i=1:bins
        %9.30.20 - swapped >= and < to > and <=
        fill=find(and(event_mat(:,11)>susbins(i),event_mat(:,11)<=susbins(i+1)));
        if i==bins;fill=find(and(event_mat(:,11)>susbins(i),event_mat(:,11)<=susbins(i+1)));end
        susbinmean(i)=nanmean(event_mat(fill,11));
        susbinmedian(i)=nanmedian(event_mat(fill,11));
        audbinmean(i)=nanmean(event_mat(fill,17));
        lumbinmean(i)=nanmean(event_mat(fill,18));
        motbinmean(i)=nanmean(event_mat(fill,19));
        speechbinmean(i)=nanmean(event_mat(fill,20));
        TRsusbin(fill)=i;
        nTRs_susbin(i)=length(fill);
    end
    if gs;figure;plot(nTRs_susbin);xlabel('Suspense bin');ylabel('# TRs');sn=[root 'pics/SusbinTRBreakdown'];print(sn,'-dpdf','-bestfit');        
        a=susbinmean;b=audbinmean;figure;scatter(a,b);[r,p]=corrcoef(a,b);lsline;title(['r=' num2str(r(2),3) ',p=' num2str(p(2),3)]);
        b=lumbinmean;figure;scatter(a,b);[r,p]=corrcoef(a,b);lsline;title(['r=' num2str(r(2),3) ',p=' num2str(p(2),3)]);
        b=motbinmean;figure;scatter(a,b);[r,p]=corrcoef(a,b);lsline;title(['r=' num2str(r(2),3) ',p=' num2str(p(2),3)]);
        b=speechbinmean;figure;scatter(a,b);[r,p]=corrcoef(a,b);lsline;title(['r=' num2str(r(2),3) ',p=' num2str(p(2),3)]);
        % compare automated versus my manual markings, to verify (not using my
        % manual anymore)
        a=n_camang;b=tot_mot;figure;scatter(a,b);r=corrcoef(a,b);end
    
    %% event matrix stuff! 
    %this will do some modifications to the event matrix (matrix w/o HRF 
    %conv -> HMM / ISC / non-GLM analyses with just shifted time course)
    %and d_mat matrix (TR-level)
    event_mat =[event_mat(:,1:11) TRsusbin event_mat(:,12:end)];
    %bin by suspense!!!
    d_mat_hrf(isnan(d_mat_hrf))=0;d_matUS_hrf(isnan(d_matUS_hrf))=0;
    %meancenter data
    for i=1:size(event_mat,2);event_mat2(:,i)=meancenter(event_mat(:,i));end
    for i=1:size(d_mat_hrf,2);d_mat_hrf2(:,i)=meancenter(d_mat_hrf(:,i));end
    for i=1:size(d_mat_hrf2,2) %xcorr
        for ii=1:size(d_mat_hrf2,2)
            r=corrcoef(d_mat_hrf2(:,i),d_mat_hrf2(:,ii),'rows','complete');rs(i,ii)=r(2);
        end
    end
    for i=1:size(d_matUS_hrf,2);d_matUS_hrf2(:,i)=meancenter(d_matUS_hrf(:,i));end
    for i=1:size(d_matUS_hrf2,2) %xcorr
        for ii=1:size(d_matUS_hrf2,2)
            r=corrcoef(d_matUS_hrf2(:,i),d_matUS_hrf2(:,ii),'rows','complete');rs2(i,ii)=r(2);
        end
    end
    %% plot some design matrices
    if gs;cats={'Game number','Seed difference','Game remaining','Possession #','Home possession',...
            'Home win probability','Winning team in possession','Possession going right',...
            'Free-throw upcoming','Surprise (unsigned)','Suspense','Suspense bin','Dunk','Unusual play','Turnover',...
            'Entropy change','Surprise (signed)','Auditory env','Luminance','Video Motion',...
            'Prosody'};
        figure;imagesc(event_mat2);ylabel('TRs');ggg=gca;ggg.XTick=1:size(rs,2);
        ggg.XTickLabel=cats;xtickangle(45);colorbar;
        sn=[root 'pics/eventmatrix'];print(sn,'-dpdf','-bestfit');
        cats={'Game number','Game remaining','Suspense','Suspense bin'};
        figure;imagesc(event_mat2(:,[1 3 11 12]));ylabel('TRs');ggg=gca;ggg.XTick=1:size(rs,2);
        ggg.XTickLabel=cats;xtickangle(45);colorbar;
        sn=[root 'pics/eventmatrixShort'];print(sn,'-dpdf','-bestfit');
        cats={'Seed difference','Game remaining','Home possession','Score (unsigned)','Score (signed)',...
            'Surprise (unsigned)','Surprise (signed)','Suspense','Turnover','Free-throw',...
            '2-point shot','3-point shot','Auditory env','Luminance','Video Motion',...
            'Prosody','Court position','Dunk','Unusual play','Entropy change'};
        figure;imagesc(rs);ggg=gca;ggg.XTick=1:size(rs,2);ggg.YTick=1:size(rs,2);
        ggg.XTickLabel=cats;ggg.YTickLabel=cats;xtickangle(45);ytickangle(45);colorbar;
        sn=[root 'pics/RegressorCorrelationCoefficient'];print(sn,'-dpdf','-bestfit');
        figure;imagesc(d_mat_hrf2);ylabel('TRs');ggg=gca;ggg.XTick=1:size(rs,2);
        ggg.XTickLabel=cats;xtickangle(45);colorbar;
        sn=[root 'pics/designmatrixHRFRS'];print(sn,'-dpdf','-bestfit');
        figure;imagesc(d_mat_hrf2);ylabel('TRs');ggg=gca;ggg.XTick=1:size(rs,2);
        ggg.XTickLabel=cats;xtickangle(45);colorbar;
        sn=[root 'pics/designmatrixHRFRS'];print(sn,'-dpdf','-bestfit');
        cats={'Game remaining','Home possession','Surprise (unsigned)','Suspense','Auditory env','Luminance','Video motion'};
        figure;imagesc(d_mat_hrf2(:,[2:3 6:7 13:15]));ggg=gca;ylabel('TRs');
        ggg.XTickLabel=cats;xtickangle(45);ytickangle(45);colorbar;
        sn=[root 'pics/designmatrixShort'];print(sn,'-dpdf','-bestfit');
        figure;scatter(susav(:, 1),susav(:,2));xlabel('Suspense');ylabel('Auditory envelope');lsline;
        sn=[root 'pics/TRXSuspenseAudioAllCorr'];print(sn,'-dpdf','-bestfit');
        figure;scatter(susav(:, 1),susav(:,3));xlabel('Suspense');ylabel('Video luminance');lsline;
        sn=[root 'pics/TRXSuspenseLumAllCorr'];print(sn,'-dpdf','-bestfit');
        [r1,p1]=corrcoef(susav(:,1),susav(:,2));[r2,p2]=corrcoef(susav(:,1),susav(:,3));
    end
    %log 1st TR of each game
    for ii=1:fmrig
        fill=find(event_mat(:,1)==ii);event_fl(ii,1)=fill(1);event_fl(ii,2)=fill(end);
        if ii==1;d_fl(ii,1)=1;d_fl(ii,2)=length(fill)+postgametrs;
        else;d_fl(ii,1)=d_fl(ii-1)+length(prevfill)+postgametrs;
            d_fl(ii,2)=d_fl(ii,1)-1+length(fill)+postgametrs;
        end
        prevfill=fill;
    end
    %% create p_fl - a log of possessions for all games by first TR
    %p_fl columns:1=game #,2=TR start (full matrix,
    %cumulative),3=possession # in orig game,4=game remaining,5=suspense,
    %6=surprise leading into possession,7=surprise @ end of
    %possession,8=dunk?,9=odd play?,10=turnover?,11=entropy leading into
    %possession,12=entropy @ end,13=win prob home @ end of possession
    j=1;jj=1;jk=1;gamenum=0;oldgamenum=0;nanpergame=zeros(fmrig,1);
    for ii=2:size(event_mat,1)%possessions
        newgame=0;
        if event_mat(ii,4)~=event_mat(ii-1,4) %possession change
            if ~isnan(event_mat(ii,4))   
                gamenum=event_mat(ii,1);if gamenum>oldgamenum;newgame=1;end
                p_fl(j,1)=ii;orig_event_fl(j,1)=event_mat(ii,4);
                gr_fl(j,1)=event_mat(ii,3);
                sus_fl(j,1)=event_mat(ii,11);sur_fl(j,1)=event_mat(ii,10);
                dunk_fl(j,1)=event_mat(ii,13);odd_fl(j,1)=event_mat(ii,14);
                to_fl(j,1)=event_mat(ii,15);
                ent_fl(j,1)=event_mat(ii,16);wph_fl(j,1)=event_mat(ii,6);
                if ~newgame;sur_post_fl(j-1,1)=sur_fl(j,1);
                    ent_post_fl(j-1,1)=ent_fl(j,1);
                    wph_post_fl(j-1,1)=wph_fl(j,1);end %don't count surprise post if it's newgame
                if gamenum>1;if newgame;sur_fl(j,1)=nan;end;end %override if it's the first possession
                j=j+1;oldgamenum=gamenum;
            else
                nanpergame(gamenum)=nanpergame(gamenum)+1;
                if nanpergame(gamenum)<2 %final update in game!
                    sur_post_fl(j-1,1)=event_mat(ii,10);
                    ent_post_fl(j-1,1)=event_mat(ii,16);
                    wph_post_fl(j-1,1)=event_mat(ii,6);
                end
            end
        end
        if event_mat(ii,6)~=event_mat(ii-1,6)
            if ~isnan(event_mat(ii,6))   
                wc_fl(jj,1)=ii;sur_wc_fl(jj,1)=event_mat(ii,10);orig_event_wc_fl(jj,1)=event_mat(ii,4);jj=jj+1;
            end
        end
    end
    %put it all together into p_fl
    p_fl(2:end+1)=p_fl;p_fl(1)=1;p_fl=p_fl-1;%subtract 1 to round down 
    orig_event_fl(2:end+1)=orig_event_fl;orig_event_fl(1)=event_mat(1,4);
    gr_fl(2:end+1)=gr_fl;gr_fl(1)=event_mat(1,3);
    sus_fl(2:end+1)=sus_fl;sus_fl(1)=event_mat(1,11);
    sur_post_fl(2:end+1)=sur_post_fl;sur_post_fl(1)=sur_fl(1);
    sur_fl(2:end+1)=sur_fl;sur_fl(1)=nan;
    dunk_fl(2:end+1)=dunk_fl;dunk_fl(1)=event_mat(1,13);
    odd_fl(2:end+1)=odd_fl;odd_fl(1)=event_mat(1,14);
    to_fl(2:end+1)=to_fl;to_fl(1)=event_mat(1,15);
    wph_post_fl(2:end+1)=wph_post_fl;wph_post_fl(1)=wph_fl(1);
    ent_post_fl(2:end+1)=ent_post_fl;ent_post_fl(1)=ent_fl(1);
    ent_fl(2:end+1)=ent_fl;ent_fl(1)=nan;
    p_fl(:,2)=p_fl;p_fl(:,3)=orig_event_fl;p_fl(:,4)=gr_fl;p_fl(:,5)=sus_fl;
    p_fl(:,6)=sur_fl;p_fl(:,7)=sur_post_fl;p_fl(:,8)=dunk_fl;p_fl(:,9)=odd_fl;
    p_fl(:,10)=to_fl;p_fl(:,11)=ent_fl;p_fl(:,12)=ent_post_fl;p_fl(:,13)=wph_post_fl;
    wc_fl(2:end+1)=wc_fl;wc_fl(1)=1;wc_fl(:,2)=wc_fl;
    sur_wc_fl(2:end+1)=sur_wc_fl;sur_wc_fl(1)=0;wc_fl(:,3)=sur_wc_fl;
    orig_event_wc_fl(2:end+1)=orig_event_wc_fl;orig_event_wc_fl(1)=0;wc_fl(:,4)=orig_event_wc_fl;
    for ii=1:size(event_fl,1)
        fill=and(p_fl(:,2)>=event_fl(ii,1)-1,p_fl(:,2)<=event_fl(ii,2)-1);
        p_fl(fill,1)=ii;
        fill=and(wc_fl(:,2)>=event_fl(ii,1)-1,wc_fl(:,2)<=event_fl(ii,2)-1);
        wc_fl(fill,1)=ii;
    end
    
    %plot a few variables across possessions
    if gs;for i=1:size(p_fl,2);p_fl2(:,i)=meancenter(p_fl(:,i));end
        cats={'Game remaining','Suspense','Surprise'};
        figure;imagesc(p_fl2(:,[4 5 7]));ylabel('Possessions');ggg=gca;ggg.XTick=1:3;
        ggg.XTickLabel=cats;%colorbar;xtickangle(45);
        sn=[root 'pics/possmemmatrix'];print(sn,'-dpdf','-bestfit');clear p_fl2;
    end
    %do games w/ more possessions have higher mean suspense / surprise? no
    %(but they are longer)
    for i=1:fmrig;ppg(i)=length(find(p_fl(:,1)==i));end
    r=corrcoef(ppg,gamebindurmean);r_ppg_dur=r(2);r=corrcoef(ppg,gamebinsurmean);r_ppg_sur=r(2);
    if exist('gamebinsusmean','var');[r,p]=corrcoef(ppg,gamebinsusmean);r_ppg_sus=r(2);p_ppg_sus=p(2);end
    
    %% save the things
    if bayes;save d_event_mat_b event_mat event_mat2 d_mat_in d_mat_hrf d_mat_hrf2 d_matUS_hrf d_matUS_hrf2 susbins d_fl event_fl p_fl newposs oldposs wc_fl;
    else;if fact4y==1;save g1 g;save d_event_mat1 event_mat event_mat2 d_mat_in d_mat_hrf d_mat_hrf2 d_matUS_hrf d_matUS_hrf2 susbins d_fl event_fl p_fl newposs oldposs wc_fl;
        elseif fact4y==0;save g0 g;save d_event_mat0 event_mat event_mat2 d_mat_in d_mat_hrf d_mat_hrf2 d_matUS_hrf d_matUS_hrf2 susbins d_fl event_fl p_fl newposs oldposs wc_fl;
        end
    end
    sn=['sus_event_mat-' num2str(sus_model)];
    save(sn,'event_mat','gamebinsusmean','gamebinsusmedian','susbinmean','lumbinmean','motbinmean','speechbinmean');
    
    load susgamebinmean game_poss n_camang
    %if going to save below, load above first to get game_poss and n_camang
    if fact4y==1;save gamebinmeans1 g_updates gamebindurmean gamebinsurmean gamebinsusmean gamebinsusmedian gamebinsurmean gamebinsurstdev gamebinsurtotal gamebinBCsurtotal gamebinBICsurtotal game_poss n_camang n_camang_mot tot_mot gamebinaudmean gamebinlummean gamebinspeechmean lumbinmean motbinmean speechbinmean susbinmean susbinmedian event_mat
    elseif fact4y==0;save gamebinmeans0 g_updates gamebindurmean gamebinsurmean gamebinsurmean gamebinsusmean gamebinsusmedian gamebinsurstdev gamebinsurtotal gamebinBCsurtotal gamebinBICsurtotal game_poss n_camang n_camang_mot tot_mot gamebinaudmean gamebinlummean gamebinspeechmean lumbinmean motbinmean speechbinmean susbinmean susbinmedian event_mat
    end
    
    %compare surprise across possessions using KP and fact4 models because
    %of some idiosyncratic ways within-possession events were interpolated,
    %I believe, but it's still ~ r=0.92
    %(slightly different than comparing win probabilities at each moment)
    compsurprise=1;
    if compsurprise;load('d_event_mat1.mat');p_fl1=p_fl;load('d_event_mat0.mat');
        x=p_fl(:,7);y=p_fl1(:,7);[r,p]=corrcoef(x,y);r(2)
        set(0,'DefaultAxesFontSize',25);figure;scatter(x,y);
        xlabel('Surprise, all possessions, expert model');ylabel('Surprise, all possessions, 4-factor model');
        sn=[root 'pics/SurpriseModelCompViewed'];print(sn,'-dpdf','-bestfit');
    end
    %view err to compare how this looks for different event types (3rd
    %column)
    for i=1:10;f=find(gflined(:,3)==i);count(i)=length(f);err(i)=nanmean(gflined(f,1)-gflined(f,2));end
end