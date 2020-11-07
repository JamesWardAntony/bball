%% heart rate / eye tracking + create movie start onset times (for fMRI)
% create Fig 4A,C,D,+ output for E, => R (generating output for Table S2)
% heart rate stuff not in use
close all;clear;plotdefs;[base,root]=load_root();addpath(genpath([root 'analysis/']));
subs={'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'};
ety=[1 0;2 0;3 0;4 1;5 0;6 0;7 1;8 1;9 1;10 1;11 1;12 1;13 1;14 1;15 1;16 0;17 1;18 1;19 1;20 1];
hry=[1 1;2 1;3 1;4 1;5 1;6 1;7 1;8 1;9 1;10 1;11 1;12 1;13 1;14 1;15 1;16 0;17 1;18 1;19 1;20 1];
%subs={'4'};ety=[4 1];hry=[4 1];%override for troubleshooting just 1 sub
nS=length(subs);gs=0;aw=4;%line width
load recall_vars rec_all_poss_xsub;%load recall data
tngs=9;ngs=3;runs=6;%total games, viewed games / run, total runs (view+recall)
hrfs=200;etfs=1000;%preset hr and et sampling rate
maxhr=110;minhr=40;maxfrq=(maxhr/60);minfrq=(minhr/60);%physiological max/min heart rate
xmax=1920;ymax=1080;%screen parameters
screenadj=0.9;%this screenadj variable =>  the inner 80% of the video, like the display
xminvid=(1-screenadj)*xmax;xmaxvid=screenadj*xmax;yminvid=(1-screenadj)*ymax;ymaxvid=screenadj*ymax;
hrsus_a=zeros(nS,1);hbsus_a=hrsus_a;hbgr_a=hrsus_a;hblum_a=hrsus_a;%presets
hbsur_a=hrsus_a;hbaud_a=hrsus_a;ersus_a=hrsus_a;ersur_a=hrsus_a;ersur_mean_a=hrsus_a;ebsus_a=hrsus_a;
ebgr_a=hrsus_a;eblum_a=hrsus_a;ebsur_a=hrsus_a;ebaud_a=hrsus_a;mass=[];mass0=[];
%time on acq clock=11:19:00,clock func output on Mac matlab=11:32:40.58
AcqMacDiff=13*60+40.5;%difference in computer time b/t acquisition and mac computer, second precision
ets=cell(nS,1);hrs=ets;
pixpd=41;fovea_angle=2;%41 pixels / degree, want 2 degrees for fovea
vis_circ=pixpd*fovea_angle;%circle for fovea
fact4y=1;
if fact4y==1;load g1;load d_event_mat1 event_fl p_fl;load gamebinmeans1;
elseif fact4y==0;load g0;load d_event_mat0 event_fl p_fl;load gamebinmeans0;end
load gminls1st9.mat;%load game lengths
load questionnaire;
startcut=10;%startcut is to avoid onset issues w/ fMRI analyses
svHR=1;%save data at end?
for i=1:nS %load data
    sub=char(subs(i)) %print progress
    dr=[root 'data/nonfmri/' sub '/'];
    hret_a=[];gset=1:ngs;doruns=1:2:runs-1;%doruns is complex because of a couple subjects w/ odd run orders
    cd(dr);load('g_o_m2.mat');load('g_o.mat');recallruns=2:2:runs;
    hret_mov=cell(tngs,1);MSTR=zeros(tngs,2);METR=MSTR;RSTR=MSTR;RETR=MSTR;%pre-assign
    files = dir(fullfile(pwd, '*PULS.log'));
    if ety(i,2);etfiles = dir(fullfile(pwd, '*.asc'));end %find files
    if gs;figure(1);end
    if strcmpi(char(sub),'4') || strcmpi(char(sub),'5')
        doruns=[0.5 1 3 5];files2=files(1);files=files(2:end); %adjust subjects 4/5
    else;doruns=1:2:runs-1;
    end
    for ii=doruns %only VIEW runs
        switch ii;case 0.5;trial1=1;case 1;trial1=1;case 3;trial1=4;case 5;trial1=7;end %funky way to make data work
        if strcmpi(char(sub),'16')
            %not using hr for 16 - NOTREAL file is another
            %copied above as placeholder
            content=fileread(files(ii).name);hr=textscan(content,'%n%s%n%s','HeaderLines',8);
            content=fileread(files(ii).name);hr_start=textscan(content,'%s%s%n%s','HeaderLines',1);
        else %other subjects
            if ii~=0.5 %if first run of sub 4/5
                content=fileread(files(ii).name);hr=textscan(content,'%n%s%n%s','HeaderLines',8);
                content=fileread(files(ii).name);hr_start=textscan(content,'%s%s%n%s','HeaderLines',1);
            else;content=fileread(files2(1).name);hr=textscan(content,'%n%s%n%s','HeaderLines',8);
                content=fileread(files2(1).name);hr_start=textscan(content,'%s%s%n%s','HeaderLines',1);
            end
        end
        hrt_h=str2double(hr_start{1,4}{1,1}(2:3));hrt_m=str2double(hr_start{1,4}{1,1}(4:5));
        hrt_s=str2double(hr_start{1,4}{1,1}(6:7));%HR file times
        mlt_h=g_o_m2{trial1}.init_time_r(4);mlt_m=g_o_m2{trial1}.init_time_r(5);
        mlt_s=g_o_m2{trial1}.init_time_r(6);%matlab times
        %diff b/t hr start and ml start - note we also need to add diff in timing as measured by testing the computers directly
        hr_mlt=(hrt_h-mlt_h)*60*60+(hrt_m-mlt_m)*60+(hrt_s-mlt_s)+AcqMacDiff;
        %% pulses
        z=hr{1,4};trcount=0;trt=zeros(length(z),1);
        for iii=1:length(z);if ~strcmpi('',string(z(iii)));trcount=trcount+1;trt(iii)=1;end;end
        runl=length(z)/hrfs;%run length
        tpts=linspace(0,runl,length(z));%0:1/hrfs:runl;%time of all points in file, in s
        %figure;plot(tpts,trt);
        %% PPG
        z=hr{1,3};%PPG signal
        diffz=diff(z);prevbeat=-2;fs=0;efperc=prctile(diffz,70);
        for ff=2:length(diffz)-1
            if diffz(ff)>diffz(ff-1)%ascending
                if diffz(ff)<diffz(ff+1) %peak
                    if diffz(ff)>efperc
                        ptime=ff/hrfs;
                        if ptime>prevbeat+0.5 %in plausible range
                            if ptime<prevbeat+1.5
                                fs=fs+1;p2ps(fs,1)=ptime;p2ps(fs,2)=(ptime-prevbeat);
                            end
                        end
                        prevbeat=ptime;
                    end
                end
            end
        end
        %figure;subplot(211);histogram(p2ps(:,2));subplot(212);histogram(diff(p2ps(:,2)));
        fn=[dr 'cwt-' num2str(ii) '.mat'];%wavelet
        if ~exist(fn,'file');[coefs,frqs]=cwt(z,hrfs);save(fn,'coefs','frqs');else;load(fn);end;
        hr_r=find(and(frqs>minfrq,frqs<maxfrq));%frequencies of interest
        pow=abs(coefs);%rectify wavelet
        [~,maxhr_i]=max(pow(hr_r,:));%find max frequency index
        clear coefs pow z
        %% align w/ signal, finding instant heart rate metric (hr_inst)
        %also find movie start / end times for alignment + output for fMRI
        %find movie start w/r/t HR start
        if ii==0.5;gset=1;
        elseif or(ii==1 && strcmpi(char(sub),'4'),ii==1 && strcmpi(char(sub),'5'))
            gset=2:ngs;
        else;gset=1:ngs;
        end
        ews=0;
        for iii=gset
            gtrialt=trial1+iii-1;
            if str2double(sub)==6
                if gtrialt==1;g_o_m2{1}.preMovieTime=g_o_m2{1}.preMovieTime-148;end %JWA, 143->148
                if gtrialt==4;g_o_m2{4}.preMovieTime=g_o_m2{4}.preMovieTime-88;end %JWA,90->88,gtrial==1->4
                if gtrialt==7;g_o_m2{7}.preMovieTime=g_o_m2{7}.preMovieTime-7;end %JWA,gtrial==1->7
            elseif str2double(sub)==1
                if gtrialt==4;g_o_m2{4}.preMovieTime=g_o_m2{4}.preMovieTime+6;end %JWA,0->-6
                if gtrialt==7;g_o_m2{7}.preMovieTime=g_o_m2{7}.preMovieTime+11;end %JWA,0->-11
            elseif str2double(sub)==4
                if gtrialt==2;g_o_m2{2}.preMovieTime=g_o_m2{2}.preMovieTime-7;end %JWA,0->-4 (or 5)
            elseif str2double(sub)==5
                if gtrialt==2;g_o_m2{2}.preMovieTime=g_o_m2{2}.preMovieTime-55;end %JWA,0->-4
                if gtrialt==4;g_o_m2{4}.preMovieTime=g_o_m2{4}.preMovieTime+2;end %JWA,0->2
                if gtrialt==7;g_o_m2{7}.preMovieTime=g_o_m2{7}.preMovieTime+3;end %JWA,0->3
            elseif str2double(sub)==7
                if gtrialt==4;g_o_m2{4}.preMovieTime=g_o_m2{4}.preMovieTime+2;end %JWA,0->2
                if gtrialt==7;g_o_m2{7}.preMovieTime=g_o_m2{7}.preMovieTime+4;end %JWA,0->4
            elseif str2double(sub)==8
                if gtrialt==7;g_o_m2{7}.preMovieTime=g_o_m2{7}.preMovieTime+2;end %JWA,0->2
            elseif str2double(sub)==9
                if gtrialt==4;g_o_m2{4}.preMovieTime=g_o_m2{4}.preMovieTime+2;end %JWA,0->2
                if gtrialt==7;g_o_m2{7}.preMovieTime=g_o_m2{7}.preMovieTime+3;end %JWA,0->3
            elseif str2double(sub)==17
                if gtrialt==4;g_o_m2{4}.preMovieTime=g_o_m2{4}.preMovieTime+4;end %JWA,0->4
                if gtrialt==7;g_o_m2{7}.preMovieTime=g_o_m2{7}.preMovieTime+7;end %JWA,0->7
            end
            if str2double(sub)<40
                if ~ews;ews=1;
                    endWaitTrigger=((g_o_m2{gtrialt}.movieEnd-gtruemin(g_o(gtrialt)))-g_o_m2{trial1}.preMovieTime);%-3
                    if gtrialt==2;endWaitTrigger=((g_o_m2{gtrialt}.movieEnd-gtruemin(g_o(gtrialt)))-g_o_m2{gtrialt}.preMovieTime);end
                end
                MSTR(gtrialt,1)=floor((g_o_m2{gtrialt}.movieEnd-gtruemin(g_o(gtrialt)))-endWaitTrigger);
                if str2double(sub)==18
                    if gtrialt==3
                        MSTR(gtrialt,1)=MSTR(gtrialt,1)-3;
                    end
                end
                MSTR(gtrialt,2)=(g_o_m2{gtrialt}.movieEnd-gtruemin(g_o(gtrialt)))-MSTR(gtrialt,1);
                MSwrtHRS=((g_o_m2{gtrialt}.movieEnd-gtruemin(g_o(gtrialt)))-g_o_m2{trial1}.init_time)-hr_mlt;% movie start
            else
                if ~ews;ews=1;
                    endWaitTrigger=(g_o_m2{gtrialt}.movieStart-g_o_m2{trial1}.preMovieTime);%-3
                end
                MSTR(gtrialt,1)=floor(g_o_m2{gtrialt}.movieStart-endWaitTrigger);
                MSTR(gtrialt,2)=g_o_m2{gtrialt}.movieStart-endWaitTrigger-MSTR(gtrialt,1);
                MSwrtHRS=(g_o_m2{gtrialt}.movieStart-g_o_m2{trial1}.init_time)-hr_mlt;% movie start
            end
            METR(gtrialt,1)=floor(g_o_m2{gtrialt}.movieEnd-endWaitTrigger);
            METR(gtrialt,2)=g_o_m2{gtrialt}.movieEnd-endWaitTrigger-METR(gtrialt,1);
            maxtrialtime(gtrialt)=g_o_m2{gtrialt}.timelog(end,end)-endWaitTrigger;
            MEwrtHRS=(g_o_m2{gtrialt}.movieEnd-g_o_m2{trial1}.init_time)-hr_mlt;%movie end
            secspace=MSwrtHRS:1:MEwrtHRS+1;%space for averaging TRs
            secspace_norm=secspace-MSwrtHRS;%space for averaging TRs
            
            hr_inst=nan(length(secspace)-1,1);
            for iiii=1:length(secspace)-1
                fill=and(tpts>=secspace(iiii),tpts<secspace(iiii+1));%find time pts w/in TR
                hr_inst(iiii)=nanmean(frqs(hr_r(maxhr_i(fill))))*60;%instant HR
            end
            %hr_inst_z=zscore(hr_inst);%figure;histogram(hr_inst_z);
            %hr_inst(abs(hr_inst_z)>3.72)=NaN;%nan out below p=0.0001, kills outliers
            hr_inst=smoothdata(hr_inst,'movmedian',5);
            %hr_inst2=ksdensity(hr_inst);
            hret_mov{gtrialt,1}.hr_inst=hr_inst;hret_mov{gtrialt,1}.tpts=tpts;
            fill=and(p2ps(:,1)>MSwrtHRS,p2ps(:,1)<MEwrtHRS);
            hret_mov{gtrialt,1}.rmssd=sqrt(mean(diff(p2ps(fill,2)).^2));%standard deviation of hr signal
            hret_mov{gtrialt,1}.secspace=secspace;hret_mov{gtrialt,1}.secspace_norm=secspace_norm;
            if gs;figure(1);subplot(3,3,gtrialt);plot(secspace_norm(1:end-1),hr_inst);
                title([sub ', movie trial ' (num2str(gtrialt))]);ax=gca;ax.LineWidth=aw;
                ax.YLim=[minhr maxhr];xlabel('Time in movie (s)');ylabel('Heart rate (bpm)');
                if gtrialt==tngs;sn=[root 'pics/HR-' sub];ppos;print(sn,'-dpdf','-bestfit');end
            end
            hret_mov{gtrialt,1}.g_id(gtrialt)=g_o(gtrialt);hret_mov{gtrialt,1}.pref=g_o_m2{gtrialt}.pref(1);
            hret_mov{gtrialt,1}.enjoy=g_o_m2{gtrialt}.enjoy(1);
        end
    end
    %% recall runs - process separately to store RSTR
    %this is simpler because there were no exceptions during running
    for ii=recallruns %only recall runs
        switch ii;case 2;trial1=10;case 4;trial1=13;case 6;trial1=16;end %funky way to make data work
        %% align w/ signal
        %find movie start w/r/t HR start
        if ii==0.5;gset=1;%in case messup later...
        else;gset=1:ngs;
        end
        ews=0;
        for iii=gset
            gtrialt=trial1+iii-1;
            if ~ews;ews=1;
                endWaitTrigger=g_o_m2{gtrialt}.timelog(2);
            end
            RSTR(gtrialt-tngs,1)=floor(g_o_m2{gtrialt}.starttime-endWaitTrigger);
            RSTR(gtrialt-tngs,2)=(g_o_m2{gtrialt}.starttime-endWaitTrigger)-RSTR(gtrialt-tngs,1);
            RETR(gtrialt-tngs,1)=floor(g_o_m2{gtrialt}.endtime-endWaitTrigger);
            RETR(gtrialt-tngs,2)=(g_o_m2{gtrialt}.endtime-endWaitTrigger)-RETR(gtrialt-tngs,1);
        end
    end
    
    %% eye tracking
    % note: restarting scans doesn't affect eye tracking data output format
    doruns=1:2:5;
    if ety(i,2)
        ems=[];ems2=[];blinks=[];saccs=[];
        gset=1:ngs;
        for ii=doruns
            switch ii;case 1;trial1=1;case 3;trial1=4;case 5;trial1=7;end %funky way to make data work
            for iii=gset
                gtrialt=trial1+iii-1;
                fn=[dr 'etb-' num2str(gtrialt) '.mat'];
                if exist(fn,'file');load(fn);
                else
                    if hry(i,2)==0
                        ews=0;
                        if ~ews;ews=1;
                            endWaitTrigger=((g_o_m2{gtrialt}.movieEnd-gtruemin(g_o(gtrialt)))-g_o_m2{trial1}.preMovieTime);%-3
                        end
                        MSTR(gtrialt,1)=floor((g_o_m2{gtrialt}.movieEnd-gtruemin(g_o(gtrialt)))-endWaitTrigger);
                        MSTR(gtrialt,2)=(g_o_m2{gtrialt}.movieEnd-gtruemin(g_o(gtrialt)))-MSTR(gtrialt,1);
                        MSwrtHRS=((g_o_m2{gtrialt}.movieEnd-gtruemin(g_o(gtrialt)))-g_o_m2{trial1}.init_time)-hr_mlt;% movie start
                        
                        METR(gtrialt,1)=floor(g_o_m2{gtrialt}.movieEnd-endWaitTrigger);
                        METR(gtrialt,2)=g_o_m2{gtrialt}.movieEnd-endWaitTrigger-METR(gtrialt,1);
                        maxtrialtime(gtrialt)=g_o_m2{gtrialt}.timelog(end,end)-endWaitTrigger;
                        MEwrtHRS=(g_o_m2{gtrialt}.movieEnd-g_o_m2{trial1}.init_time)-hr_mlt;%movie end
                        secspace=MSwrtHRS:1:MEwrtHRS+1;%space for averaging TRs
                        secspace_norm=secspace-MSwrtHRS;%space for averaging TRs
                        hret_mov{gtrialt,1}.secspace=secspace;hret_mov{gtrialt,1}.secspace_norm=secspace_norm;
                        hret_mov{gtrialt,1}.g_id(gtrialt)=g_o(gtrialt);hret_mov{gtrialt,1}.pref(gtrialt)=g_o_m2{gtrialt}.pref(1);
                        hret_mov{gtrialt,1}.enjoy(gtrialt)=g_o_m2{gtrialt}.enjoy(1);
                        hr_inst=nan(length(secspace)-1,1);
                        hret_mov{gtrialt,1}.hr_inst=hr_inst;
                    end
                    secspace_norm=hret_mov{gtrialt,1}.secspace_norm;
                    content=fileread(etfiles(gtrialt).name);clear et etmat2;
                    etmat=nan(8*60*1000,4);j=0;jj=0;jjj=0;
                    bmat=[];smat=[];
                    et=textscan(content,'%s%s%s%n%n%s','HeaderLines',27,'EmptyValue',-Inf);
                    if strcmpi(char(sub),'18');et=textscan(content,'%s%s%s%n%n%s','HeaderLines',28,'EmptyValue',-Inf);end
                    proc=1;
                    for iiii=1:length(et{1,1})-1
                        a=cell2str(et{1,1}(iiii));
                        endnum=length(cell2str(et{1,1}(iiii+1)))-3;
                        if length(a)>6
                            %store blinks / saccades
                            %if blink on, don't proceed; if end, proceed again
                            if strcmpi(a(3:8),'SBLINK');proc=0;jj=jj+1;
                                plug=cell2str(et{1,3}(iiii));%find previous point
                                bmat(jj)=str2double(plug(3:endnum));end
                            if strcmpi(a(3:8),'EBLINK');proc=1;end
                            %same deal with saccades
                            if strcmpi(a(3:7),'SSACC');jjj=jjj+1;plug=cell2str(et{1,1}(iiii+1));
                                smat(jjj)=str2double(plug(3:endnum));end
                            if strcmpi(a(3:7),'ESACC');end
                            if proc
                                endnum=length(a)-3;
                                aa=str2double(a(3:endnum));
                                if length(a)>8
                                    if ~isnan(aa)
                                        j=j+1;
                                        etmat(j,1)=aa;
                                        a=cell2str(et{1,2}(iiii));
                                        if length(a)>6
                                            endnum=length(a)-3;
                                            aa=str2double(a(3:endnum));%8
                                            if ~isnan(aa)
                                                etmat(j,2)=aa;
                                                a=cell2str(et{1,3}(iiii));
                                                endnum=length(a)-3;
                                                aa=str2double(a(3:endnum));
                                                etmat(j,3)=aa;
                                                etmat(j,4)=et{1,4}(iiii);
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                    etmat=etmat(~isnan(etmat(:,1)),:);%kill extra rows
                    etmat=etmat(1:end-1,:);%kill last row with strange number
                    bmat2=(bmat-etmat(1,1))/etfs;smat2=(smat-etmat(1,1))/etfs;
                    etmat(:,1)=(etmat(:,1)-etmat(1,1))/etfs;%normalize to first pt
                    etmat2=etmat(and(etmat(:,2)>xminvid,etmat(:,2)<xmaxvid),:);%on screen
                    etmat2=etmat2(and(etmat2(:,3)>yminvid,etmat2(:,3)<ymaxvid),:);%on screen
                    etmat2=etmat2(etmat2(:,4)>1000,:);%kill trials w/o pupil area...
                    etmat2=etmat2(abs(zscore(etmat2(:,4)))<5,:);%kill extreme outliers
                    fn=[dr 'etb-' num2str(gtrialt) '.mat'];
                    save(fn,'etmat','etmat2','secspace_norm','smat2','bmat2');
                end
                ems=[ems;etmat repmat(gtrialt,size(etmat,1),1)];
                ems2=[ems2;etmat2 repmat(gtrialt,size(etmat2,1),1)];
                blinks=[blinks;bmat2' repmat(gtrialt,length(bmat2),1)];
                saccs=[saccs;smat2' repmat(gtrialt,length(smat2),1)];
            end
        end
        %now pool together all etmats to create z-normed plot by x/y
        %coordinate...
        xsteps=7;xspace=linspace(xminvid,xmaxvid,xsteps+1);
        ysteps=5;yspace=linspace(yminvid,ymaxvid,ysteps+1);
        ems_norm=ems2;ems_norm2d=ems2;ems_smooth=ems2;
        ems_xy=nan(xsteps,ysteps);ems_xy_norm=ems_xy;ems_xy_norm2d=ems_xy;ems_xy_smooth=ems_xy;
        y=ems2(:,4);X=[ones(size(ems2,1),1) ems2(:,2) ems2(:,3) ems2(:,2).*ems2(:,3)];
        b=regress(y,X);
        ems_norm(:,4)=y-(b(1)+b(2).*ems2(:,2)+b(3).*ems2(:,3)+b(4).*(ems2(:,2).*ems2(:,3)));
        for ii=1:length(xspace)-1
            for iii=1:length(yspace)-1
                fill=find(and(and(and(ems2(:,2)>xspace(ii),ems2(:,2)<xspace(ii+1)),...
                    ems2(:,3)>yspace(iii)),ems2(:,3)<yspace(iii+1)));
                ems_xy(ii,iii)=nanmean(ems2(fill,4));num_xy(ii,iii)=length(fill);
                ems_norm2d(fill,4)=zscore(ems2(fill,4));
                ems_xy_norm2d(ii,iii)=nanmean(ems_norm2d(fill,4));
                ems_xy_norm(ii,iii)=nanmean(ems_norm(fill,4));
            end
        end
        ems_norm2d=ems_norm2d(ems_norm2d(:,4)<5,:);%kill outliers
        ems_xy_smoothavg=smoothn(ems_xy);%smooth
        for ii=1:length(xspace)-1
            for iii=1:length(yspace)-1
                fill=find(and(and(and(ems2(:,2)>xspace(ii),ems2(:,2)<xspace(ii+1)),...
                    ems2(:,3)>yspace(iii)),ems2(:,3)<yspace(iii+1)));
                ems_smooth(fill,4)=ems2(fill,4)-ems_xy_smoothavg(ii,iii);
                ems_xy_smooth(ii,iii)=nanmean(ems_smooth(fill,4));
            end
        end
        if gs %create descriptive plot about z-scoring, Fig 4A
            mx=max(max(ems_xy));mn=min(min(ems_xy));clims=[mn mx];pts=1:90000;
            figure;subplot(211);imagesc(ems_xy',clims);xlabel('x eye');
            ylabel('y eye');title('orig');c=colorbar;c.Label.String='Mean Pupil Area';
            %note range on graph below is very very very close to 0
            subplot(212);imagesc(ems_xy_norm2d',clims);xlabel('x eye');
            ylabel('y eye');title('z-scored');c=colorbar;c.Label.String='Mean Pupil Area (normed)';
            ppos;sn=[root 'pics/ETdescrip-' sub];print(sn,'-dpdf','-bestfit');
            clims(2)=1900;
            figure;imagesc(ems_xy(2:end-1,:)',clims);set(gca,'xtick',[]);set(gca,'xticklabel',[]);
            set(gca,'ytick',[]);set(gca,'yticklabel',[]);
            xlabel('Gaze location, x-coordinate');
            ylabel('Gaze location, y-coordinate');c=colorbar;c.Label.String='Mean, raw pupil area (au)';
            ppos;sn=[root 'pics/ETOrigOnly-' sub];print(sn,'-dpdf','-bestfit');
            
            %1-d version - not in use, just for viz
            figure;scatter(ems2(pts,2),ems2(pts,3));
            figure;subplot(221);scatter(ems2(pts,2),ems2(pts,4));lsline;
            subplot(222);scatter(ems2(pts,2),ems_norm(pts,4));lsline;
            subplot(223);scatter(ems2(pts,3),ems2(pts,4));lsline;
            subplot(224);scatter(ems2(pts,3),ems_norm(pts,4));lsline;
        end
        %now integrate w/ video info to consider luminance / motion
        for ii=doruns
            switch ii;case 1;trial1=1;case 3;trial1=4;case 5;trial1=7;end %funky way to make data work
            for iii=gset
                gtrialt=trial1+iii-1;
                etmat=ems_norm2d(ems_norm2d(:,5)==gtrialt,:);
                bmat=blinks(blinks(:,2)==gtrialt,1);smat=saccs(saccs(:,2)==gtrialt,1);
                hret_mov{gtrialt,1}.ems_norm2d=etmat;
                if hry(i,2);hr_inst=hret_mov{gtrialt,1}.hr_inst;
                else;hr_inst=NaN(1000,1);end
                g_id=char(gf{g_o(gtrialt)}.gf);%load video information!
                for iiii=1:length(g)
                    if isfield(g{iiii},'gf')
                        if strcmpi(g{iiii}.gf,g_id)
                            g_of_32=iiii;%which game of the 32?
                        end
                    end
                end
                %load luminance info
                fn=[root 'analysis/upper_lum-' num2str(g_of_32) '.mat'];load(fn,'lum_f','mot_f','v');
                lum_f_rs=reshape(lum_f,size(lum_f,1),size(lum_f,2)*size(lum_f,3));
                mot_f_rs=reshape(mot_f,size(mot_f,1),size(mot_f,2)*size(mot_f,3));
                [x1,y1]=meshgrid(1:v.Width,1:v.Height);x1 = x1(:);y1 = y1(:);xyi = [x1 y1];
                xfact=(xmaxvid-xminvid)/v.Width;yfact=(ymaxvid-yminvid)/v.Height;
                secspace_norm=hret_mov{gtrialt,1}.secspace_norm;
                et_inst=nan(length(secspace_norm)-1,1);b_inst=et_inst;
                s_inst=et_inst;loclum_inst=et_inst;locmot_inst=et_inst;
                clear TRsecs;TRsecs(:,1)=linspace(0,length(secspace_norm),length(secspace_norm)+1);
                for iiii=1:length(secspace_norm)-1 %for each TR
                    fill=and(etmat(:,1)>=secspace_norm(iiii),etmat(:,1)<secspace_norm(iiii+1));%find time pts w/in TR
                    et_inst(iiii)=nanmean(etmat(fill,4));%instant et
                    fill2=and(bmat>=secspace_norm(iiii),bmat<secspace_norm(iiii+1));
                    b_inst(iiii)=nansum(fill2); %instant blinks
                    fill2=and(smat>=secspace_norm(iiii),smat<secspace_norm(iiii+1));
                    s_inst(iiii)=nansum(fill2); %instant saccades
                    %local pupil info - grab time bins
                    if iiii<length(TRsecs)-1;tb=(floor(TRsecs(iiii)*v.FrameRate)+1:floor(TRsecs(iiii+1)*v.FrameRate));
                    else;tb=(floor(TRsecs(iiii)*v.FrameRate)+1:length(lum));
                    end
                    %find mode x/y location of eye for this second!
                    if iiii>1 %strange peak-y activity in first second, could be video load
                        xmode=floor(mode(etmat(fill,2)));ymode=floor(mode(etmat(fill,3)));
                        %find actual pixel in video space (video was
                        %reduced to middle 90% of screen)
                        x_adj=floor((xmode-xminvid)/1.2);y_adj=floor((ymode-yminvid)/1.2);
                        z=sqrt((x_adj-x1).^2+(y_adj-y1).^2);
                        zz=find(vis_circ>sqrt((x_adj-x1).^2+(y_adj-y1).^2));
                        loclum_inst(iiii)=nanmean(lum_f_rs(iiii,zz));
                        locmot_inst(iiii)=nanmean(mot_f_rs(iiii,zz));
                    end
                end
                
                prior=nan;prior_ll=nan;
                for ij=1:length(et_inst)
                    if isnan(et_inst(ij))
                        f=find(~isnan(et_inst(ij+1:end)));
                        if ~isempty(f);pp=et_inst(ij+f(1));% 1st non-nan after
                        else;pp=[];end
                        if ij>i;if ~isempty(pp);et_inst(ij)=nanmean([prior pp]);else;et_inst(ij)=prior;end
                        else;et_inst(ij)=pp;end
                    else;prior=et_inst(ij);
                    end
                    if isnan(loclum_inst(ij))
                        f=find(~isnan(loclum_inst(ij+1:end)));
                        if ~isempty(f);pp=loclum_inst(ij+f(1));% 1st non-nan after
                            pm=locmot_inst(ij+f(1));
                        else;pp=[];end
                        if ij>i
                            if ~isempty(pp);loclum_inst(ij)=nanmean([prior_ll pp]);
                                locmot_inst(ij)=nanmean([prior_ml pm]);
                            else;loclum_inst(ij)=prior_ll;locmot_inst(ij)=prior_ml;
                            end
                        else;loclum_inst(ij)=pp;locmot_inst(ij)=pm;
                        end
                    else;prior_ll=loclum_inst(ij);prior_ml=locmot_inst(ij);
                    end
                end
                %log info
                hret_mov{gtrialt,1}.et_inst=et_inst;
                hret_mov{gtrialt,1}.ettpts=etmat(:,1);
                hret_mov{gtrialt,1}.et_std=nanstd(et_inst);
                hret_mov{gtrialt,1}.b_inst=b_inst;
                hret_mov{gtrialt,1}.s_inst=s_inst;
                hret_mov{gtrialt,1}.loclum_inst=loclum_inst;
                hret_mov{gtrialt,1}.locmot_inst=locmot_inst;
                if gs;figure(2);subplot(3,3,gtrialt);plot(secspace_norm(1:end-1),et_inst);
                    title([sub ', movie trial ' (num2str(gtrialt))]);ax=gca;ax.LineWidth=aw;
                    xlabel('Time in movie (s)');ylabel('Pupil area');
                    if gtrialt==tngs;sn=[root 'pics/PupilArea-' sub];ppos;print(sn,'-dpdf','-bestfit');end
                end
            end
        end
    end
    
    %put all data together and put in hret_a for down below
    for ii=doruns
        switch ii;case 1;trial1=1;case 3;trial1=4;case 5;trial1=7;end %funky way to make data work
        for iii=gset
            gtrialt=trial1+iii-1;
            %load up variables
            if hry(i,2);hr_inst=hret_mov{gtrialt,1}.hr_inst;else;hr_inst=NaN(1000,1);end
            if ety(i,2);et_inst=hret_mov{gtrialt,1}.et_inst;
                b_inst=hret_mov{gtrialt,1}.b_inst;
                s_inst=hret_mov{gtrialt,1}.s_inst;
                loclum_inst=hret_mov{gtrialt,1}.loclum_inst;
                locmot_inst=hret_mov{gtrialt,1}.locmot_inst;end
            g_id=char(gf{g_o(gtrialt)}.gf);
            for iiii=1:length(g) %loop through possible games
                if isfield(g{iiii},'gf') %if match
                    if strcmpi(g{iiii}.gf,g_id)
                        %game level traits - not in use
                        fmrig=g{iiii,1}.fmrig;
                        if isfield(hret_mov{gtrialt,1},'rmssd')
                            rmssd(fmrig,i)=hret_mov{gtrialt,1}.rmssd;
                        end
                        if isfield(hret_mov{gtrialt,1},'et_std')
                            et_std(fmrig,i)=hret_mov{gtrialt,1}.et_std;
                        end
                        %TR by TR - possession stuff derived from this
                        pool=[];currpref=0;
                        if isfield(hret_mov{gtrialt,1},'pref')
                            if hret_mov{gtrialt,1}.pref==2;currpref=0;else;currpref=1;end
                        end
                        ctrs=length(g{iiii}.TRsus);
                        shot=g{iiii}.TRtwopt+g{iiii}.TRthreept;%+g{iiii}.TRft;
                        shot=shot+g{iiii}.TRft*2;
                        %put all data into 'pool' variable
                        if ety(i,2);pool=[g{iiii}.TRgr g{iiii}.TRaud ...
                                g{iiii}.TRlum g{iiii}.TRmot ...
                                g{iiii}.TRspeech g{iiii}.TRsus ...
                                g{iiii}.TRsur g{iiii}.TRpn ...
                                -g{iiii}.TRent g{iiii}.TRcp ... %negative entropy is belief-consistency
                                hr_inst(1:ctrs) et_inst(1:ctrs) ...
                                b_inst(1:ctrs) s_inst(1:ctrs) ...
                                loclum_inst(1:ctrs) locmot_inst(1:ctrs) ...
                                shot repmat(currpref,ctrs,1) ...
                                repmat(hret_mov{gtrialt,1}.enjoy,ctrs,1)];
                        else;pool=[g{iiii}.TRgr g{iiii}.TRaud ...
                                g{iiii}.TRlum g{iiii}.TRmot ...
                                g{iiii}.TRspeech g{iiii}.TRsus ...
                                g{iiii}.TRsur g{iiii}.TRpn ...
                                -g{iiii}.TRent g{iiii}.TRcp ...
                                hr_inst(1:length(g{iiii}.TRsus))];
                        end
                        fmrigm=repmat(fmrig,size(pool,1),1);%create fmri game vector
                        pool=[fmrigm pool];%reconfigure 'pool'
                        fit=event_fl(fmrig,1):event_fl(fmrig,2);
                        %carry hret_a down below!!
                        hret_a(fit,:)=pool(1+startcut:length(fit)+startcut,:);
                    end
                end
            end
        end
    end
    if hry(i,2) %if we have heart rate data - put in 'hrs' variable
        clear poss_change
        hr_a_nn=hret_a(~isnan(hret_a(:,12)),:);%grab non-nan in hr data
        hr_a_nn=hr_a_nn(~isnan(hr_a_nn(:,8)),:);%" in surprise data
        X=[ones(size(hr_a_nn,1),1) hr_a_nn(:,2:8)-nanmean(hr_a_nn(:,2:8))];%demean
        y=hr_a_nn(:,12)-nanmean(hr_a_nn(:,12));y_nm=hr_a_nn(:,12);
        b=regress(y,X);hbs_a(i,:)=b(2:end);
        prange=8;j=0;plin=linspace(-prange,prange,prange*2+1);hbl=[-7 -1];
        reg_t=hr_a_nn(:,[2:6]);%other variables to regress
        sus_t=hr_a_nn(:,7);
        for ii=2:size(hr_a_nn,1)
            if and(hr_a_nn(ii,8)>0,hr_a_nn(ii,9)~=hr_a_nn(ii-1,9)) %if any surprise and in bounds
                if and(ii>prange,ii+prange<size(hr_a_nn,1))
                    j=j+1;hrs{i}.sur(j,1)=hr_a_nn(ii,8);
                    hrs{i}.reg_t(j,:)=nanmean(reg_t(ii:ii+2,:));
                    hrs{i}.sus_t(j,:)=nanmean(sus_t(ii+hbl(1):ii+hbl(2),:));
                    poss_change(j,:)=y(ii-prange:ii+prange);
                    hrs{i}.maxhr(j)=nanmean(y(ii:ii+2))-nanmean(y(ii+hbl(1):ii+hbl(2)));%max PA change over baseline
                end
            end
        end
        X=[ones(length(hrs{i}.reg_t),1) [hrs{i}.reg_t hrs{i}.sus_t hrs{i}.sur]-...
            nanmean([hrs{i}.reg_t hrs{i}.sus_t hrs{i}.sur])];
        y=hrs{i}.maxhr';b=regress(y,X);hb_a(i,:)=b(2:end);
        hrs{i}.plin=plin;hrs{i}.tc=nanmean(poss_change);%hrs.tc_ste=stdev(poss_change)/sqrt(j);%time course
    end
    
    %put in 'ets' variable for boundary analyses and 'mass' for R analyses
    if ety(i,2) %if we have eye tracking data 
        clear poss_change poss_change_b poss_change_s
        hr_a_nn=hret_a(~isnan(hret_a(:,13)),:);%grab non-nan in et data
        hr_a_nn=hr_a_nn(~isnan(hr_a_nn(:,8)),:);%" n surprise data
        %model suspense tonic w/ PA, meancenter - not in use
        X=[ones(size(hr_a_nn,1),1) hr_a_nn(:,2:8)-nanmean(hr_a_nn(:,2:8))];
        y=hr_a_nn(:,13)-nanmean(hr_a_nn(:,13));y_nm=hr_a_nn(:,13);
        b=regress(y,X);ebs_a(i,:)=b(2:end);
        XS=[ones(size(hr_a_nn,1),1) hr_a_nn(:,[2:4 16 5 17 6 11 8 10])-nanmean(hr_a_nn(:,[2:4 16 5 17 6 11 8 10]))];
        %plot subject-specific variables
        if gs;pool=[XS(:,2:end) y];
            cats={'Game remaining','Aud env','Global luminance','Local luminance','Global motion','Local motion','Prosody','Court position','Surprise','Belief-consistency','Pupil area'};
            for ii=1:size(pool,2);pool(:,ii)=scale01(pool(:,ii));end
            yl='TR';figure;imagesc(pool);colormap('gray');ylabel(yl);ggg=gca;
            ggg.XTick=1:length(cats);ggg.XTickLabel=cats;
            sn=[root 'pics/PAxTROneSub-' num2str(i)];print(sn,'-dpdf','-bestfit');
        end
        hr_a_nn99=hr_a_nn;hr_a_nn99(hr_a_nn(:,18)>1,18)=0;%undo FT
        X2=[repmat(str2double(sub),size(X,1),1) nan(size(X,1),1) hr_a_nn(:,[2:4 15 5 16 6 11 7:8 10 13:14]) hr_a_nn99(:,18)-...
            nanmean([hr_a_nn(:,[2:4 15 5 16 6 11 7:8 10 13:14]) hr_a_nn99(:,18)])];
        mass0=[mass0;X2 y];
        y_b=hr_a_nn(:,14)-nanmean(hr_a_nn(:,14));
        b=regress(y_b,X);bbs_a(i,:)=b(2:end);
        y_s=hr_a_nn(:,15)-nanmean(hr_a_nn(:,15));
        b=regress(y_s,X);sbs_a(i,:)=b(2:end);
        r=corrcoef(hret_a(:,7),hret_a(:,13),'rows','complete');ersus_a(i)=r(2);
        %surprise w/ phasic PA
        %other variables to regress out-suspense, game remaining, aud, lum
        ebl=[-6 -2];eint=1;ebl2=[-2 -1];
        reg_t=hr_a_nn(:,[2:4 16 5 17 6 11])-nanmean(hr_a_nn(:,[2:4 16 5 17 6 11]));
        sus_t=hr_a_nn(:,7)-nanmean(hr_a_nn(:,7));cgame=hr_a_nn(:,1);j=0;jj=0;
        b_t=hr_a_nn(:,14)-nanmean(hr_a_nn(:,14));b_t_nm=hr_a_nn(:,14);
        s_t=hr_a_nn(:,15)-nanmean(hr_a_nn(:,15));s_t_nm=hr_a_nn(:,15);%NOT mean-centered for graphing
        for ii=2:size(hr_a_nn,1)
            %this method allows for the one possession change that has 0
            %surprise...6_30
            thisg=hr_a_nn(ii,1);fill=find(hr_a_nn(:,1)==thisg);gmax=max(fill);
            proc=0;
            if hr_a_nn(ii,9)-hr_a_nn(ii-1,9)==1;proc=1;
            elseif and(isnan(hr_a_nn(ii,9)),~isnan(hr_a_nn(ii-1,9)));proc=1;end
            if proc
                if and(ii>ebl(1),ii+eint<=gmax)%size(hr_a_nn,1)) %eint was 6_30, prange tried on 7_4
                    j=j+1;ets{i}.sur(j,1)=hr_a_nn(ii,8);
                    ets{i}.bc(j,1)=hr_a_nn(ii,10);
                    ets{i}.cgame(j,1)=hr_a_nn(ii,1);
                    ets{i}.sur_pre(j,1)=nan;ets{i}.maxpac_pre(j,1)=nan;
                    ets{i}.reg_t(j,1)=nanmean(reg_t(ii:ii+eint,1));%GR
                    ets{i}.reg_t(j,2:size(reg_t,2)-1)=nanmean(reg_t(ii:ii+eint,2:size(reg_t,2)-1))-nanmean(reg_t(ii+ebl(1):ii+ebl(2),2:size(reg_t,2)-1));
                    ets{i}.reg_t(j,size(reg_t,2))=nanmean(reg_t(ii:ii+eint,size(reg_t,2)));%Court position
                    ets{i}.blinks(j,:)=nanmean(b_t(ii:ii+eint))-nanmean(b_t(ii+ebl2(1):ii+ebl2(2)));
                    ets{i}.saccs(j,:)=nanmean(s_t(ii:ii+eint))-nanmean(s_t(ii+ebl2(1):ii+ebl2(2)));
                    ets{i}.sus_t(j,:)=sus_t(ii+ebl(2));
                    ets{i}.shot(j,1)=hr_a_nn(ii,18);
                    if and(ii>prange,ii+prange<gmax)
                        poss_change(j,:)=y(ii-prange:ii+prange);
                        poss_change_b(j,:)=b_t_nm(ii-prange:ii+prange);
                        poss_change_s(j,:)=s_t_nm(ii-prange:ii+prange);
                    end
                    ets{i}.maxpac(j)=nanmax(y(ii:ii+eint))-nanmean(y(ii+ebl(1):ii+ebl(2)));%max PA change over baseline
                    %see if match in poss matrix!
                    row=find(and(hr_a_nn(ii-1,1)==p_fl(:,1),hr_a_nn(ii-1,9)==p_fl(:,3)));
                    if ~isempty(row);jj=jj+1;ets{i}.p_fl_rows(jj,1)=j;ets{i}.p_fl_rows(jj,2)=row;end
                    if j>1;if ets{i}.cgame(j,1)==ets{i}.cgame(j-1,1);if row-1==prevrow
                                ets{i}.sur_pre(j,1)=ets{i}.sur(j-1,1);
                                ets{i}.maxpac_pre(j)=ets{i}.maxpac(j-1);end;end;end
                    prevrow=row;
                end
            end
        end
        if size(poss_change,1)<size(ets{i}.sur) %fill in missing rows @ end
            poss_change(size(poss_change,1)+1:size(ets{i}.sur))=nan;
            poss_change_b(size(poss_change_b,1)+1:size(ets{i}.sur))=nan;
            poss_change_s(size(poss_change_s,1)+1:size(ets{i}.sur))=nan;
        end
        X=[ones(length(ets{i}.reg_t),1) ... 
            [ets{i}.reg_t ets{i}.sus_t ets{i}.sur_pre ets{i}.sur ets{i}.bc ets{i}.maxpac_pre ets{i}.shot poss_change]-...
            nanmean([ets{i}.reg_t ets{i}.sus_t ets{i}.sur_pre ets{i}.sur ets{i}.bc ets{i}.maxpac_pre ets{i}.shot poss_change])];
        y=ets{i}.maxpac'-nanmean(ets{i}.maxpac)';
        y_b_s=[ets{i}.blinks ets{i}.saccs]-nanmean([ets{i}.blinks ets{i}.saccs]);
        X=[ones(length(ets{i}.reg_t),1) ... 
            [ets{i}.reg_t ets{i}.sus_t ets{i}.sur_pre ets{i}.sur ets{i}.bc ets{i}.maxpac_pre ets{i}.shot poss_change]-...
            nanmean([ets{i}.reg_t ets{i}.sus_t ets{i}.sur_pre ets{i}.sur ets{i}.bc ets{i}.maxpac_pre ets{i}.shot poss_change])];
        %time courses
        ets{i}.plin=plin;ets{i}.tc=nanmean(poss_change);ets{i}.poss_change=poss_change;
        ets{i}.tc_b=nanmean(poss_change_b);ets{i}.poss_change_b=poss_change_b;
        ets{i}.tc_s=nanmean(poss_change_s);ets{i}.poss_change_s=poss_change_s;
        %memory correlation analysis
        if i<=size(rec_all_poss_xsub,1)
            parows=ets{i}.p_fl_rows(:,1);memrows=ets{i}.p_fl_rows(:,2);
            X2=[repmat(str2double(sub),size(memrows,1),1) memrows X(parows,2:end) y_b_s(parows,:) y(parows)];
            y2=rec_all_poss_xsub(i,memrows)';
            mass=[mass;X2 y2];
        end
    end
    save phys_mov hret_mov hret_a;
    save('MSMEhr2.mat','MSTR','METR','RSTR','RETR','hr_inst','-v7');
    hret_a_a{i}.hret_a=hret_a;
end
cd([root 'analysis/']);
clear ems ems2 ems_norm ems_norm2d ems_smooth etmat etmat2 maxhr_i xyi y1
if svHR;save AllXSubHR;end %load AllXSubHR 
%% time course analyses
% integrate new regressors and use poss_change to calculate PAC @ each pt
figure;j=0;adj=0.2;
for i=1:nS
    if ety(i,2)
        f=1:length(ets{i}.plin)-1;
        j=j+1;plot(ets{i}.plin(f)+0.5,ets{i}.tc(f),'--k','LineWidth',0.25);hold on;
        tcs(j,:)=ets{i}.tc;
        shots=ets{i}.shot==1;nshots=ets{i}.shot==0;fts=ets{i}.shot==2;
        tcs_shot(j,:)=mean(ets{i}.poss_change(shots,:),1);
        tcs_nshot(j,:)=mean(ets{i}.poss_change(nshots,:),1);
        tcs_ft(j,:)=mean(ets{i}.poss_change(fts,:),1);
    end
end
errorbar(ets{i}.plin(f)+0.5,mean(tcs(:,f)),std(tcs(:,f))/sqrt(j),'k','LineWidth',4);
set(0,'DefaultAxesFontSize',25);gg=gca;gg.LineWidth=2;
xlabel('Time (relative to possession change)');ylabel('Pupil area (z-scored)');box off;
sn=[root 'pics/SurpriseVPhasicPupilAreaTimeCourse'];print(sn,'-dpdf','-bestfit');
int1=ebl;int2=[0 eint];
[~,pPA]=ttest(mean(tcs(:,prange+1+int1(1):prange+1+int1(2)),2),...
    mean(tcs(:,prange+1+int2(1):prange+1+int2(2)),2))
figure;errorbar(ets{i}.plin(f)+0.5,mean(tcs_shot(:,f)),std(tcs_shot(:,f))/sqrt(j),'b','LineWidth',3);
hold on;errorbar(ets{i}.plin(f)+0.5+adj,mean(tcs_nshot(:,f)),std(tcs_nshot(:,f))/sqrt(j),'r','LineWidth',3);
errorbar(ets{i}.plin(f)+0.5+adj*2,mean(tcs_ft(:,f)),std(tcs_ft(:,f))/sqrt(j),'c','LineWidth',3);
xlabel('Time (relative to possession change)');ylabel('Pupil area (z-scored)');legend('Shot','Non-shot','FT');legend boxoff;
sn=[root 'pics/SurpriseVPhasicPupilAreaTimeCourseShotVN'];print(sn,'-dpdf','-bestfit');
shotval=mean(tcs_shot(:,prange+1+int1(1):prange+1+int1(2)),2)-...
    mean(tcs_shot(:,prange+1+int2(1):prange+1+int2(2)),2);
nshotval=mean(tcs_nshot(:,prange+1+int1(1):prange+1+int1(2)),2)-...
    mean(tcs_nshot(:,prange+1+int2(1):prange+1+int2(2)),2);
[~,pPA_shot]=ttest(shotval,nshotval)

figure;jj=0;
for i=1:nS
    if hry(i,2)
        f=1:length(hrs{i}.plin)-1;
        jj=jj+1;plot(hrs{i}.plin(f)+0.5,hrs{i}.tc(f),'--k','LineWidth',0.5);hold on;
        htcs(jj,:)=hrs{i}.tc;
    end
end
errorbar(hrs{i}.plin(f)+0.5,mean(htcs(:,f)),std(htcs(:,f))/sqrt(jj),'k','LineWidth',3);
xlabel('Time (relative to possession change)');ylabel('Heart rate (bpm)');
sn=[root 'pics/SurpriseVPhasicHRTimeCourse'];print(sn,'-dpdf','-bestfit');

figure;j=0;
for i=1:nS
    if ety(i,2)
        f=1:length(ets{i}.plin)-1;
        j=j+1;plot(ets{i}.plin(f)+0.5,ets{i}.tc_b(f),'--k','LineWidth',0.5);hold on;
        tcs_b(j,:)=ets{i}.tc_b;
        shots=ets{i}.shot==1;nshots=ets{i}.shot==0;fts=ets{i}.shot==2;
        tcs_b_shot(j,:)=mean(ets{i}.poss_change_b(shots,:),1);
        tcs_b_nshot(j,:)=mean(ets{i}.poss_change_b(nshots,:),1);
        tcs_b_ft(j,:)=mean(ets{i}.poss_change_b(fts,:),1);
    end
end
errorbar(ets{i}.plin(f)+0.5,mean(tcs_b(:,f)),std(tcs_b(:,f))/sqrt(j),'k','LineWidth',3);
xlabel('Time (relative to possession change)');ylabel('Blink rate');
sn=[root 'pics/SurpriseVBlinksTimeCourse'];print(sn,'-dpdf','-bestfit');
int1=ebl2;int2=[1 2];
[~,pB]=ttest(mean(tcs_b(:,prange+1+int1(1):prange+1+int1(2)),2),...
    mean(tcs_b(:,prange+1+int2(1):prange+1+int2(2)),2))
figure;errorbar(ets{i}.plin(f)+0.5,mean(tcs_b_shot(:,f)),std(tcs_b_shot(:,f))/sqrt(j),'b','LineWidth',3);
hold on;errorbar(ets{i}.plin(f)+0.5+adj,mean(tcs_b_nshot(:,f)),std(tcs_b_nshot(:,f))/sqrt(j),'r','LineWidth',3);
errorbar(ets{i}.plin(f)+0.5+adj*2,mean(tcs_b_ft(:,f)),std(tcs_b_ft(:,f))/sqrt(j),'c','LineWidth',3);
xlabel('Time (relative to possession change)');ylabel('Blink rate');legend('Shot','Non-shot','FT');legend boxoff;
sn=[root 'pics/SurpriseVBlinksTimeCourseShotVN'];print(sn,'-dpdf','-bestfit');
shotval=mean(tcs_b_shot(:,prange+1+int1(1):prange+1+int1(2)),2)-...
    mean(tcs_b_shot(:,prange+1+int2(1):prange+1+int2(2)),2);
nshotval=mean(tcs_b_nshot(:,prange+1+int1(1):prange+1+int1(2)),2)-...
    mean(tcs_b_nshot(:,prange+1+int2(1):prange+1+int2(2)),2);
[~,pB_shot]=ttest(shotval,nshotval)

figure;j=0;
for i=1:nS
    if ety(i,2)
        f=1:length(ets{i}.plin)-1;
        j=j+1;plot(ets{i}.plin(f)+0.5,ets{i}.tc_s(f),'--k','LineWidth',0.5);hold on;
        tcs_s(j,:)=ets{i}.tc_s;
        shots=ets{i}.shot==1;nshots=ets{i}.shot==0;fts=ets{i}.shot==2;
        tcs_s_shot(j,:)=mean(ets{i}.poss_change_s(shots,:),1);
        tcs_s_nshot(j,:)=mean(ets{i}.poss_change_s(nshots,:),1);
        shots=find(ets{i}.shot==1);nshots=find(ets{i}.shot==0);
        tcs_s_ft(j,:)=mean(ets{i}.poss_change_s(fts,:),1);
    end
end
errorbar(ets{i}.plin(f)+0.5,mean(tcs_s(:,f)),std(tcs_s(:,f))/sqrt(j),'k','LineWidth',3);
xlabel('Time (relative to possession change)');ylabel('Saccade rate');
sn=[root 'pics/SurpriseVSaccadesTimeCourse'];print(sn,'-dpdf','-bestfit');
int1=-2;int2=-1;
[~,pS]=ttest(tcs_s(:,prange+1+int1(1)),tcs_s(:,prange+1+int2(1)))
figure;errorbar(ets{i}.plin(f)+0.5,mean(tcs_s_shot(:,f)),std(tcs_s_shot(:,f))/sqrt(j),'b','LineWidth',3);
hold on;errorbar(ets{i}.plin(f)+0.5+adj,mean(tcs_s_nshot(:,f)),std(tcs_s_nshot(:,f))/sqrt(j),'r','LineWidth',3);
errorbar(ets{i}.plin(f)+0.5+adj*2,mean(tcs_s_ft(:,f)),std(tcs_s_ft(:,f))/sqrt(j),'c','LineWidth',3);
xlabel('Time (relative to possession change)');ylabel('Saccade rate');legend('Shot','Non-shot','FT');legend boxoff;
sn=[root 'pics/SurpriseVSaccadesTimeCourseShotVN'];print(sn,'-dpdf','-bestfit');
shotval=tcs_s_shot(:,prange+1+int1(1))-tcs_s_shot(:,prange+1+int2(1));
nshotval=tcs_s_nshot(:,prange+1+int1(1))-tcs_s_nshot(:,prange+1+int2(1));
[~,pS_shot]=ttest(shotval,nshotval)
%% full game analyses - not in use
ff=linspace(0,100,101);ff=ff(1:end-1); %minimum length of possession
for h=1:tngs
    prange=event_fl(h,1):event_fl(h,2);prs=event_fl(h,2)-event_fl(h,1)+1;j=0;jj=0;%clear hgtcs etgtcs
    for i=1:nS
        if hry(i,2)
            f=linspace(0,100,prs);jj=jj+1;
            for ii=1:100
                fill=and(f>=ii-1,f<ii);
                hgtcs(jj,ii)=mean(hret_a_a{i}.hret_a(prange(fill),12));
            end
        end
        if ety(i,2)
            f=linspace(0,100,prs);
            j=j+1;
            for ii=1:100
                fill=and(f>=ii-1,f<ii);
                etgtcs(jj,ii)=mean(hret_a_a{i}.hret_a(prange(fill),13));
            end
        end
    end
    hgtcs_a(h,:,:)=hgtcs;etgtcs_a(h,:,:)=etgtcs;
    co=gamebinsurmean(h)/max(gamebinsurmean);
    figure(11);plot(ff+0.5,nanmean(hgtcs),'color',[co co co],'LineWidth',0.5);hold on;
    ylabel('Heart rate (bpm)');xlabel('Percent of time into game');
    figure(12);plot(ff+0.5,nanmean(etgtcs),'color',[co co co],'LineWidth',0.5);hold on;
    ylabel('Pupil area (z-scored)');xlabel('Percent of time into game');
end

figure(11);plot(ff+0.5,squeeze(nanmean(nanmean(hgtcs_a,2),1)),'k','LineWidth',3);
set(0,'DefaultAxesFontSize',25);gg=gca;gg.LineWidth=2;box off;
sn=[root 'pics/HRFullGame'];print(sn,'-dpdf','-bestfit');
figure(12);plot(ff+0.5,squeeze(nanmean(nanmean(etgtcs_a,2),1)),'k','LineWidth',3);
gg=gca;gg.LineWidth=2;box off;sn=[root 'pics/ETFullGame'];print(sn,'-dpdf','-bestfit');

%% full possession (0-100%) analysis - not in use
sr=1;%1=surprise,2=suspense,3=sur X sus
possthresh=10;pf=100/possthresh;perc=20;surcol=6;suscol=5;%surcol = 6 for pre
hisus=prctile(p_fl(:,suscol),100-perc);losus=prctile(p_fl(:,suscol),perc);
hipsur=prctile(p_fl(:,surcol),100-perc);lopsur=prctile(p_fl(:,surcol),perc);
hj=0;hj2=0;hj3=0;hj4=0;hj5=0;hj2_4=0;hj2_5=0;hj3_4=0;hj3_5=0;prange=7;fullposs=1;
clear hptcs etptcs hptcs_a etptcs_a etptcs_hpsur_a etptcs_lpsur_a etptcs_hsus_a etptcs_lsus_a ...
    etptcs_hpsur_hsus_a etptcs_hpsur_lsus_a etptcs_lpsur_hsus_a etptcs_lpsur_lsus_a;
for h=1:size(p_fl,1)
    proc=0;
    if fullposs;ff=linspace(0,100,pf+1);ff=ff(1:end-1);proc=1;
        if h<size(p_fl,1);prange2=p_fl(h,2):p_fl(h+1,2);prs=p_fl(h+1,2)-p_fl(h,2)+1;
        else;prange2=p_fl(h,2):event_fl(end,2);prs=event_fl(end,2)-p_fl(h,2)+1;end
    else;if p_fl(h,4)>prange %fixed from row 2...
            if h~=size(p_fl,1)
                ff=(-prange:prange)+0.5;ff=ff(1:end-1);prs=p_fl(h+1,2)-p_fl(h,2)+1;
                prange2=p_fl(h,2)-prange:p_fl(h,2)+prange-1;proc=1;
            end
        end
    end
    if proc
        if prs>possthresh;j=0;jj=0;hj=hj+1;
            for i=1:nS
                if hry(i,2)
                    if fullposs;f=linspace(0,100,prs);jj=jj+1;
                        for ii=1:pf
                            fill=and(f>(ii-1)*pf,f<=ii*pf);
                            hptcs(jj,ii)=nanmean(hret_a_a{i}.hret_a(prange2(fill),12));%%%%
                        end
                    else;jj=jj+1;hptcs(jj,:)=hret_a_a{i}.hret_a(prange2,12);end
                end
                if ety(i,2)
                    if fullposs;f=linspace(0,100,prs);
                        j=j+1;
                        for ii=1:pf
                            fill=and(f>(ii-1)*pf,f<=ii*pf);
                            etptcs(j,ii)=nanmean(hret_a_a{i}.hret_a(prange2(fill),13));
                        end
                    else;j=j+1;etptcs(j,:)=hret_a_a{i}.hret_a(prange2,13);end
                end
            end
            hptcs_a(hj,:,:)=hptcs;etptcs_a(hj,:,:)=etptcs;
            if p_fl(h,surcol)>hipsur;hj2=hj2+1;etptcs_hpsur_a(hj2,:,:)=etptcs;
                if p_fl(h,suscol)>hisus;hj2_4=hj2_4+1;etptcs_hpsur_hsus_a(hj2_4,:,:)=etptcs;
                elseif p_fl(h,suscol)<losus;hj2_5=hj2_5+1;etptcs_hpsur_lsus_a(hj2_5,:,:)=etptcs;
                end
            elseif p_fl(h,surcol)<lopsur;hj3=hj3+1;etptcs_lpsur_a(hj3,:,:)=etptcs;
                if p_fl(h,suscol)>hisus;hj3_4=hj3_4+1;etptcs_lpsur_hsus_a(hj3_4,:,:)=etptcs;
                elseif p_fl(h,suscol)<losus;hj3_5=hj3_5+1;etptcs_lpsur_lsus_a(hj3_5,:,:)=etptcs;
                end
            end
            %suspense separately
            if p_fl(h,suscol)>hisus;hj4=hj4+1;etptcs_hsus_a(hj4,:,:)=etptcs;
            elseif p_fl(h,suscol)<losus;hj5=hj5+1;etptcs_lsus_a(hj5,:,:)=etptcs;
            end
        end
    end
end
lw=3;
hptcs_a=squeeze(nanmean(hptcs_a,1));etptcs_a=squeeze(nanmean(etptcs_a,1));
if sr==1;etptcs_hpsur_a=squeeze(nanmean(etptcs_hpsur_a,1));etptcs_lpsur_a=squeeze(nanmean(etptcs_lpsur_a,1));
elseif sr==2;etptcs_hsus_a=squeeze(nanmean(etptcs_hsus_a,1));etptcs_lsus_a=squeeze(nanmean(etptcs_lsus_a,1));
elseif sr==3;etptcs_hpsur_hsus_a=squeeze(nanmean(etptcs_hpsur_hsus_a,1));etptcs_hpsur_lsus_a=squeeze(nanmean(etptcs_hpsur_lsus_a,1));
    etptcs_lpsur_hsus_a=squeeze(nanmean(etptcs_lpsur_hsus_a,1));etptcs_lpsur_lsus_a=squeeze(nanmean(etptcs_lpsur_lsus_a,1));
end
figure;hold on;
if fullposs;add=5;else;add=0;end
if sr==1;errorbar(ff+add,squeeze(nanmean(etptcs_hpsur_a,1)),squeeze(nanstd(etptcs_hpsur_a,1))/sqrt(nS),'b','LineWidth',lw);
    errorbar(ff+add+adj,squeeze(nanmean(etptcs_lpsur_a,1)),squeeze(nanstd(etptcs_lpsur_a,1))/sqrt(nS),'r','LineWidth',lw);
    if fullposs;legend('Preceded by high surprise','Preceded by low surprise');xlabel('Percent of time into possession');
    else;legend('High surprise','Low surprise');xlabel('Time (relative to possession change');end
elseif sr==2;errorbar(ff+add,squeeze(nanmean(etptcs_hsus_a,1)),squeeze(nanstd(etptcs_hsus_a,1))/sqrt(nS),'b','LineWidth',lw);
    errorbar(ff+add+adj,squeeze(nanmean(etptcs_lsus_a,1)),squeeze(nanstd(etptcs_lsus_a,1))/sqrt(nS),'r','LineWidth',lw);
    if fullposs;xlabel('Percent of time into possession');else;xlabel('Time (relative to possession change');end
    legend('High suspense','Low suspense');
elseif sr==3;errorbar(ff+add,squeeze(nanmean(etptcs_hpsur_hsus_a,1)),squeeze(nanstd(etptcs_hpsur_hsus_a,1))/sqrt(nS),'b','LineWidth',lw);
    errorbar(ff+add+adj,squeeze(nanmean(etptcs_hpsur_lsus_a,1)),squeeze(nanstd(etptcs_hpsur_lsus_a,1))/sqrt(nS),'r','LineWidth',lw);
    errorbar(ff+add,squeeze(nanmean(etptcs_lpsur_hsus_a,1)),squeeze(nanstd(etptcs_lpsur_hsus_a,1))/sqrt(nS),'k','LineWidth',lw);
    errorbar(ff+add+adj,squeeze(nanmean(etptcs_lpsur_lsus_a,1)),squeeze(nanstd(etptcs_lpsur_lsus_a,1))/sqrt(nS),'m','LineWidth',lw);
    if fullposs;xlabel('Percent of time into possession');else;xlabel('Time (relative to possession change');end
    legend('High sur, high sus','High sur, low sus','Low sur, high sus','Low sur, low sus');
end;ylabel('Pupil area (z-scored)');
if sr<3
    if fullposs;firstseg=1:2;[~,firstseg_p]=ttest(mean(etptcs_hpsur_a(:,firstseg),2),mean(etptcs_lpsur_a(:,firstseg),2))
    else;firstseg=8:9;[~,firstseg_p]=ttest(mean(etptcs_hpsur_a(:,firstseg),2)-mean(etptcs_hpsur_a(:,2:6),2),...
            mean(etptcs_lpsur_a(:,firstseg),2)-mean(etptcs_lpsur_a(:,2:6),2))
    end
end
sn=[root 'pics/SSVPhasicETPossession-' num2str(sr) '-' num2str(fullposs)];print(sn,'-dpdf','-bestfit');
%% export to R
Sub_num=mass0(:,1);GR=mass0(:,3);Aud=mass0(:,4);Lum=mass0(:,5);
LocLum=mass0(:,6);Mot=mass0(:,7);LocMot=mass0(:,8);Pros=mass0(:,9);
Courtpos=mass0(:,10);Sus=mass0(:,11);Sur=mass0(:,12);BC=mass0(:,13);
Blinks=mass0(:,14);Saccades=mass0(:,15);Shot=mass0(:,16);PA=mass0(:,17);
T=table(Sub_num,GR,Aud,Lum,LocLum,Mot,LocMot,Pros,Sus,Sur,Blinks,Saccades,Shot,PA);
writetable(T,[root 'Rstuff/PA.csv'])%
col=1;Sub_num=mass(:,col);col=col+1;P_num=mass(:,col);col=col+1;
GR=mass(:,col);col=col+1;Aud=mass(:,col);col=col+1;
Lum=mass(:,col);col=col+1;LocLum=mass(:,col);col=col+1;
Mot=mass(:,col);col=col+1;LocMot=mass(:,col);col=col+1;
Pros=mass(:,col);col=col+1;Courtpos=mass(:,col);col=col+1;
Sus=mass(:,col);col=col+1;Sur_pre=mass(:,col);col=col+1;
Sur=mass(:,col);col=col+1;BC=mass(:,col);col=col+1;
PAC_pre=mass(:,col);col=col+1;Shot=mass(:,col);col=col+1;
PAn8=mass(:,col);col=col+1;PAn7=mass(:,col);col=col+1;PAn6=mass(:,col);col=col+1;
PAn5=mass(:,col);col=col+1;PAn4=mass(:,col);col=col+1;PAn3=mass(:,col);col=col+1;
PAn2=mass(:,col);col=col+1;PAn1=mass(:,col);col=col+1;PA0=mass(:,col);col=col+1;
PA1=mass(:,col);col=col+1;PA2=mass(:,col);col=col+1;PA3=mass(:,col);col=col+1;
PA4=mass(:,col);col=col+1;PA5=mass(:,col);col=col+1;PA6=mass(:,col);col=col+1;
PA7=mass(:,col);col=col+1;PA8=mass(:,col);col=col+1;%col=34;
Blinks=mass(:,col);col=col+1;Saccades=mass(:,col);col=col+1;
PAC=mass(:,col);col=col+1;Mem=mass(:,col);col=col+1;

T=table(Sub_num,GR,P_num,Aud,Lum,LocLum,Mot,LocMot,Pros,Courtpos,Sus,...
    Sur_pre,Sur,BC,PAC_pre,Shot,PAn8,PAn7,PAn6,PAn5,PAn4,PAn3,PAn2,PAn1,...
    PA0,PA1,PA2,PA3,PA4,PA5,PA6,PA7,PA8,Blinks,Saccades,PAC,Mem);
writetable(T,[root 'Rstuff/PAMem.csv'])%
nPoss=size(p_fl,1);
massA=NaN(nS*nPoss,size(mass,2));%convert to nS*nPoss space for alignment w/ other variables in R
for s=1:nS
    if ety(s,2)
        fill=find(mass(:,1)==s);%rows of mass to use
        fill2=NaN(nPoss,size(mass,2));fill2(mass(fill,2),:)=mass(fill,:);
        rows=(s-1)*nPoss+1:s*nPoss;massA(rows,:)=fill2;
    end
end
col=1;Sub_num=massA(:,col);col=col+1;P_num=massA(:,col);col=col+1;
GR=massA(:,col);col=col+1;Aud=massA(:,col);col=col+1;
Lum=massA(:,col);col=col+1;LocLum=massA(:,col);col=col+1;
Mot=massA(:,col);col=col+1;LocMot=massA(:,col);col=col+1;
Pros=massA(:,col);col=col+1;Courtpos=massA(:,col);col=col+1;
Sus=massA(:,col);col=col+1;Sur_pre=massA(:,col);col=col+1;
Sur=massA(:,col);col=col+1;BC=massA(:,col);col=col+1;
PAC_pre=massA(:,col);col=col+1;Shot=massA(:,col);col=col+1;
PAn8=massA(:,col);col=col+1;PAn7=massA(:,col);col=col+1;PAn6=massA(:,col);col=col+1;
PAn5=massA(:,col);col=col+1;PAn4=massA(:,col);col=col+1;PAn3=massA(:,col);col=col+1;
PAn2=massA(:,col);col=col+1;PAn1=massA(:,col);col=col+1;PA0=massA(:,col);col=col+1;
PA1=massA(:,col);col=col+1;PA2=massA(:,col);col=col+1;PA3=massA(:,col);col=col+1;
PA4=massA(:,col);col=col+1;PA5=massA(:,col);col=col+1;PA6=massA(:,col);col=col+1;
PA7=massA(:,col);col=col+1;PA8=massA(:,col);col=col+1;%col=34;
Blinks=massA(:,col);col=col+1;Saccades=massA(:,col);col=col+1;
PAC=massA(:,col);col=col+1;Mem=massA(:,col);col=col+1;
T=table(Sub_num,GR,P_num,Aud,Lum,LocLum,Mot,LocMot,Pros,Courtpos,Sus,...
    Sur_pre,Sur,BC,PAC_pre,Shot,PAn8,PAn7,PAn6,PAn5,PAn4,PAn3,PAn2,PAn1,...
    PA0,PA1,PA2,PA3,PA4,PA5,PA6,PA7,PA8,Blinks,Saccades,PAC,Mem);
writetable(T,[root 'Rstuff/PAMemA.csv'])%

%average across possession level
for i=1:nPoss
    fill=find(i==P_num);PACPoss(i,1)=nanmean(PAC(fill));
    PACPoss_pre(i,1)=nanmean(PAC_pre(fill));
    MemPoss(i,1)=nanmean(Mem(fill));
    SurPoss(i,1)=nanmean(Sur(fill));
end
T=table(PACPoss_pre,PACPoss,MemPoss,SurPoss);writetable(T,[root 'Rstuff/PAPossMem.csv']);
save PACPoss PACPoss PACPoss_pre;

%create table with rmssd and ETV @ game level for exporting to R!
Sub_num=[];HRV=[];ETV=[];ETC=nan(tngs*nS,1);G_num=[];
for i=1:nS
    Sub_num=[Sub_num;repmat(str2num(char(subs(i))),tngs,1)];
    HRV=[HRV;rmssd(:,i)];ETV=[ETV;et_std(:,i)];
    if ety(i,2)==1%find game
        for ii=1:tngs
            g_rows=find(p_fl(:,1)==ii);%game rows
            et_rows=find(and(ets{i}.p_fl_rows(:,2)>=g_rows(1),ets{i}.p_fl_rows(:,2)<=g_rows(end)));
            if ~isempty(et_rows);ETC((i-1)*tngs+ii,1)=mean(ets{i}.maxpac(et_rows));end
        end
    end
end
HRV(HRV==0)=nan;G_num=repmat(1:tngs,1,nS)';
T=table(Sub_num,G_num,HRV,ETV,ETC);writetable(T,[root 'Rstuff/HRETGame.csv'])%

%for showing surprise vs. PA regression - visualization only
sampsubr=find(mass(:,1)==4);pool=[mass(sampsubr,[3 5:8 4 9:10 13 36])];
cats={'Game remaining','Global luminance','Local luminance','Global video motion','Local video motion',' Auditory envelope','Prosody (F0)','Court position','Surprise','Pupil area change'};%'Belief-consistency of surprise',
for ii=1:size(pool,2);pool(:,ii)=scale01(pool(:,ii));end
yl='Possession';figure;set(gcf,'position',[0 0 1200 1200]);
imagesc(pool);colormap('gray');ylabel(yl);ggg=gca;
ggg.XTick=1:length(cats);ggg.XTickLabel=cats;xtickangle(45);
sn=[root 'pics/PAVSurRegressImage'];print(sn,'-dpdf','-bestfit');

%game-level traits - not in use
fill=find(hry(:,2)>0);
for i=1:size(rmssd,1);m(i)=mean(rmssd(i,fill));sem(i)=std(rmssd(i,fill))/sqrt(length(fill));end;
figure;errorbar(m,sem);xlabel('Game');ylabel('Heart rate variability (RMSSD)');
for i=1:length(fill);r=corrcoef(gamebinsurmean,rmssd(:,fill(i)));rs(i)=r(2);end
[~,p]=ttest(rs',zeros(length(rs),1))

fill=find(ety(:,2)>0);
for i=1:size(et_std,1);m(i)=mean(et_std(i,fill));sem(i)=std(et_std(i,fill))/sqrt(length(fill));end;
figure;errorbar(m,sem);xlabel('Game');ylabel('Pupil area variability');
for i=1:length(fill);r=corrcoef(gamebinsurmean,et_std(:,fill(i)));rs(i)=r(2);end
[~,p]=ttest(rs',zeros(length(rs),1))

%% surprise - PAC by possession analysis - not in use
x=SurPoss;y=PACPoss;figure;scatter(x,y);
PAC_r=corrcoef(x,y,'rows','complete');PAC_r=PAC_r(2)
PAC_rS=corr(x,y,'Type','Spearman','rows','complete');PAC_rS

iters=10000;%scramble, preserving game structure
scram_meth=3;%1=scramble all,2=scramble game order,keep poss order,3=circ shift w/in game
for i=1:iters
    sur_i=[];sur_i2=[];sur_i_part=[];
    switch scram_meth;case 1;sur_i=x;
        case 2;fill=randperm(tngs);sur_i=[];
            for ii=1:tngs;f2=find(fill(ii)==p_fl(:,1));sur_i=[sur_i;x(f2)];end
        case 3
            fill=randperm(tngs);
            for ii=1:tngs
                f2=find(ii==p_fl(:,1));
                proc=0;
                while proc==0;proc=1;seed=randperm(length(f2));seed=seed(1);
                    if seed<3;proc=0;elseif seed==length(f2);proc=0;end
                end
                f2=[f2(seed:end);f2(1:seed-1)];sur_i=[sur_i;x(f2)];
            end
    end
    r=corrcoef(sur_i,y,'rows','complete');PAC_rs(i)=r(2);
    r=corr(sur_i,y,'Type','Spearman','rows','complete');PAC_rsS(i)=r;
end
PAC_r
PAC_r_p=1-length(find(PAC_r>PAC_rs))/iters
PAC_rS %spearman
PAC_rS_p=1-length(find(PAC_rS>PAC_rsS))/iters

%% bootstrapping analysis - not in the paper but for reviewer point
Sub_num=massA(:,1);iters=1000;
d=ety(ety(:,2)==1,1);dl=length(d);bs_dist=zeros(iters,1);
for h=1:iters
	subset=datasample(d,dl); %subset of subjects
    x2=[];y2=[];p2=[];
    %concatenate dataset using these subjects
    for ii=1:dl
        %find subject rows
        fill=find(Sub_num==subset(ii));
        p2=[p2;P_num(fill)];
        x2=[x2;Sur(fill)];
        y2=[y2;PAC(fill)];
    end
    %now find average for each possession
    for ii=1:nPoss
        fill=find(ii==p2);
        SurPoss2(ii,1)=nanmean(x2(fill));
        PACPoss2(ii,1)=nanmean(y2(fill));
    end
    %run correlation - this is the ground truth value for this bootstrap!
    true_r=corrcoef(SurPoss2,PACPoss2,'rows','complete');
        
    for i=1:iters
        sur_i=[];
        switch scram_meth;case 1;sur_i=SurPoss2(randperm(nPoss));
            case 2;fill=randperm(tngs);sur_i=[];
                for ii=1:tngs;f2=find(fill(ii)==p_fl(:,1));sur_i=[sur_i;SurPoss2(f2)];end
            case 3
                fill=randperm(tngs);
                for ii=1:tngs
                    f2=find(ii==p_fl(:,1));
                    proc=0;
                    while proc==0;proc=1;seed=randperm(length(f2));seed=seed(1);
                        if seed<3;proc=0;elseif seed==length(f2);proc=0;end
                    end
                    f2=[f2(seed:end);f2(1:seed-1)];sur_i=[sur_i;SurPoss2(f2)];
                end
        end
        %run correlation for this bootstrap's particular shift
        r=corrcoef(sur_i,PACPoss2,'rows','complete');y2_rs(i)=r(2);
    end
    %find z-score of the true vs null distribution
    tog=[true_r(2) y2_rs];togz=zscore(tog);
    bs_dist(h)=togz(1); %assign to bootstrap distribution
end

%plot histogram of z-scores
figure;histogram(bs_dist);ylabel('# instances');xlabel('z (PAC and Sur)');
mean(bs_dist)
std(bs_dist)
pval=((iters-length(find(bs_dist>0)))/iters)*2 %p value
