%{
init_ss.m 06.11.18
Run this program at the start of experiment, filling in info in box.
%}
%% initiating everything...
g_o_m{(phase-1)*ngs+1}.init_time=GetSecs();
g_o_m{(phase-1)*ngs+1}.init_time_r=clock;
PsychDefaultSetup(2);bgColor=[128 128 128];
Screen('Preference','SkipSyncTests',1);%skip sync tests
root=load_root();
dbstop if error %% To debug your code..
%rng('shuffle');reset(RandStream.getGlobalStream,sum(100*clock));  %randomize
%config_io;Pulse(0);%initalize IO64 for Pulse Script
if ~exist('phase','var');phase=1;end
screens = Screen('Screens');% Get the screen numbers
screenNumber = max(screens);% Draw to external screen (if avaliable)
HideCursor;
white = WhiteIndex(screenNumber);% Define black and white
black = BlackIndex(screenNumber);grey = white / 2;inc = white - grey;

wf=60;fs=11025;
pct=0.3;%post click time
scpha=3;TR=1.5;%TR time
pngs=ngs;
%belief test games
if phase > scpha*2 % out of scanner
    load belieftest;%loads 'beliefs'
    pngs=tnbgs;
    btg=[];
end

[window, windowRect] = PsychImaging('OpenWindow', screenNumber, white); %open "on screen
[screenXpixels, screenYpixels] = Screen('WindowSize', window);%grab dims
[xCenter, yCenter] = RectCenter(windowRect);%grab center
ratioX = screenXpixels/1200;ratioY = screenYpixels/700;%adjust ratios
Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');%alpha blend for smoothness
Screen('FillRect', window, white);Screen('TextSize', window, 30);% word defaults
Screen('TextFont', window, 'Times New Roman');
baseRect = [0 0 150 150];penWidthPixels = 6;colorSquares = [0 0 0];
pSquares=3;pBoxes = NaN(4,pSquares);prefsc=275;%preference scale
x_p=[-prefsc 0 prefsc]+xCenter;y_p=zeros(length(x_p),1)+prefsc+yCenter;
for i = 1:pSquares;pBoxes(:,i)=CenterRectOnPointd(baseRect*1.5,x_p(i),y_p(i));end
eSquares=7;eBoxes = NaN(4,eSquares);enjsc=160;%enjoyment scale
x_e=[-enjsc*3 -enjsc*2 -enjsc*1 0 enjsc*1 enjsc*2 enjsc*3]+xCenter;
y_e=zeros(length(x_e),1)+prefsc+yCenter;enjtext=strings(eSquares,1);
for i = 1:eSquares;eBoxes(:,i)=CenterRectOnPointd(baseRect,x_e(i),y_e(i));
    enjtext(i)=num2str(i);end
for ng=1:pngs
    if phase<=scpha;gtrial=(phase-1)*ngs+ng;g_id=g_o(gtrial);ml_id=ml(g_id);
    elseif phase<=scpha*2;gtrial=(phase-1)*ngs+ng;g_id=g_o(gtrial-(phase-1)*ngs);ml_id=ml(g_id);
    else;gtrial=sgs*2+ng;b_id=gb_o(ng);g_id=b_id+sgs;homestring=gf{g_id}.homestring;
    end
    home=gf{g_id}.home{:};visitor=gf{g_id}.visitor{:};tl=1;
    if phase<=scpha
%         if et
%             lookX{ng}=0;PointRT(ng) = NaN;
%             lookY{ng}=0;pupil{ng}=0;
%             EyelinkStartTrialMessage(window,ng);
%             baselineTime = 1;
%             ITIs = ITIs -baselineTime;
%         end
        hs=gf{g_id}.hs{:};vs=gf{g_id}.vs{:};
        fn=[root '720res/' char(gf{g_id}.gf) 'cut.mov'];
        g_o_m{gtrial}.timelog(tl)=GetSecs();tl=tl+1;
        DrawFormattedText(window,['The current score with 5 minutes left of this game is ... \n ' home ', ' hs ' \n to \n ' visitor ', ' vs '.'],'center','center',black,wf*(screenXpixels/1500),[],[],1.20);
        Screen('Flip',window);
        WaitSecs(2*TR);
        g_o_m{gtrial}.timelog(tl)=GetSecs();tl=tl+1;
        %playMovEZ('/Users/jantony/Desktop/surprisesuspense/720res/4_8SuspenseExample.mov',0,window,root)
%         if et;
%             [lookX{ng} lookY{ng} pupil{ng} keyResponse{ng} PointRT(ng)] = ...
%                 WaitForButtonsAndEyes(lookX, lookY, pupil, ng,...
%                 timeToRespond, stimulusTime(ng), TriggerDevice, TriggerKey, eyeTracking, ITIs(i), window );
%         end
        if ng==1;if et;g_o_m{gtrial}.etstart=clock;Eyelink('StartRecording');end;end 
        playMovEZ(fn,0,window,root,0)%ml_id-5)
        Screen('Close', window);
        %preference test
        g_o_m{gtrial}.timelog(tl)=GetSecs();tl=tl+1;
        [window, windowRect] = PsychImaging('OpenWindow', screenNumber, white); %open "on screen
        Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');%alpha blend for smoothness
        Screen('FillRect', window, white);Screen('TextSize', window, 30);% word defaults
        Screen('TextFont', window, 'Times New Roman');
        SetMouse(screenXpixels/2, screenYpixels/2);ShowCursor;t0 = GetSecs();
        DrawFormattedText(window,'Were you cheering for one specific team? If so, which? Otherwise, please mark "no preference".','center','center',black,wf*(screenXpixels/1500),[],[],1.20);
        DrawFormattedText(window,home,'center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],pBoxes(:,1)');
        DrawFormattedText(window,'No pref','center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],pBoxes(:,2)');
        DrawFormattedText(window,visitor,'center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],pBoxes(:,3)');
        %Screen('FrameRect',window,colorSquares,pBoxes,penWidthPixels); % show blank boxes also
        Screen('Flip',window);
        g_o_m{gtrial}.timelog(tl)=GetSecs();tl=tl+1;
        [mx, my, buttons] = GetMouse(window);
        DrawFormattedText(window,'Were you cheering for one specific team? If so, click on it. Otherwise, please click "no preference".','center','center',black,wf*(screenXpixels/1500),[],[],1.20);
        DrawFormattedText(window,home,'center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],pBoxes(:,1)');
        DrawFormattedText(window,'No preference','center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],pBoxes(:,2)');
        DrawFormattedText(window,visitor,'center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],pBoxes(:,3)');
        %Screen('FrameRect',window,colorSquares,pBoxes,penWidthPixels); % show blank boxes also
        Screen('Flip',window);
        g_o_m{gtrial}.timelog(tl)=GetSecs();tl=tl+1;
        inside=zeros(size(pBoxes,2),1);
        while sum(buttons) == 0
            [mx, my, buttons] = GetMouse(window);
            for ii=1:size(pBoxes,2);inside(ii)=IsInRect(mx,my,pBoxes(:,ii));end
            if sum(inside)>0 && sum(buttons)>0;pBox=find(inside==1);
                pBox=pBox(1);break;else;buttons = 0;end
        end
        HideCursor;t1=GetSecs();
        pc=[pBox mx my t1-t0 t0 t1];
        g_o_m{gtrial}.pref=pc;
        WaitSecs(pct);
        %enjoyment test
        g_o_m{gtrial}.timelog(tl)=GetSecs();tl=tl+1;
        SetMouse(screenXpixels/2, screenYpixels/2);ShowCursor;t0 = GetSecs();
        DrawFormattedText(window,'How enjoyable did you find the game (1 = not at all enjoyable, 4 = moderately enjoyable, 7 = highly enjoyable','center','center',black,wf*(screenXpixels/1500),[],[],1.20);
        DrawFormattedText(window,char(enjtext(1)),'center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],eBoxes(:,1)');
        DrawFormattedText(window,char(enjtext(2)),'center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],eBoxes(:,2)');
        DrawFormattedText(window,char(enjtext(3)),'center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],eBoxes(:,3)');
        DrawFormattedText(window,char(enjtext(4)),'center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],eBoxes(:,4)');
        DrawFormattedText(window,char(enjtext(5)),'center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],eBoxes(:,5)');
        DrawFormattedText(window,char(enjtext(6)),'center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],eBoxes(:,6)');
        DrawFormattedText(window,char(enjtext(7)),'center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],eBoxes(:,7)');
        %Screen('FrameRect',window,colorSquares,eBoxes,penWidthPixels); % show blank boxes also
        Screen('Flip',window);
        [mx, my, buttons] = GetMouse(window);
        DrawFormattedText(window,'How enjoyable did you find the game (1 = not at all enjoyable, 4 = moderately enjoyable, 7 = highly enjoyable','center','center',black,wf*(screenXpixels/1500),[],[],1.20);
        DrawFormattedText(window,char(enjtext(1)),'center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],eBoxes(:,1)');
        DrawFormattedText(window,char(enjtext(2)),'center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],eBoxes(:,2)');
        DrawFormattedText(window,char(enjtext(3)),'center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],eBoxes(:,3)');
        DrawFormattedText(window,char(enjtext(4)),'center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],eBoxes(:,4)');
        DrawFormattedText(window,char(enjtext(5)),'center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],eBoxes(:,5)');
        DrawFormattedText(window,char(enjtext(6)),'center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],eBoxes(:,6)');
        DrawFormattedText(window,char(enjtext(7)),'center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],eBoxes(:,7)');
        %Screen('FrameRect',window,colorSquares,eBoxes,penWidthPixels); % show blank boxes also
        Screen('Flip',window);
        g_o_m{gtrial}.timelog(tl)=GetSecs();tl=tl+1;
        inside=zeros(size(eBoxes,2),1);
        while sum(buttons) == 0
            [mx, my, buttons] = GetMouse(window);
            for ii=1:size(eBoxes,2);inside(ii)=IsInRect(mx,my,eBoxes(:,ii));end
            if sum(inside)>0 && sum(buttons)>0
                eBox=find(inside==1);eBox=eBox(1);break;else;buttons = 0;end
        end
        HideCursor;t1 = GetSecs();
        ec=[eBox mx my t1-t0 t0 t1];
        g_o_m{gtrial}.enjoy=ec;
        g_o_m{gtrial}.timelog(tl)=GetSecs();tl=tl+1;
    elseif phase<=scpha*2 % recall test
        g_o_m{gtrial}.timelog(tl)=GetSecs();tl=tl+1;
        if ng==1 % instructions
            DrawFormattedText(window,['Please recall the next game in as much detail as possible. Please include the score and approximate amount of time left during any possession you can recollect and any parts of that possession as they unfold (e.g., a pass, a drive, a rebound). If you are having difficulty, try to think of any highlights or any major aspects of the "narrative" of the game. Please also include as part of your recollection the team that won and the final score.'],'center','center',black,wf*(screenXpixels/1500),[],[],1.20);
            Screen('Flip',window);
            [mx, my, buttons] = GetMouse(window);
            while sum(buttons) == 0;[mx, my, buttons] = GetMouse(window);end
        end
        g_o_m{gtrial}.timelog(tl)=GetSecs();tl=tl+1;
        DrawFormattedText(window,['Please recall as much as you can about the game between \n ' home '\n versus \n ' visitor '.\n Click the mouse when you are ready to begin and we will start the recording. When you are done remembering everything you can, simply say something like, "I am done."'],'center','center',black,wf*(screenXpixels/1500),[],[],1.20);
        Screen('Flip',window);
        [mx, my, buttons] = GetMouse(window);
        while sum(buttons) == 0;[mx, my, buttons] = GetMouse(window);end
        g_o_m{gtrial}.starttime=GetSecs;
        recordData=audiorecorder(fs,16,1);
        record(recordData);
        while (1);FlushEvents('keyDown');temp = GetChar;% experimenter controls this part
        if (temp == 'q');break;end;end
        stop(recordData);
        g_o_m{gtrial}.recall=getaudiodata(recordData);
        g_o_m{gtrial}.endtime=GetSecs;
        g_o_m{gtrial}.RT=g_o_m{gtrial}.endtime-g_o_m{gtrial}.starttime;
    else %belief test
        g_o_m{gtrial}.b_id=b_id;
        for ii=1:length(beliefs{b_id}.gl.SecondsInVideo)
            gr=beliefs{b_id}.gl.GameRemaining(ii);wp=beliefs{b_id}.gl.VidWinProb(ii);
            nummins=floor(gr/60);numsecs=mod(gr,60);
            hs=num2str(beliefs{b_id}.gl.HomeScore(ii));vs=num2str(beliefs{b_id}.gl.VisitorScore(ii));
            poss=string(beliefs{b_id}.gl.TeamPoss(ii));
            question=['The current score with ' num2str(nummins) ' minutes, ' num2str(numsecs) ' seconds left is ... \n ' home ', ' hs ' \n to \n ' visitor ', ' vs '.\n Possession: ' char(poss) '.\nHow likely is ' homestring ' to win?'];
            %             question=['The current score with ' num2str(nummins) ' minutes, ' num2str(numsecs) ' seconds left is ... \n ' home ', ' hs ' \n to \n ' visitor ', ' vs '.\nHow likely is ' homestring ' to win?'];
            endPoints = {'0%','100%'};
            [position,RT,answer]=slideScale(window,question,windowRect,endPoints,'device','mouse','displayposition',true,'range',2);
            g_o_m{gtrial}.belief(ii,:)=[gr,wp,position,RT,answer];btg=[btg;wp position];
            %playMovEZ('/Users/jantony/Desktop/surprisesuspense/720res/4_8SuspenseExample.mov',0,window,root)
            if ii<length(beliefs{b_id}.gl.SecondsInVideo);g_o_m{gtrial}.vid_time_r(ii,:)=clock;
                cliplength=beliefs{b_id}.gl.SecondsInVideo(ii+1)-beliefs{b_id}.gl.SecondsInVideo(ii);
                fn=[root '720res/' char(gf{g_id}.gf) '_' num2str(ii) '.mov'];
                playMovEZ(fn,0,window,root,cliplength-0.5,1);
                Screen('Close', window);
                [window, windowRect] = PsychImaging('OpenWindow', screenNumber, white); %open "on screen
                Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');%alpha blend for smoothness
                Screen('FillRect', window, white);Screen('TextSize', window, 30);% word defaults
                Screen('TextFont', window, 'Times New Roman');
            end
        end
        DrawFormattedText(window,'Next Game!','center','center',black,wf*(screenXpixels/1500),[],[],1.20);
        Screen('Flip',window);
        WaitSecs(2*TR);
    end
end
g_o_m{gtrial}.exit_time_r=clock;
sn=[root 'subs/' sub '/' sub '_phase' num2str(phase) '.mat'];
if ~exist(sn,'file');save(sn);else;q=clock;sn=[sn(1:end-4) '_' num2str(floor(q(6))*q(5)*q(4)) '.mat'];save(sn);end;
sca;
% if phase > scpha % out of scanner
%     figure;scatter(btg(:,1),btg(:,2));xlabel('True win probability');
%     ylabel('Guessed win probability');lsline;r=corrcoef(btg(:,1),btg(:,2));
%     tn =['r = ' num2str(r(2))];title(tn);
% end