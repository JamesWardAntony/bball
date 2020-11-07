%% init_ss.m 06.11.18 Run this program to run exp, filling in info in box.
% initiating everything...
g_o_m{(phase-1)*ngs+1}.init_time=GetSecs();
qz=clock;g_o_m{(phase-1)*ngs+1}.init_time_r=qz;
PsychDefaultSetup(2);bgColor=[128 128 128];
Screen('Preference','SkipSyncTests',1);%skip sync tests
devices=PsychHID('Devices');
if inpo==1;jsnum=1;
else;for i=1:length(devices);if(devices(i).buttons==10);jsnum=i;end;end%index==2
end
dbstop if error %% To debug your code..
%rng('shuffle');reset(RandStream.getGlobalStream,sum(100*clock));  %randomize
%config_io;Pulse(0);%initalize IO64 for Pulse Script
if ~exist('phase','var');phase=1;end
screens = Screen('Screens');% Get the screen numbers
screenNumber = max(screens);% Draw to external screen (if avaliable)
HideCursor;
white = WhiteIndex(screenNumber);% Define black and white
black = BlackIndex(screenNumber);grey = white / 2;inc = white - grey;
wf=60;fs=11025;chill=30;jse=32768;
pct=0.3;%post click time
scpha=3;TR=1;%TR time
pngs=ngs;
%belief and sus pref test games
if phase==scpha*2+1
    load belieftest;%loads 'beliefs'
    pngs=tnbgs;btg=[];
elseif phase==scpha*2+2
    pngs=tnsgs;stg=[];
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
    elseif phase<=scpha*2;gtrial=(phase-1)*ngs+ng;g_id=g_o(gtrial-tngs);ml_id=ml(g_id);
    elseif phase<=scpha*2+1;gtrial=tngs*2+ng;b_id=gb_o(ng);g_id=b_id+tngs;homestring=gf{g_id}.homestring;
    elseif phase<=scpha*2+2;gtrial=tngs*2+tnbgs+ng;s_id=gs_o(ng);g_id=100+s_id;
    end
    home=gf{g_id}.home{:};visitor=gf{g_id}.visitor{:};tl=1;
    if phase<=scpha
        hs=gf{g_id}.hs{:};vs=gf{g_id}.vs{:};mode=1;
        fn=[root '720res/' char(gf{g_id}.gf) 'cut.mov'];
        g_o_m{gtrial}.g_id=g_id;
        g_o_m{gtrial}.timelog(tl)=GetSecs();tl=tl+1;
        if et
            if ng==1
                EyelinkInit();el=EyelinkInitDefaults(window);
                EyelinkDoTrackerSetup(el,'c');mode=0;
            end
        end
        g_o_m{gtrial}.timelog(tl)=GetSecs();tl=tl+1;
        DrawFormattedText(window,['The current score with 5 minutes left of this game is ... \n ' home ', ' hs ' \n to \n ' visitor ', ' vs '.'],'center','center',black,wf*(screenXpixels/1500),[],[],1.20);
        Screen('Flip',window);
        WaitSecs(3*TR);
        g_o_m{gtrial}.timelog(tl)=GetSecs();tl=tl+1;
        %playMovEZ('/Users/jantony/Desktop/surprisesuspense/720res/4_8SuspenseExample.mov',0,window,root)
        %g_o_m=playMovEZ(g_o_m,fn,fmri,window,root,ml_id-20,mode,sub,phase,gtrial,qz,et);%run short
        g_o_m=playMovEZ(g_o_m,fn,fmri,window,root,0,mode,sub,phase,gtrial,qz,et);Screen('Close', window);
        %preference test
        g_o_m{gtrial}.timelog(tl)=GetSecs();tl=tl+1;
        [window, windowRect] = PsychImaging('OpenWindow', screenNumber, white); %open "on screen
        Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');%alpha blend for smoothness
        Screen('FillRect', window, white);Screen('TextSize', window, 30);% word defaults
        Screen('TextFont', window, 'Times New Roman');
        if inpo==1;SetMouse(screenXpixels/2, screenYpixels/2,window);ShowCursor;end
        t0 = GetSecs();
        DrawFormattedText(window,'Were you cheering for one specific team? If so, which? Otherwise, please mark "no preference".','center','center',black,wf*(screenXpixels/1500),[],[],1.20);
        DrawFormattedText(window,home,'center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],pBoxes(:,1)');
        DrawFormattedText(window,'No pref','center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],pBoxes(:,2)');
        DrawFormattedText(window,visitor,'center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],pBoxes(:,3)');
        %Screen('FrameRect',window,colorSquares,pBoxes,penWidthPixels); % show blank boxes also
        Screen('Flip',window);
        g_o_m{gtrial}.timelog(tl)=GetSecs();tl=tl+1;
        if inpo==1;[mx, my, buttons] = GetMouse(window);end
        [keyIsDown, secs, keyCode] = PsychHID('KbCheck',jsnum);
        DrawFormattedText(window,'Were you cheering for one specific team? If so, click on it. Otherwise, please click "no preference".','center','center',black,wf*(screenXpixels/1500),[],[],1.20);
        DrawFormattedText(window,home,'center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],pBoxes(:,1)');
        DrawFormattedText(window,'No preference','center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],pBoxes(:,2)');
        DrawFormattedText(window,visitor,'center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],pBoxes(:,3)');
        %Screen('FrameRect',window,colorSquares,pBoxes,penWidthPixels); % show blank boxes also
        Screen('Flip',window);
        g_o_m{gtrial}.timelog(tl)=GetSecs();tl=tl+1;
        inside=zeros(size(pBoxes,2),1);answer = 0;
        if inpo==1
            while sum(buttons) == 0
                [mx, my, buttons] = GetMouse(window);
                for ii=1:size(pBoxes,2);inside(ii)=IsInRect(mx,my,pBoxes(:,ii));end
                if sum(inside)>0 && sum(buttons)>0;pBox=find(inside==1);
                    pBox=pBox(1);break;else;buttons = 0;end
            end
        elseif inpo==2
            xy=[xCenter,yCenter];
            while answer == 0
                axisState = [Gamepad('GetAxis',2, 1), Gamepad('GetAxis',2, 2)];%jwa not 2,1 etc for laptop
                dxy =[((axisState(1)/jse)*(screenXpixels/2))/chill,((axisState(2)/jse*(screenYpixels/2)))/chill];
                if and(xy(1)+dxy(1)>0,xy(1)+dxy(1)<screenXpixels)
                    if and(xy(2)+dxy(2)>0,xy(2)+dxy(2)<screenYpixels)
                        xy=[xy(1)+dxy(1),xy(2)+dxy(2)];
                    end
                end
                mx=xy(1);my=xy(2);
                %xy =[(axisState(1)/jse)*(screenXpixels/2) + xCenter, (axisState(2)/jse*(screenYpixels/2)) + yCenter];
                [keyIsDown, secs, keyCode] = PsychHID('KbCheck', jsnum);
                DrawFormattedText(window,'Were you cheering for one specific team? If so, click on it. Otherwise, please click "no preference".','center','center',black,wf*(screenXpixels/1500),[],[],1.20);
                DrawFormattedText(window,home,'center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],pBoxes(:,1)');
                DrawFormattedText(window,'No preference','center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],pBoxes(:,2)');
                DrawFormattedText(window,visitor,'center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],pBoxes(:,3)');
                DrawFormattedText(window,'o',mx,my,black,100*wf*(screenXpixels/1500));
                Screen('Flip',window);
                for ii=1:size(pBoxes,2);inside(ii)=IsInRect(mx,my,pBoxes(:,ii));end
                if sum(inside)>0 && sum(keyIsDown)>0;pBox=find(inside==1);
                    pBox=pBox(1);break;else;answer = 0;end
            end
        end
        HideCursor;t1=GetSecs();
        g_o_m{gtrial}.pref=[pBox mx my t1-t0 t0 t1];
        WaitSecs(pct);
        %enjoyment test
        g_o_m{gtrial}.timelog(tl)=GetSecs();tl=tl+1;
        if inpo==1;SetMouse(screenXpixels/2, screenYpixels/2,window);ShowCursor;end
        t0 = GetSecs();
        DrawFormattedText(window,'How enjoyable did you find the game (1 = not at all enjoyable, 4 = moderately enjoyable, 7 = highly enjoyable)?','center','center',black,wf*(screenXpixels/1500),[],[],1.20);
        for j=1:length(enjtext);DrawFormattedText(window,char(enjtext(j)),'center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],eBoxes(:,j)');end
        %Screen('FrameRect',window,colorSquares,eBoxes,penWidthPixels); % show blank boxes also
        Screen('Flip',window);
        [mx, my, buttons] = GetMouse(window);
        DrawFormattedText(window,'How enjoyable did you find the game (1 = not at all enjoyable, 4 = moderately enjoyable, 7 = highly enjoyable)?','center','center',black,wf*(screenXpixels/1500),[],[],1.20);
        for j=1:length(enjtext);DrawFormattedText(window,char(enjtext(j)),'center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],eBoxes(:,j)');end
        %Screen('FrameRect',window,colorSquares,eBoxes,penWidthPixels); % show blank boxes also
        Screen('Flip',window);
        g_o_m{gtrial}.timelog(tl)=GetSecs();tl=tl+1;
        inside=zeros(size(eBoxes,2),1);answer = 0;
        if inpo==1
            while sum(buttons) == 0
                [mx, my, buttons] = GetMouse(window);
                for ii=1:size(eBoxes,2);inside(ii)=IsInRect(mx,my,eBoxes(:,ii));end
                if sum(inside)>0 && sum(buttons)>0
                    eBox=find(inside==1);eBox=eBox(1);break;else;buttons = 0;end
            end
        elseif inpo== 2
            xy=[xCenter,yCenter];
            while answer == 0
                axisState = [Gamepad('GetAxis',2, 1), Gamepad('GetAxis',2, 2)];%jwa not 2,1 etc for laptop
                dxy =[((axisState(1)/jse)*(screenXpixels/2))/chill,((axisState(2)/jse*(screenYpixels/2)))/chill];
                if and(xy(1)+dxy(1)>0,xy(1)+dxy(1)<screenXpixels)
                    if and(xy(2)+dxy(2)>0,xy(2)+dxy(2)<screenYpixels)
                        xy=[xy(1)+dxy(1),xy(2)+dxy(2)];
                    end
                end
                mx=xy(1);my=xy(2);
                DrawFormattedText(window,'How enjoyable did you find the game (1 = not at all enjoyable, 4 = moderately enjoyable, 7 = highly enjoyable','center','center',black,wf*(screenXpixels/1500),[],[],1.20);
                for j=1:length(enjtext);DrawFormattedText(window,char(enjtext(j)),'center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],eBoxes(:,j)');end
                DrawFormattedText(window,'o',mx,my,black,100*wf*(screenXpixels/1500));
                Screen('Flip',window);
                [keyIsDown, secs, keyCode] = PsychHID('KbCheck',jsnum);
                for ii=1:size(eBoxes,2);inside(ii)=IsInRect(mx,my,eBoxes(:,ii));end
                if sum(inside)>0 && sum(keyIsDown)>0
                    eBox=find(inside==1);eBox=eBox(1);break;else;answer = 0;end
            end
        end
        HideCursor;t1 = GetSecs();
        g_o_m{gtrial}.enjoy=[eBox mx my t1-t0 t0 t1];
        g_o_m{gtrial}.timelog(tl)=GetSecs();tl=tl+1;
    elseif phase<=scpha*2 % recall test
        g_o_m{gtrial}.g_id=g_id;
        g_o_m{gtrial}.timelog(tl)=GetSecs();tl=tl+1;keyIsDown=0;
        if ng==1 % instructions
            DrawFormattedText(window,['Please recall these games in as much detail as possible. Please include the score and approximate amount of time left during any possession you can recollect and any parts of that possession as they unfold (e.g., a pass, a drive, a rebound). If you are having difficulty, try to think of any highlights or any major aspects of the "narrative" of the game. Please also include as part of your recollection the team that won and the final score. When you are done remembering everything you can, simply say something like, "I am done."'],'center','center',black,wf*(screenXpixels/1500),[],[],1.20);
            Screen('Flip',window);
            if inpo==1;[mx, my, buttons] = GetMouse(window);
                while sum(buttons) == 0;[mx, my, buttons] = GetMouse(window);end
            elseif inpo==2;answer=0;
                while sum(keyIsDown) == 0;[keyIsDown, secs, keyCode] = PsychHID('KbCheck',jsnum);end
            end
            if fmri;fill=WaitTRPulsePTB3_prisma(3);end
        end
        g_o_m{gtrial}.timelog(tl)=GetSecs();tl=tl+1;
        DrawFormattedText(window,['Speak now!\n\n' home '\n versus \n ' visitor '\n\n Say "I am done." when you are done speaking.'],'center','center',black,wf*(screenXpixels/1500),[],[],1.20);
        if fmri;fill=WaitTRPulsePTB3_prisma(1);end
        Screen('Flip',window);
        g_o_m{gtrial}.timelog(tl)=GetSecs();tl=tl+1;
        %[mx, my, buttons] = GetMouse(window);while sum(buttons) == 0;[mx, my, buttons] = GetMouse(window);end
        g_o_m{gtrial}.starttime=GetSecs;
        recordData=audiorecorder(fs,16,1);
        record(recordData);
        while (1);FlushEvents('keyDown');temp = GetChar;% experimenter controls this part
            if (temp == 'q');break;end;end
        stop(recordData);
        g_o_m{gtrial}.timelog(tl)=GetSecs();tl=tl+1;
        g_o_m{gtrial}.recall=getaudiodata(recordData);
        g_o_m{gtrial}.endtime=GetSecs;
        g_o_m{gtrial}.RT=g_o_m{gtrial}.endtime-g_o_m{gtrial}.starttime;
    elseif phase==scpha*2+1 %belief test
        g_o_m{gtrial}.b_id=b_id;
        if ng==1 % instructions
            DrawFormattedText(window,['We will now present to you a series of scenarios from different games in 2012 NCAA basketball tournament. On each trial, we will give you the scores of each team, the amount of time left, and which team is in possession of the ball. We will then ask you the likelihood that the higher seeded team will win the game. Please slide the scale from 0 to 100% to make your prediction. These scenarios will begin with 5 minutes remaining in the game and will go until the game is completed.'],'center','center',black,wf*(screenXpixels/1500),[],[],1.20);
            Screen('Flip',window);keyIsDown=0;
            if inpo==1;[mx, my, buttons] = GetMouse(window);while sum(buttons) == 0;[mx, my, buttons] = GetMouse(window);end
            elseif inpo==2;while sum(keyIsDown) == 0;[keyIsDown, secs, keyCode] = PsychHID('KbCheck',jsnum);end
            end
        end
        
        for ii=1:length(beliefs{b_id}.gl.SecondsInVideo)
            gr=beliefs{b_id}.gl.GameRemaining(ii);wp=beliefs{b_id}.gl.VidWinProb(ii);
            hs=num2str(beliefs{b_id}.gl.HomeScore(ii));vs=num2str(beliefs{b_id}.gl.VisitorScore(ii));
            nummins=floor(gr/60);numsecs=mod(gr,60);poss=string(beliefs{b_id}.gl.TeamPoss(ii));
            question=['The current score with ' num2str(nummins) ' minutes, ' num2str(numsecs) ' seconds left is ... \n ' home ', ' hs ' \n to \n ' visitor ', ' vs '.\n Possession: ' char(poss) '.\nHow likely is ' homestring ' to win?'];
            %             question=['The current score with ' num2str(nummins) ' minutes, ' num2str(numsecs) ' seconds left is ... \n ' home ', ' hs ' \n to \n ' visitor ', ' vs '.\nHow likely is ' homestring ' to win?'];
            endPoints = {'0%','100%'};
            if inpo==1;[position,RT,answer]=slideScale(window,question,windowRect,endPoints,'device','mouse','displayposition',true,'range',2);
            elseif inpo==2;[position,RT,answer]=slideScale(window,question,windowRect,endPoints,'device','joystick','jsnum',jsnum,'displayposition',true,'range',2);
            end
            g_o_m{gtrial}.belief(ii,:)=[gr,wp,position,RT,answer];btg=[btg;wp position];
            WaitSecs(pct);
            %playMovEZ('/Users/jantony/Desktop/surprisesuspense/720res/4_8SuspenseExample.mov',0,window,root)
            %             if ii<length(beliefs{b_id}.gl.SecondsInVideo);g_o_m{gtrial}.vid_time_r(ii,:)=clock;
            %                 cliplength=beliefs{b_id}.gl.SecondsInVideo(ii+1)-beliefs{b_id}.gl.SecondsInVideo(ii);
            %                 fn=[root '720res/' char(gf{g_id}.gf) '_' num2str(ii) '.mov'];
            %                 playMovEZ(fn,0,window,root,cliplength-0.5,1);Screen('Close', window);
            %                 [window, windowRect] = PsychImaging('OpenWindow', screenNumber, white); %open "on screen
            %                 Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');%alpha blend for smoothness
            %                 Screen('FillRect', window, white);Screen('TextSize', window, 30);% word defaults
            %                 Screen('TextFont', window, 'Times New Roman');
            %             end
        end
        if ng<pngs
            DrawFormattedText(window,'Next Game!','center','center',black,wf*(screenXpixels/1500),[],[],1.20);
            Screen('Flip',window);
            WaitSecs(2*TR);
        end
    elseif phase==scpha*2+2 %suspense pref test
        if ng==1 % instructions
            DrawFormattedText(window,['In this final phase of the experiment, we will show you the score of a number of different games with 5 minutes remaining, this time from the 2013 NCAA tournament (one year after the games you just watched). Please indicate how excited you would be to watch this game (from 5 minutes remaining until the end) by clicking a number from 1 (least excited) to 7 (most excited). This phase is hypothetical; you will not be watching any of these games.'],'center','center',black,wf*(screenXpixels/1500),[],[],1.20);
            Screen('Flip',window);keyIsDown=0;
            if inpo==1;[~, ~, buttons] = GetMouse(window);while sum(buttons) == 0;[~, ~, buttons] = GetMouse(window);end
            elseif inpo==2;while sum(keyIsDown) == 0;[keyIsDown, secs, keyCode] = PsychHID('KbCheck',jsnum);end
            end
        end
        g_o_m{gtrial}.s_id=s_id;hwp=str2num(char(gf{g_id}.hwp));hs=char(gf{g_id}.hs);vs=char(gf{g_id}.vs);t0 = GetSecs();
        question=['The current score with 5 minutes left is ... \n ' home ', ' hs ' \n to \n ' visitor ', ' vs '.\nHow interested would you be in watching this game \n(1 = not at all interested, 4 = moderately interested, 7 = highly interested)?'];
        if inpo==1;SetMouse(screenXpixels/2, screenYpixels/2,window);ShowCursor;end
        xy=[xCenter,yCenter];mx=xy(1);my=xy(2);
        DrawFormattedText(window,question,'center','center',black,wf*(screenXpixels/1500),[],[],1.20);
        for j=1:length(enjtext);DrawFormattedText(window,char(enjtext(j)),'center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],eBoxes(:,j)');end
        Screen('Flip',window);
        if inpo==1
            ShowCursor;
            [mx, my, buttons] = GetMouse(window);
            while sum(buttons) == 0
                [mx, my, buttons] = GetMouse(window);
                for ii=1:size(eBoxes,2);inside(ii)=IsInRect(mx,my,eBoxes(:,ii));end
                if sum(inside)>0 && sum(buttons)>0
                    eBox=find(inside==1);eBox=eBox(1);break;else;buttons = 0;end
            end
        elseif inpo==2
            xy=[xCenter,yCenter];answer = 0;
            while answer == 0
                axisState = [Gamepad('GetAxis',2, 1), Gamepad('GetAxis',2, 2)];%jwa not 2,1 etc for laptop
                dxy =[((axisState(1)/jse)*(screenXpixels/2))/chill,((axisState(2)/jse*(screenYpixels/2)))/chill];
                if and(xy(1)+dxy(1)>0,xy(1)+dxy(1)<screenXpixels)
                    if and(xy(2)+dxy(2)>0,xy(2)+dxy(2)<screenYpixels)
                        xy=[xy(1)+dxy(1),xy(2)+dxy(2)];
                    end
                end
                mx=xy(1);my=xy(2);
                DrawFormattedText(window,question,'center','center',black,wf*(screenXpixels/1500),[],[],1.20);
                for j=1:length(enjtext);DrawFormattedText(window,char(enjtext(j)),'center','center',black,wf*(screenXpixels/1500),[],[],1.20,[],eBoxes(:,j)');end
                DrawFormattedText(window,'o',mx,my,black,100*wf*(screenXpixels/1500));
                Screen('Flip',window);
                [keyIsDown, secs, keyCode] = PsychHID('KbCheck',jsnum);
                for ii=1:size(eBoxes,2);inside(ii)=IsInRect(mx,my,eBoxes(:,ii));end
                if sum(inside)>0 && sum(keyIsDown)>0;eBox=find(inside==1);eBox=eBox(1);break;else;answer=0;end
            end
        end
        t1 = GetSecs();preference=[s_id hwp eBox mx my t1-t0 t0 t1];
        g_o_m{gtrial}.preference=preference;stg=[stg;hwp eBox];
        WaitSecs(pct);
    end
end
g_o_m{gtrial}.exit_time_r=clock;sca;
sn=[root 'subs/' sub '/' sub '_phase' num2str(phase) '.mat'];
if ~exist(sn,'file');save(sn);else;sn=[sn(1:end-4) '_' num2str(floor(qz(6))*qz(5)*qz(4)) '.mat'];save(sn);end
if et;Eyelink('Shutdown');ListenChar(1);end
% 
% if phase==scpha*2+1
%     figure;scatter(btg(:,1),btg(:,2));xlabel('True win probability');
%     ylabel('Guessed win probability');lsline;r=corrcoef(btg(:,1),btg(:,2));
%     tn =['r = ' num2str(r(2))];title(tn);
% elseif phase==scpha*2+2;d50=100-abs(stg(:,1)-50);
%     figure;scatter(d50,stg(:,2));xlabel('Entropy');ylabel('Interest');
%     lsline;r=corrcoef(d50,stg(:,2));tn =['r = ' num2str(r(2))];title(tn);
% end