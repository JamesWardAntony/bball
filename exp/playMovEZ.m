% function playMovEZ(movieName,doTrigger)
% Plays movies, easy!
%  movieName: filename of your movie ('DDA_all_forward.mp4')
%  doTrigger: 1=wait for scanner pulse to start movie; 0=start without scanner
%
% jc 04/09/12 adapted from playMov.m, adapted from NTB's langabs.m 5/11/10
% - cleaned up some leftovers. got rid of doFixation param, which was obsolete.
% jc 01/25/13 adapted from playMovIvx, cut out the Ivx (eyetracking) stuff.
% jwa 04/17/19 adapted to play within another script, so doesn't just close
% out

function g_o_m=playMovEZ(g_o_m,movieName,doTrigger,window,root,starttime,mode,sub,phase,gtrial,qz,et,suspy)
preall = GetSecs;
if ~exist('doTrigger','var'); doTrigger = 1; end
if ~exist('movieName','var'); fprintf('I need a movie file to play!\n'); return; end
if ~exist('starttime','var'); starttime = 0; end
if ~exist('mode','var'); mode = 0; end

% boilerplate
seed = sum(100*clock);rand('twister',seed);ListenChar(2);
HideCursor%ShowCursor;%1/20/20

% platform-independent responses
KbName('UnifyKeyNames');
dbstop if error;
% set-up screens
Screen('Preference', 'SkipSyncTests', 2);%2
Screen('Preference', 'SuppressAllWarnings', 1);
Screen('Preference', 'Verbosity', 0);Screen('Preference', 'VisualDebugLevel', 1);
Screen('Preference', 'SkipSyncTests', 1);%12/8/19
%screenX = 1800; screenY = 1200;
screens = Screen('Screens');% Get the screen numbers
screenNumber = max(screens);% Draw to external screen (if avaliable)
%HideCursor;
white = WhiteIndex(screenNumber);% Define black and white
black=BlackIndex(screenNumber);
%[window, windowRect] = PsychImaging('OpenWindow', screenNumber, white); %open "on screen
[screenX, screenY] = Screen('WindowSize', window);%grab dims

%adjust this for screen formatting in fMRI
screen_adjustment=.8; % original was 0.85
%screen_adjustment=.5; % changed on 03/05/12 jc
screenXadjusted = screenX*screen_adjustment;
screenYadjusted = screenY*screen_adjustment;

centerX = (screenX/2);centerY = (screenY/2); % bottom of screen cut off by head coil, adjust upwards based on earlier calibration
backColor = 0;textColor = 127;fixationColor = 127;respColor = 255;

% run in dual-display mode if a second window is available
% otherwise run in single-display mode
Screens = Screen('Screens');mainWindow = Screen('OpenWindow',max(Screens),backColor);

Screen(mainWindow, 'TextFont','Arial');Screen(mainWindow, 'TextSize',24);
Screen(mainWindow,'TextColor',textColor);dotSize = 4;
fixDotRect = [centerX-dotSize,centerY-dotSize,centerX+dotSize,centerY+dotSize];

moviePointer = Screen('OpenMovie',mainWindow,movieName);
fill=mmfileinfo(movieName);mlen=fill.Duration;
prerunStart=0;
% show instructions
if mode==0
    instructString = 'The experiment will begin shortly';
    DrawFormattedText(mainWindow,instructString,'center','center');
    if doTrigger;DrawFormattedText(mainWindow,'Press q to wait for trigger','center',10);
    else;DrawFormattedText(mainWindow,'Press q to continue without trigger','center',50);
    end;Screen('Flip',mainWindow);
    
    % wait for a specific key to advance (prevents subject from advancing)
    while (1);FlushEvents('keyDown');temp = GetChar;if (temp == 'q');break;end;end
    Screen(mainWindow,'FillRect',backColor);Screen('Flip',mainWindow);
    prerunStart = GetSecs;
    if doTrigger;runStart = WaitTRPulsePTB3_prisma(3); % MA changed to wait for 3 scanner pulses
    else;runStart = GetSecs;
    end
else;runStart = GetSecs;
end
% wait for scanner pulse
Priority(MaxPriority(mainWindow)); % not necessary on fast machine
goTime = runStart;Priority(0);Screen(mainWindow,'FillRect',backColor);
Screen(mainWindow, 'FillOval',fixationColor,fixDotRect);Screen('Flip',mainWindow);

% seek to start of movie (timeindex 0):
Screen('SetMovieTimeIndex', moviePointer,starttime);

% show fixation 1000 ms before trial%jwa from 3000
if mode==0
    goTime = goTime + 1.0;while(GetSecs<goTime);end
    Screen(mainWindow,'FillRect',backColor);
    Screen(mainWindow,'FillOval',respColor,fixDotRect);
    Screen('Flip',mainWindow);
    
%     % show fixation 1000 ms before trial
%     goTime = goTime + 1;%jwa from 1.3
%     while(GetSecs<goTime); end
%     Screen(mainWindow,'FillRect',backColor);
%     Screen(mainWindow,'FillOval',respColor,fixDotRect);
%     Screen('Flip',mainWindow);
end
% show first view on scanner pulse
Priority(MaxPriority(mainWindow));
if doTrigger;WaitTRPulsePTB3_prisma(1);%,goTime+1); % wait up to 1.5s for next pulse
    Screen('Flip',mainWindow);
else;Screen('Flip',mainWindow);
end

if et;edfFile=[sub '_' num2str(gtrial) '.edf'];
    Eyelink('Openfile',edfFile);Eyelink('StartRecording');Eyelink('Message','SYNCTIME');
end

% start movie
Screen('PlayMovie', moviePointer,1,0,1.0);
movieStart = GetSecs;preMovieTime = movieStart - runStart;
g_o_m{gtrial}.preMovieTime=preMovieTime;g_o_m{gtrial}.movieStart=movieStart;g_o_m{gtrial}.prerunStart=prerunStart;
framePointer = 0;frameCount = 1;
%save([root 'mdfiles/MovieData_' datestr(now,'dd.mm.yyyy.HH.MM') '.mat']);
sn=[root 'subs/' sub '/' sub '_phase' num2str(phase) '_trial' num2str(gtrial) '_' num2str(floor(qz(6))*qz(5)*qz(4)) 'MovieData.mat'];
save(sn);
j=1;clickTimes=[];suspCount=[];xy=[];ListenChar(1);%12/8/19 add
rK1=KbName('RightArrow');rK2=KbName('LeftArrow');susp=4;s_lim=7;wf=300;lc=1;
try
    while (framePointer>-1)
        runTime = Screen('GetMovieTimeIndex', moviePointer);
        framePointer=Screen('GetMovieImage',mainWindow,moviePointer,0,[],[],1);
        if (framePointer>0)
            Screen('DrawTexture',mainWindow,framePointer,[],[centerX-(.5*screenXadjusted),centerY-(.5*screenYadjusted),centerX+(.5*screenXadjusted),centerY+(.5*screenYadjusted)]);
            if suspy
                if lc+3<frameCount
                    [~,~,keyCode] = KbCheck;
                    if keyCode(rK1)==1;if susp<s_lim;susp=susp+1;
                            lc=frameCount;end;end
                    if keyCode(rK2)==1;if susp>1;susp=susp-1;
                            lc=frameCount;end;end
                end
                suspt=['Suspense level: ' num2str(susp) ' / 7'];
                DrawFormattedText(mainWindow,suspt,'center',screenY*0.95,[respColor respColor respColor],wf*(screenX/1500),[],[],1.20);
            end
            Screen('Flip',mainWindow);Screen('Close',framePointer);frameCount = frameCount+1;
        end
        runTimes(frameCount)=runTime;suspCount(frameCount)=susp;
        %12/8/19 - save click times %1/20/20
        [~,~,buttons]=GetMouse(window);%xy(frameCount,1)=x;xy(frameCount,2)=y;
        if sum(buttons)>0;clickTimes(j)=GetSecs;buttons=0;j=j+1;end
    end
catch
    %save([root 'mdfiles/MovieData_' datestr(now,'dd.mm.yyyy.HH.MM') '.mat']);
    save(sn);
end
if et;Eyelink('StopRecording');Eyelink('CloseFile');end%jwa528
Priority(0);Screen(mainWindow,'FillRect',backColor);
Screen(mainWindow, 'FillOval',fixationColor,fixDotRect);
Screen('Flip',mainWindow);Screen('PlayMovie',moviePointer,0);
Screen('CloseMovie',moviePointer);movieEnd=GetSecs;
movieLength=movieEnd-movieStart;clickTimes=clickTimes-movieStart;
g_o_m{gtrial}.movieEnd=movieEnd;g_o_m{gtrial}.movieLength=movieLength;
g_o_m{gtrial}.clickTimes=clickTimes;g_o_m{gtrial}.xy=xy;%1/20/20
g_o_m{gtrial}.suspCount=suspCount;
%save([root 'mdfiles/MovieData_' datestr(now,'dd.mm.yyyy.HH.MM') '.mat']);
save(sn);
%clean up and go home
ListenChar(1);ShowCursor;
%fclose('all');
%Screen('CloseAll');