%% loads game files from Ken Pomeroy's corpus & re-arranges
%% first load in variables from csv
filename = [root 'analysis/winprob_all.csv'];
opts = delimitedTextImportOptions("NumVariables", 10);opts.Delimiter = ",";
switch regplay;case 1;opts.DataLines = [2, 364563];
    case 2;opts.DataLines = [364564, 366679];end
opts.VariableNames = ["DateOfGame", "GameID", "TimeLeft", "Period", "Visitor", "VisScore", "Home", "HomeScore", "TeamPoss", "WinProbHome"];
opts.VariableTypes = ["datetime", "double", "string", "double", "string", "double", "string", "double", "string", "double"];
opts = setvaropts(opts, 1, "InputFormat", "yyyy-MM-dd");
opts = setvaropts(opts, 3, "WhitespaceRule", "preserve");
opts = setvaropts(opts, [3, 5, 7, 9], "EmptyFieldRule", "auto");
opts.ExtraColumnsRule = "ignore";opts.EmptyLineRule = "read";
dataArray = readtable("/Volumes/norman/jantony/surprisesuspense/analysis/winprob_all.csv", opts);
DateOfGame = dataArray{:, 1};GameID = dataArray{:, 2};
TimeLeft = dataArray{:, 3};Period = dataArray{:, 4};
Visitor = dataArray{:, 5};VisScore = dataArray{:, 6};
Home = dataArray{:, 7};HomeScore = dataArray{:, 8};
TeamPoss = dataArray{:, 9};WinProbHome = dataArray{:, 10};%KP's win probability prediction
clearvars filename delimiter startRow formatSpec fileID dataArray ans;
%% load historical win % by seed. Note - this is never used.
%source: https://www.printyourbrackets.com/ncaa-tournament-records-by-seed.html
if regplay==2;load gamehistwinpercs;end
%% re-arrange data
%first, if regular season, get rid of data from 1st half of game
if regplay==1;K=find(Period==2);
elseif regplay==2;K=find(and(and(Period==2,GameID>5715),GameID<5789));end% select only first round

DateOfGame=DateOfGame(K,:);GameID=GameID(K,:);Home=Home(K,:);
HomeScore=HomeScore(K,:);Period=Period(K,:);TeamPoss=TeamPoss(K,:);
TimeLeft=TimeLeft(K,:);Visitor=Visitor(K,:);VisScore=VisScore(K,:);
WinProbHome=WinProbHome(K,:);clear K;

%eliminate games we don't have final score for AND
%create variable for easily indexing whether the home team won
games=unique(GameID);HomeWin=nan(length(WinProbHome),1);E=[];
for i=1:length(games)
    rows=find(GameID==games(i));
    %assess whether home team won
    if WinProbHome(rows(end))==100
        HomeWin(rows)=1;
    elseif WinProbHome(rows(end))==0
        HomeWin(rows)=0;
    else %no final score ...
        E=[E;rows];
    end
end
j=1;K=zeros(length(DateOfGame)-length(E),1);
for i=1:length(DateOfGame);if isempty(find(E==i,1));K(j,1)=i;j=j+1;end;end
DateOfGame=DateOfGame(K,:);GameID=GameID(K,:);Home=Home(K,:);
HomeScore=HomeScore(K,:);Period=Period(K,:);TeamPoss=TeamPoss(K,:);
TimeLeft=TimeLeft(K,:);Visitor=Visitor(K,:);VisScore=VisScore(K,:);
WinProbHome=WinProbHome(K,:);HomeWin=HomeWin(K,:);clear K;
games=unique(GameID);% reassign if there's no final score

%convert date/time to # seconds left.
SecLeft=nan(length(GameID),1);
for i=1:length(TimeLeft)
    timefound=0;q=datestr(TimeLeft(i));
    for j=1:length(q)
        if strcmpi(q(j),' ') %find blank
            if regplay==1;mint=str2double(q(j+1:j+2));%(j+4:j+5)
            sec=str2double(q(j+4:j+5));timefound=1;
            %else;regplay=1;mint=str2double(q(j+1:j+2));sec=str2double(q(j+4:j+5));timefound=1;%%%3/23
            elseif regplay==2;mint=str2double(q(j+1:j+2));sec=str2double(q(j+4:j+5));timefound=1;
            end
        end
    end
    if timefound;SecLeft(i)=mint*60+sec;
    else;SecLeft(i)=0;
    end
end