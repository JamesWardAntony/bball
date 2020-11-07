%% import games for value test
[~, ~, raw] = xlsread([root 'analysis/IngameVars.xlsx'],'2013 Games');
raw = raw(2:29,1:8);raw(cellfun(@(x) ~isempty(x) && isnumeric(x) && isnan(x),raw)) = {''};
stringVectors = string(raw(:,[1,3,5,6]));stringVectors(ismissing(stringVectors)) = '';
raw = raw(:,[2,4,7,8]);
R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),raw);raw(R) = {NaN};
data = reshape([raw{:}],size(raw));games2013 = table;
games2013.Home = stringVectors(:,1);games2013.HomeSeed = data(:,1);
games2013.Visitor = stringVectors(:,2);games2013.VisitorSeed = data(:,2);
games2013.FinalScore = stringVectors(:,3);games2013.mScore = stringVectors(:,4);
games2013.mWinProb = data(:,3);games2013.mWinProbHome = data(:,4);
clearvars data raw stringVectors R;
%% check all teams and sort against 2012 teams
teams2012=strings(length(games2013.Home),1);j=0;
for i=1:length(g)
    if g{i}.HomeSeed~=1
        j=j+1;
        if regplay==1;fill=char(g{i}.Home(1));
        else;fill=char(g{i}.Home(1));
        end
        teams2012((j-1)*2+1)=fill;
        fill=num2str(g{i}.HomeSeed);teams2012((j-1)*2+2,:)=fill;
    end
end

teams2013=[games2013.Home;games2013.Visitor];
[teams2012,ind2012]=sort(teams2012);
[teams2013,ind2013]=sort(teams2013);
allteams=[teams2012 teams2013];
