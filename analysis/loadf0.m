filename=[root 'exp/vids/720res/' gf '.txt'];
delimiter = ' ';startRow = 2;
formatSpec = '%s%s%[^\n\r]';fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'MultipleDelimsAsOne', true, 'TextType', 'string', 'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
fclose(fileID);
raw = repmat({''},length(dataArray{1}),length(dataArray)-1);
for col=1:length(dataArray)-1
    raw(1:length(dataArray{col}),col) = mat2cell(dataArray{col}, ones(length(dataArray{col}), 1));
end
numericData = NaN(size(dataArray{1},1),size(dataArray,2));
for col=[1,2]
    % Converts text in the input cell array to numbers. Replaced non-numeric
    % text with NaN.
    rawData = dataArray{col};
    for row=1:size(rawData, 1)
        % Create a regular expression to detect and remove non-numeric prefixes and
        % suffixes.
        regexstr = '(?<prefix>.*?)(?<numbers>([-]*(\d+[\,]*)+[\.]{0,1}\d*[eEdD]{0,1}[-+]*\d*[i]{0,1})|([-]*(\d+[\,]*)*[\.]{1,1}\d+[eEdD]{0,1}[-+]*\d*[i]{0,1}))(?<suffix>.*)';
        try
            result = regexp(rawData(row), regexstr, 'names');
            numbers = result.numbers;
            
            % Detected commas in non-thousand locations.
            invalidThousandsSeparator = false;
            if numbers.contains(',')
                thousandsRegExp = '^\d+?(\,\d{3})*\.{0,1}\d*$';
                if isempty(regexp(numbers, thousandsRegExp, 'once'))
                    numbers = NaN;
                    invalidThousandsSeparator = true;
                end
            end
            % Convert numeric text to numbers.
            if ~invalidThousandsSeparator
                numbers = textscan(char(strrep(numbers, ',', '')), '%f');
                numericData(row, col) = numbers{1};
                raw{row, col} = numbers{1};
            end
        catch
            raw{row, col} = rawData{row};
        end
    end
end
R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),raw); % Find non-numeric cells
raw(R) = {NaN}; % Replace non-numeric cells
f0 = table;f0.Time_s = cell2mat(raw(:, 1));f0.F0_Hz = cell2mat(raw(:, 2));
clearvars filename delimiter startRow formatSpec fileID dataArray ans raw col numericData rawData row regexstr result numbers invalidThousandsSeparator thousandsRegExp R;
maxt=ceil(max(f0.Time_s));f0m=zeros(maxt,1);
for iii=1:maxt
    f=find(and(f0.Time_s>iii-1,f0.Time_s<iii));
    if numel(f)>0;f0m(iii)=nanmean(f0.F0_Hz(f));end
end
if numel(f0m)<numel(TRsecs)-1;f0m(numel(f0m)+1:numel(TRsecs)-1)=0;
elseif numel(f0m)>numel(TRsecs)-1;f0m=f0m(1:numel(TRsecs)-1);end
% srate=100;
% fpt=srate*fsec;lpt=srate*lsec;c1=f0.F0_Hz(fpt:lpt);
% figure;plot(fpt:lpt,c1,'k');
% gg=gca;gg.XAxis.Visible='off';gg.YAxis.Visible='off';gg.XTick=[];gg.YTick=[];
% sn=[root 'pics/ProsEx-' num2str(game)];print(sn,'-dpdf','-bestfit');
    
