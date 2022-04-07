
% Specify the folder where the files live.
myFolder = 'C:\Users\ebteh\Desktop\p1\Test data\EYASE';
% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(myFolder, '**/*.wav'); % Change to whatever pattern you need.
sounds = dir(filePattern);
num_files = length(sounds);
mydata = cell(1, num_files);
for k = 1 : num_files 
    baseFileName{k} = sounds(k).name;
    fullFileName = fullfile(sounds(k).folder, baseFileName{k});
    %fprintf(1, 'Now reading %s\n', fullFileName);
    % Now do whatever you want with this file name,
    [mydata{k}, Fs] = audioread(fullFileName);
    [c,l]=wavedec(mydata{k},4,'db4');

    % details coefficients
    [cd1{k},cd2{k},cd3{k},cd4{k}] = detcoef(c,l,[1,2,3,4]);

    % approxmiations coefficients
    cA4{k} = appcoef(c,l,'db4',4);

    % energy
    [Ea{k},Ed{k}]=wenergy(c,l);

    % wentropy
    went{k} = wentropy(mydata{k}, 'shannon')
end