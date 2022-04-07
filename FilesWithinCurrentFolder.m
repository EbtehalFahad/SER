 sounds = dir('*.wav');
 num_files = length(sounds);
 mydata = cell(1, num_files);
 for n = 1:num_files 
     baseFileName {n} = sounds(n).name;
     [mydata{n},Fs] = audioread(sounds(n).name);
     [c,l]=wavedec(mydata{n},4,'db4');

     % details coefficients
     [cd1{n},cd2{n},cd3{n},cd4{n}] = detcoef(c,l,[1,2,3,4]);

     % approxmiations coefficients
     cA{n} = appcoef(c,l,'db4',4);

     [Ea{n},Ed{n}] = wenergy(c,l);

     %energie = {Ea, Ed};

     wentropyy{n} = wentropy(mydata{n}, 'shannon')
 end