addpath('C:\Users\ipboy\Documents\GitHub\SenLabModeling\pytoc')
output = load('output_compressed.mat');

output2 = squeeze(output.output);

figure;
spy(output2)
