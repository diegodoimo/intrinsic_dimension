clear
addpath('/u/d/ddoimo/intrinsic_dimension/scripts/utils/matlab/idEstimation')
import idEstimation.*
maxNumCompThreads(8);

%syntetic datasets
%names = ['uniform20', 'normal']
%for name in names:
    


datasets = load('/u/d/ddoimo/intrinsic_dimension/datasets/datasets_16k_eps0.01.mat');
%whos('-file', '/u/d/ddoimo/intrinsic_dimension/datasets/matlab_datasets.mat')

fn = fieldnames(datasets);
for k=1:numel(fn)
    %print(k)
    %print(fn{k})
    x = datasets.(fn{k});
    x = transpose(cast(x, "double"));
    ndata = size(x, 2);

    fraction =[1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
    N = [];
    ids = [];
    fmt = "./results/syntetic/DANCo_16k_eps0.01_"+fn{k}+".txt";
    fileID  = fopen(fmt, 'w');
   
    for i = 1:length(fraction)
    
        n = floorDiv(ndata, fraction(i))
    
    
        for j =1:fraction(i)
            fraction(i);
            x_boots = x(:, randperm(ndata, n));
            id = DANCoFit(x_boots, 'fractal',true);
            fprintf(fileID,'%6.0f %12.5f\r\n', size(x_boots,2), id);
        end
    end
end


