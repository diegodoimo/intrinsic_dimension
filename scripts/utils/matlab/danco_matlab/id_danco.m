clear
addpath('/u/d/ddoimo/danco_test/idEstimation')
import idEstimation.*
maxNumCompThreads(8);


%cifar dataset
cifar = load('/u/d/ddoimo/danco_test/datasets/cifar_training.mat').images;
x = transpose(cast(cifar, "double"));


%mnist dataset
% load('/u/d/ddoimo/danco_test/datasets/mnist.mat')
% mnist = training.images(:, :, training.labels==1);
% mnist = training.images;
% x = reshape(mnist, 784, 60000);

%isolet
% isolet = load('/u/d/ddoimo/danco_test/datasets/isolet1+2+3+4.data');
% isolet5 = load('/u/d/ddoimo/danco_test/datasets/isolet5.data');
% isolet = cat(1, isolet, isolet5);
% x = transpose(isolet);

%isomap
% isomap = load('face_data.mat')
%x = isomap.images;

ndata = size(x, 2);

fraction =[1, 2, 4, 8, 16, 32, 64, 128, 256];
N = [];
times = [];
ids = [];
fileID  = fopen('DANCo_cifar_N.txt', 'w');
fprintf(fileID, '%6s %12s %12s\r\n','N','id','time');
for i = 1:length(fraction)
    
    n = floorDiv(ndata, fraction(i))
    
    
    for j =1:1
        subtrain = x(:, randperm(ndata, n));
        tic;
        id = DANCoFit(subtrain, 'fractal',true);
        toc;
        
        fprintf(fileID,'%6.0f %12.5f %12.5f\r\n', size(x, 1), id, toc);
    end
end 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Scale up features of MNIST dataset
% side_length = [7, 9, 14, 19, 28, 39, 56, 79, 112, 158, 224]
%side_length = [4, 5, 8, 11, 16, 22, 32, 45, 64, 90, 128, 181]
% P = [];
% times = [];
% ids = [];
% fileID  = fopen('DANCo_cifar_P.txt', 'w');
% fprintf(fileID, '%6s %12s %12s\r\n','P','id','time');
% 
% for i = 1:length(side_length)
%     p = side_length(i)
%     x = load(sprintf('/u/d/ddoimo/danco_test/datasets/cifar_cat_%dx%d.mat', p, p)).images;
% %     x = load(sprintf('/u/d/ddoimo/danco_test/datasets/mnist_ones_%dx%d.mat', p, p)).mnist_ones;
%     x = transpose(cast(x, "double"));
%     tic;
%     id = DANCoFit(x, 'fractal',true);
%     toc;
%     fprintf(fileID,'%6.0f %12.5f %12.5f\r\n', size(x, 1), id, toc);
% 
%     P = [P;size(i)];
%     ids = [ids;id];
%     times = [times;toc];
% %     
% end

% for i=1:length(size)
%     fprintf(fileID,'%6.0f %12.5f %12.5f\r\n', P(i), ids(i), times(i));
% end


