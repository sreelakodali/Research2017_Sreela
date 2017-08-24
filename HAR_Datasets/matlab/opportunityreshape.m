% opportunity reshape

filename = 'OPP3.h5';
datasets = cell(1,6);
datasets{1} = opp1test;
datasets{2} = opp1test_targ;
datasets{3} = opp1train;
datasets{4} = opp1train_targ;
datasets{5} = opp1val;
datasets{6} = opp1val_targ;

path = cell(1,6);
path{1} = '/test/inputs';
path{2} = '/test/targets';
path{3} = '/training/inputs';
path{4} = '/training/targets';
path{5} = '/validation/inputs';
path{6} = '/validation/targets';

output = cell(1,6);

for i = 1:length(datasets)
    a = opportunity(datasets{i});
    disp(isinteger(a));
    output{i} = a;
    h5create(filename, path{i}, size(a));
    if (mod(i,2) == 0)
        h5write(filename, path{i}, int64(a));
    else
        h5write(filename, path{i}, a);
    end
end