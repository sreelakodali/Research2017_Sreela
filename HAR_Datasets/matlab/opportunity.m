function reshape = opportunity(dataset)

% initialize 2D matrix
reshape = zeros(size(dataset,2),30);

%endval
a = floor(size(dataset,1)/30);
endval = (30*a - 30)+1;
%disp(endval);

k = 1;
% increment i by 15
for i = 1:15:endval
    j = i + 15;
    windowA = dataset(i:i+14,:);
    windowB = dataset(j:j+14,:);
    x = [windowA; windowB];
    if (size(dataset,2) == 1)
        if (i == 1)
            reshape = transpose(x);
        else
            reshape = [reshape, transpose(x)];
        end
    else
    reshape(:,:,k) = transpose(x);
    end
    k = k + 1;
end

%disp(size(reshape));

end