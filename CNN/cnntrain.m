function net = cnntrain(net, x, y, opts)
    m = size(x, 3);
    numbatches = m / opts.batchsize;
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    net.rL = [];
    net.L = 998998; %初始 本次误差值 一个不可能的值
    i = 1;
    %训练超过指定次数或者误差小于指定误差，指定误差从 1/numbatches 到0是有意义的
    while ~((i>opts.numepochs) || ((i>numbatches)&&(sum(net.rL(end-numbatches+1:end))/numbatches < opts.error_limit)))
        time = tic;
        kk = randperm(m);
        for L = 1 : numbatches
            temp = tic;
            disp(['Epoch ' num2str(i) '/' num2str(opts.numepochs) ' batches ' num2str(L) '/' num2str(numbatches) '...']);
            batch_x = x(:, :, kk((L - 1) * opts.batchsize + 1 : L * opts.batchsize));
            batch_y = y(:,    kk((L - 1) * opts.batchsize + 1 : L * opts.batchsize));
            net = cnnff(net, batch_x);
            net = cnnbp(net, batch_y);
            net = cnnapplygrads(net, opts);
            if isempty(net.rL)
                net.rL(1) = net.L;
            end            
            net.rL(end + 1) = net.L;
            disp(['this batch ' num2str(toc(temp)) 's']);
        end
        disp(['                                  this Epoch ' num2str(toc(time)/60.) 'min'  ' rL=' num2str(net.rL(end))]);
        i = i+1;
    end
    
end
