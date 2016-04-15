function net = cnntrain(net, x, y, opts)
    m = size(x, 3);
    numbatches = m / opts.batchsize;
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    net.rL = [];
    for i = 1 : opts.numepochs
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
            net.rL(end + 1) = 0.9 * net.rL(end) + 0.1 * net.L; % 还不清楚这个是什么，像卡尔曼
            disp(['this batch ' num2str(toc(temp)) 's ' 'rL=' num2str(net.rL(end))]);
        end
        disp(['                                  this Epoch ' num2str(toc(time)/60.) 'min']);
    end
    
end
