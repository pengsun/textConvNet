require 'optim'
require 'xlua'
require 'sys'

--[[ global options ]]--
opt = opt or {
  nThread = 2,
  logPath = 'log/one', -- output path for log files
  dataSize = 'small',
  epMax = 12,  -- max epoches
  teFreq = 3, -- test every teFreq epoches
  isCuda = false,
}
print('[global options]')
print(opt)
if opt.isCuda then 
  require('cunn')
  print('switch to CUDA')
end
print('\n')

--[[ data ]]--
local trData, teData = dofile'data_toy3.lua'

--[[ net ]]--
local md, loss, set_numpool_one = dofile'net_toy3.lua'
if opt.isCuda then 
  md:cuda(); loss:cuda();
end

--[[ optimization ]]--
local stOptim = {}
stOptim.learningRate = 0.05
stOptim.momentum = 0.005
stOptim.learningRateDecay = 5e-7

--[[ observer: log, display... ]]
info, logger = dofile 'observer.lua'

--[[ train & test ]]--
local epMax = opt.epMax or 3
param, gradParam = md:getParameters()
for ep = 1, epMax do
  
  function train(data)
    print('training epoch ' .. ep)
    -- random shuffling
    data:randperm_ind()
    -- reset/init info
    info.tr.conf:zero()
    info.tr.ell[ep] = 0
    
    -- sgd over each datum
    time = sys.tic()------------------------------
    for i = 1, data:size() do
      -- get one (instance, label) pair randomly
      ii = data.ind[i]
      input, target = data:get_datum(ii)
      if opt.isCuda then
        input, target = input:cuda(), target:cuda()
      end
      
      -- closure doing all
      local feval = function (tmp)
        gradParam:zero()
        -- fprop
        output = md:forward(input)
        f = loss:forward(output, target)
        -- bprop
        gradOutput = loss:backward(output, target)
        md:backward(input, gradOutput)
        -- TODO: L1 L2 penality
        -- update error, loss
        info.tr.conf:updateValids()
        info.tr.conf:add(output:squeeze(), target)
        info.tr.ell[ep] = info.tr.ell[ep] + f
        --
        return f, gradParam
      end
      
      -- update parameters 
      optim.sgd(feval, param, stOptim)
      
      -- print TODO: print loss stuff?
      xlua.progress(i, data:size())
    end -- for i
    time = sys.toc(time)-------------------------
    
    -- update error, loss
    info.tr.err[ep] = info.tr.conf.totalValid
    info.tr.ell[ep] = info.tr.ell[ep] / data:size()
    -- print
    print(info.tr.conf)
    print(string.format('time = %ds, speed = %d data/s, or %f ms/data',
        time, data:size()/time, time/data:size()*1000))
  end -- trian

  function test(data)
    print('testing epoch ' .. ep)
    -- reset/init info
    info.te.conf:zero()
    info.te.ell[ep] = 0
    
    -- test each datum
    time = sys.tic()------------------------------
    for i = 1, data:size() do
      -- get one (instance, label) pair 
      input, target = data:get_datum(i)
      if opt.isCuda then
        input, target = input:cuda(), target:cuda()
      end
      
      -- fprop
      output = md:forward(input)
      f = loss:forward(output, target)
      -- update error, loss
      info.te.conf:add(output:squeeze(), target)
      info.te.ell[ep] = info.tr.ell[ep] + f
      
      -- print TODO: print loss stuff?
      xlua.progress(i, data:size())
    end -- for i
    time = sys.toc(time)--------------------------
    
    -- update error, loss
    info.te.conf:updateValids()
    info.te.err[ep] = info.te.conf.totalValid
    info.te.ell[ep] = info.te.ell[ep] / data:size()
    -- print
    print(info.te.conf)
    print(string.format('time = %ds, speed = %d data/s, or %f ms/data',
        time, data:size()/time, time/data:size()*1000))    
  end
  
  train(trData)
  if ep % opt.teFreq == 0 then
    test(teData)
  end
  print('\n')
  
  -- move stuff in info to logger
  logger.ell:add{info.tr.ell[ep]}
  logger.err:add{info.te.err[ep]}
  
  -- plot
  logger.ell:style{'lp'}; logger.ell:plot();
  logger.err:style{'-'}; logger.err:plot()
end -- for ep