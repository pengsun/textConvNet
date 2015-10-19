--[[ main logic to do training & testing ]]--

require 'optim'
require 'xlua'
require 'sys'

--[[ global options ]]--
opt = opt or {
  nThread = 2,
  logPath = 'log/imdb_small_toy', -- output path for log files
  dataSize = 'small',
  epMax = 10,  -- max epoches
  teFreq = 2, -- test every teFreq epoches
  isCuda = true,
  gpuInd = 1, -- gpu #
  C = 1024,   -- #channels
  V = 30000, -- #vocabulary
  fnData = 'data_imdb.lua', -- filie name for data generator
  fnModel = './net/toy3.lua', -- file name for model
  stOptim =  {
    learningRate = 1,
    learningRateDecay = 1e-7,
    weightDecay = 0.0005,
    momentum = 0.5,
  },
  shrinkFreq = 4, -- shrink every # iteration
}
print('[global options]')
print(opt)
if opt.isCuda then 
  require('cunn')
  print('switch to CUDA')
  cutorch.setDevice(opt.gpuInd)
  print('use GPU #' .. opt.gpuInd)
end
print('\n')

--[[ data ]]--
local trData, teData = dofile(opt.fnData)

--[[ net ]]--
local md, loss, print_flow = dofile(opt.fnModel)
if opt.isCuda then 
  md:cuda(); loss:cuda();
end

--[[ optimization ]]--
local stOptim = opt.stOptim or {
  learningRate = 1,
  learningRateDecay = 1e-7,
  weightDecay = 0.0005,
  momentum = 0.5,
}
local shrinkFreq = opt.shrinkFreq or 25 

--[[ observer: log, display... ]]
info, logger = dofile 'observer.lua'

--[[ train & test ]]--
local epMax = opt.epMax or 3
param, gradParam = md:getParameters()
for ep = 1, epMax do
  
  function train(data)
    print('training epoch ' .. ep)
    -- set training (enable dropout)
    md:training()
    -- random shuffling
    data:randperm_ind()
    -- reset/init info
    info.tr.conf:zero()
    info.tr.ell[ep] = 0
    -- shrink learningRate when necessary
    if ep % shrinkFreq == 0 then 
      print('perform learningRate shrinking...')
      stOptim.learningRate = stOptim.learningRate / 2
    end
    
    -- sgd over each datum
    local time = sys.tic()---------------------------
    for i = 1, data:size() do
      -- get one (instance, label) pair randomly
      local ii = data.ind[i]
      local input, target = data:get_datum(ii)
      if opt.isCuda then 
        input, target = input:cuda(), target:cuda()
      end
      
      -- closure doing all
      local feval = function (tmp)
        gradParam:zero()
        -- fprop
        local output = md:forward(input)
        local f = loss:forward(output, target)
        -- bprop
        local gradOutput = loss:backward(output, target)
        md:backward(input, gradOutput)
        
        -- TODO: L1 L2 penality
        
        -- update error, loss
        info.tr.conf:add(output, target[1])
        info.tr.ell[ep] = info.tr.ell[ep] + f
        
        -- print debug info
--        local str = '%d: out = (%f, %f), ' .. 
--                    'f = %f, acc ell = %f'
--        print(string.format(str, 
--            i, output[1], output[2],
--            f, info.tr.ell[ep]))
        --
        return f, gradParam
      end
      
      -- update parameters 
      optim.sgd(feval, param, stOptim)
      
      -- print
      xlua.progress(i, data:size())
      -- print debug info
      --print(input:size())
      --print_flow()
    end -- for i
    time = sys.toc(time)-----------------------------
    
    -- update error, loss
    info.tr.conf:updateValids()
    info.tr.err[ep] = 1 - info.tr.conf.totalValid
    info.tr.ell[ep] = info.tr.ell[ep] / data:size()
    -- print
    --print(info.tr.conf)
    print(string.format('ell = %f, err = %d %%',
        info.tr.ell[ep], info.tr.err[ep]*100))
    print(string.format('time = %ds, speed = %d data/s, or %f ms/data',
        time, data:size()/time, time/data:size()*1000))
  end -- trian

  function test(data)
    print('testing epoch ' .. ep)
    -- set testing (disable dropout)
    md:evaluate()
    -- reset/init info
    info.te.conf:zero()
    info.te.ell[ep] = 0
    
    -- test each datum
    local time = sys.tic()---------------------------------
    for i = 1, data:size() do
      -- get one (instance, label) pair 
      local input, target = data:get_datum(i)
      if opt.isCuda then 
        input, target = input:cuda(), target:cuda()
      end
      
      -- fprop
      local output = md:forward(input)
      local f = loss:forward(output, target)
  
      -- update error, loss
      info.te.conf:add(output, target[1])
      info.te.ell[ep] = info.te.ell[ep] + f
      
      -- print
      xlua.progress(i, data:size())
      -- print debug info
      --print(input:size())
      --print_flow()
    end -- for i
    time = sys.toc(time)-----------------------------------
    
    -- update error, loss
    info.te.conf:updateValids()
    info.te.err[ep] = 1 - info.te.conf.totalValid
    info.te.ell[ep] = info.te.ell[ep] / data:size()
    -- print
    print(info.te.conf)
    print(string.format('ell = %f, err = %d %%',
        info.te.ell[ep], info.te.err[ep]*100))
    print(string.format('time = %ds, speed = %d data/s, or %f ms/data',
        time, data:size()/time, time/data:size()*1000))
  end
  
  -- do training
  train(trData)
  
  -- do testing
  if ep % opt.teFreq == 0 then
    print('\n')
    test(teData)
  end
  print('\n')
  
  -- move stuff in info to logger
  logger.ell:add{info.tr.ell[ep]}
  logger.err:add{info.te.err[ep]}
  
  -- plot
  logger.ell:style{'lp'}; logger.ell:plot();
  logger.err:style{'-'}; logger.err:plot();
end -- for ep