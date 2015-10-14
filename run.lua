require 'optim'
require 'xlua'
require 'sys'

--[[ data ]]--
local data = dofile 'data_toy.lua'

--[[ net ]]--
local md, loss, set_numpool_one = dofile 'net_toy.lua'

--[[ optimization ]]--
local stOptim = {
  learningRate = 0.05,
  momentum = 0.005,
  learningRateDecay = 5e-7
}

--[[ observer: log, save... ]]-- TODO: a seperate file!
trConf = optim.confustionMatrix({'1', '2'})
ell = {}
trLogger = optim.Logger('train.log')
teLogger = optim.Logger('test.log')

--[[ train & test ]]--
epMax = 10
param, gradParam = md:getParameters()
for ep = 1, epMax do
  function train(data)
    print('traing epoch ' .. ep)
    data:randperm_ind()
    conf:zero()
    
    time = sys.tic()------------------------------
    for i = 1, data.size() do
      -- get one (instance, label) pair randomly
      j = data.ind[i]
      input, target = data.get_datum(i)
      
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
        nil
        -- other update
        conf:add(output, target)
      end
      
      -- update parameters
      optim.sgd(feval, param, stOptim)
      
      -- print
      -- TODO: print loss stuff?
      xlua.progress(i, data.size())
    end -- for i
    time = sys.tic(time)-------------------------
    
    -- print
    print(conf)
  end -- trian

  function test(data)
  end
  
  train()
  test()
end

