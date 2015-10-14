require 'nn'

--[[ data size flow ]]--
-- #input channel = 8
-- #words in doc = W
--
--  W, 8
--       conv: in=8, out=16, size=3
--  W, 16
--       pool: size=W, stride=W (i.e., output size = 1)
--  1, 16
--       linear: in=16, out=2
--  1, 2
--

--[[ model ]]--
local szConv = 3
local md = nn:Sequential()
-- Conv Layer
md:add(nn.TemporalConvolution(8, 16, szConv))
md:add(nn.Tanh())
md:add(nn.TemporalMaxPooling(1))  -- kW, dW will be reset dynamically, 
                                  -- ensuring output size = 1
-- Output Layer
md:add(nn.Reshape(16))
md:add(nn.Linear(16, 2)) -- binary classification
md:add(nn.LogSoftMax())

--[[ loss ]]--
local loss = nn.ClassNLLCriterion()

--[[ manipulator ]]--
local iM = 3 -- module index for the pooling layer
local set_numpool_one = function (szInput)
  sz = szInput - szConv + 1 -- size for the output at last layer
  md.modules[iM].kW, md.modules[iM].dW = sz, sz -- size 1 for output
end

return md, loss, set_numpool_one