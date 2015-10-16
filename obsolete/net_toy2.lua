require 'nn'

--[[ data flow & size specs ]]--
-- #input channel = 1024
-- #words in doc = M
--
--  a0:   M, 1024
--------------------------------------------
--                  conv: 1024, 256, 3
--  a1:   M', 256
--                  pool: 2, 2
--  a1':  M'', 256
--------------------------------------------
--                  conv: 256, 256, 3
--  a2:   M''',256
--                  pool: M''', M'''
--  a2':  1, 256
--------------------------------------------
--                  Linear: 256, 2
--  a3:   1, 2


--[[ model ]]--
local md = nn:Sequential()
-- ConvPool Layer I
md:add(nn.TemporalConvolution(1024, 256, 3))
md:add(nn.ReLU())
md:add(nn.TemporalMaxPooling(2, 2))
-- ConvPool Layer II
md:add(nn.TemporalConvolution(256, 256, 3))
md:add(nn.ReLU())
md:add(nn.TemporalMaxPooling(1, 1)) -- will be changed dynamically
md:add(nn.Dropout(0.5))
-- Output Layer
md:add(nn.Linear(256, 2)) -- binary classification
md:add(nn.LogSoftMax())

--[[ loss ]]--
local loss = nn.ClassNLLCriterion()

--[[ manipulator ]]--
local iM = 6 -- module index for the pooling layer
local set_numpool_one = function (szInput)
  -- after ConvPool Layer I
  sz1 = torch.floor( (szInput - 3 + 1)/2 )
  -- after Conv Layer II
  sz = sz1 - 3 + 1
  -- output size 1 for Pool Layer II
  md.modules[iM].kW, md.modules[iM].dW = sz, sz 
end

print('[net]')
print(md)
print('\n')

return md, loss, set_numpool_one