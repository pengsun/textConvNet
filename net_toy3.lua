require 'nn'

--[[ data flow & size specs ]]--
-- #vocabulary size = V (e.g., 30000)
-- #input channel = 256
-- #words in doc = M
--
--  a0:   M
--                  lookup table: V, C
--  a0':  M, C
--------------------------------------------
--                  conv: C, 256, 3
--  a1:   M', 256
--                  pool: 2, 2
--  a1':  M'', 256
--------------------------------------------
--                  conv: 256, 256, 3
--  a2:   M''',256
--                  Max: dim 1 
--  a2':  1, 256
--------------------------------------------
--                  Reshape: 256
--  a3:   256
--                  Linear: 256, 2
--  a3':  2
local V = 30000
local C = 512

--[[ model ]]--
local md = nn:Sequential()
-- word2vec Layer
md:add(nn.LookupTable(V, C)) -- (vocabulary size, #channeds)
-- ConvPool Layer I
md:add(nn.TemporalConvolution(C, 256, 3))
md:add(nn.ReLU())
md:add(nn.TemporalMaxPooling(2, 2))
-- ConvPool Layer II
md:add(nn.TemporalConvolution(256, 256, 3))
md:add(nn.ReLU())
md:add(nn.Max(1))
md:add(nn.Dropout(0.5))
-- Output Layer
md:add(nn.Reshape(256, false))
md:add(nn.Linear(256, 2)) -- binary classification
md:add(nn.LogSoftMax())
md:float()

--[[ loss ]]--
local loss = nn.ClassNLLCriterion()
loss:float()

print('[net]')
print(md)
print('\n')

return md, loss