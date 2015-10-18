--[[ Conv + Conv + Pool, times 2]]--

--[[ data flow & size specs 
-- #vocabulary size = V (e.g., 30000)
-- #input channel = C
-- #words in doc = M (e.g., 85)

85
------------------------------------
        lookup table: V, C
85, C
        pool: 2
------------------------------------
42, C
        conv: C, 64, 2
41, 64
        conv: 64, 64, 2
40, 64
        pool: 2
------------------------------------
20, 64
        conv: 64, 64, 2
19, 64
        conv: 64, 64, 2
18, 64
        max: dim1
------------------------------------
1, 64
        reshape: 64
64
        Dropout: 0.5
64
        Linear: 64, 64      
------------------------------------       
64
        Linear: 64, 2
------------------------------------
2
]]--

require 'nn'

opt = opt or {
  V = 30000,
  C = 128,
}

--[[ model ]]--
local V = opt.V
local C = opt.C
local md = nn:Sequential()
-- word2vec Layer
md:add(nn.LookupTable(V, C)) -- (vocabulary size, #channeds)
md:add(nn.TemporalMaxPooling(2,2))
-- ConvConvPool Layer I
md:add(nn.TemporalConvolution(C, 64, 2))
md:add(nn.ReLU())
md:add(nn.TemporalConvolution(64, 64, 2))
md:add(nn.ReLU())
md:add(nn.TemporalMaxPooling(2, 2))
-- ConvConvPool Layer II
md:add(nn.TemporalConvolution(64, 64, 2))
md:add(nn.ReLU())
md:add(nn.TemporalConvolution(64, 64, 2))
md:add(nn.ReLU())
md:add(nn.Max(1))
-- full connection
md:add(nn.Reshape(64))
md:add(nn.Dropout(0.5))
md:add(nn.Linear(64, 64))
md:add(nn.ReLU())
-- Output Layer
md:add(nn.Linear(64, 2))
md:add(nn.LogSoftMax())
md:float()

--[[ loss ]]--
local loss = nn.ClassNLLCriterion()
--local loss = nn.CrossEntropyCriterion()
loss:float()

--[[ manipulators ]]--
local print_size = function()
  print('model data size flow:')
  -- Modules
  local tmpl = '(%d): %s %s'
  for i = 1, #md.modules do
    local str = string.format(tmpl, i, 
      md.modules[i]:__tostring(),
      md.modules[i].output:size():__tostring__())
    print(str)
  end
  print('\n')
end


print('[net]')
print(md)
print('\n')

return md, loss, print_size