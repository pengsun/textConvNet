--[[ Conv + Pool, times 2, Full Connection, times 1]]--

require 'nn'

opt = opt or {
  V = 30000,
  C = 64,
}

--[[ model ]]--
local V = opt.V
local C = opt.C
local md = nn:Sequential()
-- word2vec Layer
md:add(nn.LookupTable(V, C)) -- (vocabulary size, #channeds)
-- ConvPool Layer I
md:add(nn.TemporalConvolution(C, 64, 3))
md:add(nn.ReLU())
md:add(nn.TemporalMaxPooling(2, 2))
-- ConvPool Layer II
md:add(nn.TemporalConvolution(64, 64, 3))
md:add(nn.ReLU())
md:add(nn.Max(1))
-- Full Connection Layer
md:add(nn.Reshape(64))
md:add(nn.Dropout(0.5))
md:add(nn.Linear(64, 64))
md:add(nn.Dropout(0.5))
-- Output Layer
md:add(nn.Linear(64, 2)) -- binary classification
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