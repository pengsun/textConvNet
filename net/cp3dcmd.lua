--[[ Conv + Pool, times 2]]--

require 'nn'

opt = opt or {
  V = 30000,
  C = 512,
}

--[[ model ]]--
local V = opt.V
local C = opt.C
local md = nn:Sequential()
-- word2vec Layer
md:add(nn.LookupTable(V, C)) -- (vocabulary size, #channeds)
md:add(nn.Dropout(0.5))
-- ConvPool Layer I
md:add(nn.TemporalConvolution(C, 256, 3))
md:add(nn.ReLU(true))
md:add(nn.TemporalMaxPooling(3, 2))
md:add(nn.Dropout(0.5))
-- ConvPool Layer II
md:add(nn.TemporalConvolution(256, 256, 3))
md:add(nn.ReLU(true))
md:add(nn.Max(1))
md:add(nn.Dropout(0.5))
-- Output Layer
md:add(nn.Reshape(256))
md:add(nn.Linear(256, 2)) -- binary classification
md:add(nn.LogSoftMax())
md:float()

--[[ weight initialization ]]--

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