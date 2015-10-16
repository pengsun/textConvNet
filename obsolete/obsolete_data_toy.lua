--[[ toy data: randomly generated
--#data = n
--#words in doc = M (variable length)
data layout
--instance: {n} list, each entry: {M} tensor of indices to vocabulary
--labes: {n, 2}, 1-hot target for binary classification
]]--

require('torch')
require('dgLT')

--[[ options ]]--
opt = opt or {
  dataSize = 'small'
}

--[[ training & testing raw data ]]--
local trN, teN = 25000, 25000 -- #instances
local V = 3000 -- #vocabulary size
local Mmin, Mmax = 80, 120 -- #words in a doc

local gen_rand_data = function(n)
  -- instances
  local X = {}
  for i = 1, n do
    -- #words for the i-th doc
    local M = torch.floor( torch.uniform(Mmin, Mmax) ) 
    local tmp = torch.rand(M, 'torch.FloatTensor')
    X[i] = torch.ceil( tmp:mul(V) )
  end
  -- labels
  local tmp = torch.rand(n, 'torch.FloatTensor')
  local Y = torch.zeros(n, 2) -- binary 0/1 labels
  for i = 1, n do
    if tmp[i] < 0.5 then Y[i][1] = 1
    else Y[i][2] = 1 
    end
  end

  return X, Y
end

if opt.dataSize == 'small' then
  trN, teN = 100, 50
end

local trX, trY = gen_rand_data(trN)
local teX, teY = gen_rand_data(teN)

--[[ training & testing data generator ]]--
local tr, te = dgLT(trX, trY), dgLT(teX, teY)

print('[data]')
print('#tr = ' .. tr:size() .. ', #te = ' .. te:size())
print('\n')

return tr, te