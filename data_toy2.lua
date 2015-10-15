require('torch')
require('dataLoader')

--[[ options ]]--
opt = opt or {
  dataSize = 'small'
}

--[[ training & testing raw data ]]--
-- #data = n
-- #words in doc = M
-- #channels(vocabulary size) = C
-- instance: n, M, C
-- labes: n
local trN, teN = 300, 300 -- #instances
local C = 1024 -- #channels
local Mmin, Mmax = 80, 120 -- #words in a doc

local gen_rand_data = function(n)
  X = {}
  for i = 1, n do
    M = torch.floor( torch.uniform(Mmin, Mmax) ) 
    X[i] = torch.randn(M, C, 'torch.FloatTensor')
  end
-- labels
  Y = torch.FloatTensor(n):apply(
    function (elem)
      return torch.bernoulli(0.5) + 1
    end)

  return X, Y
end

if opt.dataSize == 'small' then
  trN, teN = 10, 5
end

local trX, trY = gen_rand_data(trN)
local teX, teY = gen_rand_data(teN)

--[[ training & testing data loader ]]--
local tr, te = dataLoader(trX, trY), dataLoader(teX, teY)

print('[data]')
print('#tr = ' .. tr:size() .. ', #te = ' .. te:size())
assert(tr:nchannel()==te:nchannel())
print('#channels = ' .. tr:nchannel())
print('\n')

return tr, te