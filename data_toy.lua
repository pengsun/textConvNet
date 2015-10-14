require('torch')

local n = 25 -- #instances
local C = 8 -- #channels
local Mmin, Mmax = 12, 21 -- #words in a doc

-- instances
X = {}
for i = 1, n do
  M = torch.floor( torch.uniform(Mmin, Mmax) ) 
  X[i] = torch.randn(C, M, 'torch.FloatTensor')
end
-- labels
Y = torch.FloatTensor(n):apply(
  function (elem)
    return torch.bernoulli(0.5)
  end)


-- loader
local dl_toy = {ind = torch.randperm(n)}

function dl_toy.size()
  return n
end

function dl_toy.randperm_ind()
  dl_toy.ind = torch.randperm(n)
end

function dl_toy.get_datum(i)
  return X[i], Y[i]
end

return dl_toy