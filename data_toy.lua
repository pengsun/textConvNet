require('torch')

-- data size
-- instance: n, M, C
-- labes: n
local n = 25 -- #instances
local C = 8 -- #channels
local Mmin, Mmax = 12, 21 -- #words in a doc

-- instances
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


-- loader
local dl_toy = {ind = torch.randperm(n)}

function dl_toy.size()
  return n
end

function dl_toy.randperm_ind()
  dl_toy.ind = torch.randperm(n)
end

function dl_toy.get_datum(i)
  xx = torch.FloatTensor(X[i]:size()):copy(X[i])
  yy = torch.FloatTensor(1):fill(Y[i])
  return xx, yy
end

return dl_toy