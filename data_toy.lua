require('torch')

--[[ data specs ]]--
-- instance: n, M, C
-- labes: n
local trN, teN = 30, 10 -- #instances
local C = 8 -- #channels
local Mmin, Mmax = 12, 21 -- #words in a doc

--[[ training & testing raw data ]]--
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
local trX, trY = gen_rand_data(trN)
local teX, teY = gen_rand_data(teN)

--[[ training & testing data loader ]]--
do
  local loader = torch.class('toyLoader')

  function loader:__init(X, Y)
    assert(#X == Y:size(1))
    self.X = X
    self.Y = Y
    self.ind = torch.randperm(#X)
  end

  function loader:size()
    return #self.X
  end
  
  function loader:randperm_ind()
    self.ind = torch.randperm(#self.X)
  end

  function loader:get_datum(i)
    xx = self.X[i]:clone()
    yy = torch.FloatTensor(1):fill(self.Y[i])
    return xx, yy
  end
end

return toyLoader(trX, trY), toyLoader(teX, teY)