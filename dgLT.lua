--[[ Data Generator for List of Tensor data
deviced for variable length sequential data
X: {n} List of Tensor, instances
Y: {n, K} Tensor, labels
]]--

local dg = torch.class('dgLT')

function dg:__init(X, Y)
  assert(#X == Y:size(1))
  assert(Y:dim() == 1, 'tensor Y must be in size {n}')
  self.X = X
  self.Y = Y
  self.ind = torch.randperm(#X)
end

function dg:cuda()
  require'cutorch'
  for i = 1, #self.X do
    self.X[i] = self.X[i]:cuda()
  end
  self.Y = self.Y:cuda()
end

function dg:size()
  return #self.X
end

function dg:randperm_ind()
  self.ind = torch.randperm(#self.X)
end

function dg:get_datum(i)
  local xx = self.X[i]:clone()
  local yy = torch.FloatTensor(1):type(self.Y:type())
  yy:fill(self.Y[i])
  return xx, yy
end