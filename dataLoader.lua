local loader = torch.class('dataLoader')

function loader:__init(X, Y)
  assert(#X == Y:size(1))
  self.X = X
  self.Y = Y
  self.ind = torch.randperm(#X)
end

function loader:cuda()
  require'cutorch'
  for i = 1, #self.X do
    self.X[i] = self.X[i]:cuda()
  end
  self.Y = self.Y:cuda()
end

function loader:size()
  return #self.X
end

function loader:nchannel()
  return self.X[1]:size(2)
end

function loader:randperm_ind()
  self.ind = torch.randperm(#self.X)
end

function loader:get_datum(i)
  local xx = self.X[i]:clone()
  local yy = torch.FloatTensor(1):type(self.Y:type())
  yy:fill(self.Y[i])
  return xx, yy
end