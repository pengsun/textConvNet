  local loader = torch.class('dataLoader')

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