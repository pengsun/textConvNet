require 'optim'

local classes = {'1', '2'}
local tmpl = {
  err = {}, -- error rate
  ell = {}, -- loss value
  conf = optim.ConfusionMatrix(classes)
}
local info = {tr = tmpl, te = tmpl}

local logger = {
  err = optim.Logger('error.log'),
  ell = optim.Logger('loss.log')
}
logger.err:setNames{'training error', 'testing error'}
logger.ell:setNames{'testing loss', 'testing loss'}

return info, logger