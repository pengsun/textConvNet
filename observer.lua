require 'optim'

local classes = {'1', '2'}
--local tmpl = {
--  err = {}, -- error rate
--  ell = {}, -- loss value
--  conf = optim.ConfusionMatrix(classes)
--}
local info = {tr = {}, te = {}}
for key, _ in pairs(info) do
  info[key].err = {} -- error rate
  info[key].ell = {} -- loss value
  info[key].conf = optim.ConfusionMatrix(classes)
end

local logger = {
  err = optim.Logger('error.log'),
  ell = optim.Logger('loss.log')
}
logger.err:setNames{'training error', 'testing error'}
logger.ell:setNames{'training loss', 'testing loss'}

return info, logger