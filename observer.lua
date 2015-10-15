require 'optim'

--[[ options ]]--
opt = opt or {
  logPath = './'
}

--[[ information, or intermediate results]]--
local classes = {'1', '2'}
local info = {tr = {}, te = {}}
for key, _ in pairs(info) do
  info[key].err = {} -- error rate
  info[key].ell = {} -- loss value
  info[key].conf = optim.ConfusionMatrix(classes)
end

--[[ text log ]]--
local curdir = paths.dirname(paths.thisfile())
local logger = {
  err = optim.Logger(paths.concat(curdir, opt.logPath, 'error.log')),
  ell = optim.Logger(paths.concat(curdir, opt.logPath, 'loss.log'))
}
logger.err:setNames{'training error', 'testing error'}
logger.ell:setNames{'training loss', 'testing loss'}

print('[observer]')
print('log path: ' .. curdir)
print('\n')

return info, logger