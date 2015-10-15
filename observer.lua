require 'optim'

--[[ options ]]--
opt = opt or {
  logPath = './'
}

--[[ information, or intermediate results]]--
local classes = {'1', '2'}
local info = {tr = {}, te = {}}
for key, _ in pairs(info) do
  info[key].err = {} 
  info[key].ell = {} 
  info[key].conf = optim.ConfusionMatrix(classes)
end

--[[ text log ]]--
local curdir = paths.dirname(paths.thisfile())
local logger = {
  err = optim.Logger(paths.concat(curdir, opt.logPath, 'error.log')),
  ell = optim.Logger(paths.concat(curdir, opt.logPath, 'loss.log'))
}
logger.err:setNames{'testing error'}
logger.ell:setNames{'training loss'}

print('[observer]')
print('log path: ' .. curdir)
print('\n')

return info, logger