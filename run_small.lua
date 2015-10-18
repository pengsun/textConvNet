opt = {
  nThread = 2,
  logPath = 'log/imdb_small_C128', -- output path for log files
  dataSize = 'small',
  epMax = 100,  -- max epoches
  teFreq = 5, -- test every teFreq epoches
  isCuda = true,
  gpuInd = 2, -- gpu #
  C = 1024,   -- #channels
  V = 30000, -- #vocabulary
  fnData = 'data_imdb.lua', -- filie name for data generator
  fnModel = 'net_imdb.lua', -- file name for model
}

dofile('main.lua')