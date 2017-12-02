require 'torch'
require 'nn'
require 'nngraph'

require 'loadcaffe'

local netUtils = require 'utils.netUtils'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a adversarial network')
cmd:text()
cmd:text('Options')

cmd:option('-input_h5', 'coco_cap_dataset.h5', 'path to h5 file')
cmd:option('-input_json', 'coco_cap_mappings.json', 'path to json file storing dataset stats')
--cnn setting
cmd:option('-cnn_proto', 'vgg_deploy.prototxt', 'path to cnn prototxt file in Caffe format.')
cmd:option('-cnn_model', 'vgg_final.caffemodel', 'path to cnn model file, Caffe format.') 
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-feature_file', '', '')

cmd:text()

local opt = cmd:parse(arg)
--torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')

if opt.gpuid >= 0 then
	require 'cutorch'
	require 'cunn'
	require 'cudnn'
--	cutorch.manualSeed(opt.seed)
	cutorch.setDevice(opt.gpuid + 1)
end

require 'hdf5'
local h5_file = hdf5.open(opt.input_h5, "r")
local size = h5_file:read("/img"):dataspaceSize()
local nsample = size[1]
local cnn_backend_str = "cudnn"
if opt.gpuid == -1 then cnn_backend_str = 'nn' end
local cnn_raw = loadcaffe.load(opt.cnn_proto, opt.cnn_model, cnn_backend_str)
local cnn = netUtils.buildCNN(cnn_raw, {cnn_output_size = 4096, backend = cnn_backend_str})

local feature_tensor = torch.FloatTensor(nsample, 4096):zero()
for i = 1, nsample do
	local img = h5_file:read("/img"):partial({i, i}, {1, 3}, {1, 256}, {1,256})
	img = netUtils.prepro(img, {cnn_input_size = 224}, false, opt.gpuid >= 0)
	feature_tensor[i] = cnn:forward(img):float()
	if i % 500 == 0 then
		print(i .. "/" .. nsample)
	end
end
require 'hdf5'
local file = hdf5.open(opt.feature_file, "w")
file:write("/feature", feature_tensor)
file:close()

