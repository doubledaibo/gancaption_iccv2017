local misc = require 'utils.misc'
local netUtils = {}

function netUtils.decodeGSequence(seq, itow)
	local sentence = {}
	for i = 1, seq:size(1) do
		sentence[i] = itow[tostring(seq[{i, 1}])]
	end
	return sentence
end

function netUtils.listNNGraphModules(g)
	local omg = {}
	for i, node in ipairs(g.forwardnodes) do
		local m = node.data.module
		if m then
			table.insert(omg, m)
		end
	end
	return omg
end 

function netUtils.listModules(net)
	local t = torch.type(net)
	local module_list
	if t == 'nn.gModule' then
		module_list = netUtils.listNNGraphModules(net)
	else
		module_list = net:listModules()
	end
	return module_list
end

function netUtils.sanitizeGradients(net)
	local module_list = netUtils.listModules(net)
	for k, m in ipairs(module_list) do
		if m.weight and m.gradWeight then
			m.gradWeight = nil
		end
		if m.bias and m.gradBias then
			m.gradBias = nil
		end
	end
end

function netUtils.unsanitizeGradients(net)
	local module_list = netUtils.listModules(net)
	for k, m in ipairs(module_list) do
		if m.weight and (not m.gradWeight) then
			m.gradWeight = m.weight:clone():zero()
		end
		if m.bias and (not m.gradBias) then
			m.gradBias = m.bias:clone():zero()
		end
	end 
end

function netUtils.cloneList(lst)
	local new = {}
	for k, v in pairs(lst) do
		new[k] = v:clone()
	end
	return new
end

function netUtils.initDModel(opt, loader)
	local d
	if string.len(opt.d_start_from) > 0 then
		print("initializing weights from " .. opt.d_start_from)
		local loaded_checkpoint = torch.load(opt.d_start_from)
		d = loaded_checkpoint.protos
	--	netUtils.unsanitizeGradients(d.cnn)
		local d_modules = d.d:getModulesList()
		for k, v in pairs(d_modules) do netUtils.unsanitizeGradients(v) end
	else
		d = {}
		local gdOpt = {}
		gdOpt.vocab_size = loader:getWordSize()
		gdOpt.input_encoding_size = opt.input_encoding_size
		gdOpt.rnn_size = opt.rnn_size
		gdOpt.num_layers = opt.num_layers
		gdOpt.dropout = opt.dropout
		gdOpt.max_seq_length = loader:getWordSeqLength() 
		gdOpt.gpuid = opt.gpuid
		d.d = nn.D(gdOpt)
--		local cnn_backend 
--		if opt.gpuid == -1 then cnn_backend = nn else require 'cudnn' cnn_backend = cudnn end
--		local cnn_backend_str = opt.backend
--		if opt.gpuid == -1 then cnn_backend_str = 'nn' end
--		require 'loadcaffe'
--		local cnn_raw = loadcaffe.load(opt.cnn_proto, opt.cnn_model, cnn_backend_str)
--		d.cnn = netUtils.buildCNN(cnn_raw, {cnn_output_size = opt.cnn_output_size, backend = cnn_backend_str})
--		d.cnn_proj = nn.Sequential()
--		d.cnn_proj:add(nn.Linear(opt.cnn_output_size, opt.input_encoding_size))
--		d.cnn_proj:add(cnn_backend.ReLU(true))
	end
	return d
end

function netUtils.initGModel(opt, loader)
	local g
	if string.len(opt.g_start_from) > 0 then
		print("initializing weights from " .. opt.g_start_from)
		local loaded_checkpoint = torch.load(opt.g_start_from)
		g = loaded_checkpoint.protos
--		netUtils.unsanitizeGradients(g.cnn)
		local g_modules = g.g:getModulesList()
		for k, v in pairs(g_modules) do netUtils.unsanitizeGradients(v) end
	else
		g = {}
		local gdOpt = {}
		gdOpt.vocab_size = loader:getWordSize()
		gdOpt.input_encoding_size = opt.input_encoding_size
		gdOpt.rnn_size = opt.rnn_size
		gdOpt.num_layers = opt.num_layers
		gdOpt.dropout = opt.dropout
		gdOpt.max_seq_length = loader:getWordSeqLength() 
		gdOpt.gpuid = opt.gpuid
		g.g = nn.G(gdOpt)
--		local cnn_backend 
--		if opt.gpuid == -1 then cnn_backend = nn else require 'cudnn' cnn_backend = cudnn end
--		local cnn_backend_str = opt.backend
--		if opt.gpuid == -1 then cnn_backend_str = 'nn' end
--		require 'loadcaffe'
--		local cnn_raw = loadcaffe.load(opt.cnn_proto, opt.cnn_model, cnn_backend_str)
--		g.cnn = netUtils.buildCNN(cnn_raw, {cnn_output_size = opt.cnn_output_size, backend = cnn_backend_str})
--		g.cnn_proj = nn.Sequential()
--		g.cnn_proj:add(nn.Linear(opt.cnn_output_size, opt.input_encoding_size))
--		g.cnn_proj:add(cnn_backend.ReLU(true))
	end
	return g
end

return netUtils
