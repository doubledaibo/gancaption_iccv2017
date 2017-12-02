local misc = require 'utils.misc'
local netUtils = {}

function netUtils.decodeGSequence(seq, itow)
	local sentence = {}
	for i = 1, seq:size(1) do
		sentence[i] = itow[tostring(seq[{i, 1}])]
	end
	return sentence
end
	

function netUtils.buildCNN(cnn, opt) 
	local layer_num = misc.getOpt(opt, 'layer_num', 38)
	local backend = misc.getOpt(opt, 'backend', 'cudnn')
	local encoding_size = misc.getOpt(opt, 'encoding_size', 512)
	local cnn_out_size = misc.getOpt(opt, 'cnn_output_size', 4096)
	
	if backend == 'cudnn' then
		require 'cudnn'
		backend = cudnn
	else
		error(string.format('Unrecognized backend "%s"', backend))
	end

	local cnn_part = nn.Sequential()
	for i = 1, layer_num do
		local layer = cnn:get(i)
	--	if i == 1 then
	--		local w = layer.weight:clone()
	--		print('converting first layer conv filters from BGR to RGB...')
	--		layer.weight[{ {}, 1, {}, {} }]:copy(w[{ {}, 3, {}, {} }])
	--		layer.weight[{ {}, 3, {}, {} }]:copy(w[{ {}, 1, {}, {} }])
	--	end
		cnn_part:add(layer)
	end
	return cnn_part
end

function netUtils.prepro(img, opt, data_augment, on_gpu)
	assert(data_augment ~= nil)
	assert(on_gpu ~= nil)
	local h, w = img:size(3), img:size(4)
	local cnn_input_size = misc.getOpt(opt, 'cnn_input_size', 224)
	if h > cnn_input_size or w > cnn_input_size then
		local xoff, yoff 
		if data_augment then
			xoff, yoff = torch.random(w - cnn_input_size), torch.random(h - cnn_input_size)
		else
			xoff, yoff = math.ceil((w - cnn_input_size) / 2), math.ceil((h - cnn_input_size) / 2)
		end
		img = img[{ {}, {}, {yoff, yoff + cnn_input_size - 1}, {xoff, xoff + cnn_input_size - 1} }]
	end
	if on_gpu then img = img:cuda() else img = img:float() end
	
	if not netUtils.vgg_mean then
		netUtils.vgg_mean = torch.FloatTensor{123.68, 116.779, 103.939}:view(1, 3, 1, 1)
	end
	netUtils.vgg_mean = netUtils.vgg_mean:typeAs(img)
	img:add(-1, netUtils.vgg_mean:expandAs(img))
	return img			
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
	require 'dDotprod'
	local d
	local have_feat = string.len(misc.getOpt(opt, 'input_feat', '')) > 0
	if string.len(opt.d_start_from) > 0 then
		print("initializing weights from " .. opt.d_start_from)
		local loaded_checkpoint = torch.load(opt.d_start_from)
		d = {}
		d.d = loaded_checkpoint.protos.d
		if have_feat then return d end
		if loaded_checkpoint.protos.cnn ~= nil then
			d.cnn = loaded_checkpoint.protos.cnn
			netUtils.unsanitizeGradients(d.cnn)
		else
			d.cnn = netUtils.initCNN(opt, loader)
		end	
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
		gdOpt.cnn_output_size = opt.cnn_output_size
		d.d = nn.D(gdOpt)
		if have_feat then return d end
		d.cnn = netUtils.initCNN(opt, loader)
	end
	return d
end

function netUtils.getThinD(d)
	local thin_d = {}
	thin_d.d = d.d:clone()
	thin_d.d.core:share(d.d.core, 'weight', 'bias')
	thin_d.d.lookup_table:share(d.d.lookup_table, 'weight', 'bias')
	thin_d.d.img_proj:share(d.d.img_proj, 'weight', 'bias')
	thin_d.d.decoder:share(d.d.decoder, 'weight', 'bias')
	if d.cnn ~= nil then
		thin_d.cnn = d.cnn:clone('weight', 'bias')
	end
	return thin_d
end

function netUtils.getThinGNoise(g)
	local thin_g = {}
	thin_g.g = g.g:clone()
	thin_g.g.core:share(g.g.core, 'weight', 'bias')
	thin_g.g.lookup_table:share(g.g.lookup_table, 'weight', 'bias')
	thin_g.g.proj:share(g.g.proj, 'weight', 'bias')
	if g.cnn ~= nil then
		thin_g.cnn = g.cnn:clone('weight', 'bias')
	end
	return thin_g
end

function netUtils.initGModelNoise(opt, loader)
	require "gNoise"
	local g
	local have_feat = string.len(misc.getOpt(opt, 'input_feat', '')) > 0
	if string.len(opt.g_start_from) > 0 then
		print("initializing weights from " .. opt.g_start_from)
		local loaded_checkpoint = torch.load(opt.g_start_from)
		g = {}
		g.g = loaded_checkpoint.protos.g
		if have_feat then return g end
		if loaded_checkpoint.protos.cnn ~= nil then
			g.cnn = netUtils.initCNN(opt, loader)
		else
			g.cnn = loaded_checkpoint.protos.cnn
			netUtils.unsanitizeGradients(g.cnn)
		end			
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
		gdOpt.cnn_output_size = opt.cnn_output_size
		gdOpt.noise_size = opt.noise_size
		g.g = nn.G(gdOpt)
		if have_feat then return g end
		g.cnn = netUtils.initCNN(opt, loader)
	end
	return g
end

function netUtils.initCNN(opt, loader)
	local cnn_backend = opt.backend
	if opt.gpuid == -1 then cnn_backend = 'nn' end
	local cnn_raw = loadcaffe.load(opt.cnn_proto, opt.cnn_model, cnn_backend)
	return netUtils.buildCNN(cnn_raw, {cnn_output_size = opt.cnn_output_size, backend = cnn_backend})
end

return netUtils
