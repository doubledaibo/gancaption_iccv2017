require 'torch'
require 'nn'
require 'nngraph'

require 'loadcaffe'

local misc = require 'utils.misc'
require 'utils.DataLoaderSeq'
require 'dSeq'
require 'gSeq'
local netUtils = require 'utils.netUtilsSeq'
local netTrain = require 'utils.netTrainSeq'
require 'utils.optimUpdates'
require 'policyCrit'
require 'gSeqCrit'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a adversarial network')
cmd:text()
cmd:text('Options')

cmd:option('-input_h5', 'coco_cap_dataset.h5', 'path to h5 file')
cmd:option('-val_input_h5', '', '')
cmd:option('-input_json', 'coco_cap_mappings.json', 'path to json file storing dataset stats')
--cnn setting
--cmd:option('-cnn_proto', 'vgg_deploy.prototxt', 'path to cnn prototxt file in Caffe format.')
--cmd:option('-cnn_model', 'vgg_final.caffemodel', 'path to cnn model file, Caffe format.') 
--data setting
cmd:option('-d_start_from', '', 'path to a d checkpoint, Empty = don\'t')
cmd:option('-g_start_from', '', 'path to a g checkpoint')
--model setting
cmd:option('-num_layers', 1, 'number of hidden layers in rnn units')
cmd:option('-rnn_size', 512, 'size of the rnn in number of hidden nodes in each layer')
cmd:option('-hidden_sizes', {64}, 'sizes of hidden layers in decoder')
cmd:option('-input_encoding_size', 512, 'the encoding size of each node')
--cmd:option('-cnn_input_size', 224, 'input size of image')
--cmd:option('-cnn_output_size', 4096, 'length of vector outputed by cnn')
--general
cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-batch_size', 16, 'what is the batch size for updating parameter')
cmd:option('-iter_size', 4, 'total samples per iter is iter_size * batch_size')
cmd:option('-grad_clip', 5, 'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
cmd:option('-dropout', 0.5, 'strength of dropout in the Language Model RNN')
--cmd:option('-finetune_cnn_after', -1, 'after what iteration do we start finetuning the CNN? (-1 = disable; 0 = finetune from start)')
--optimization
cmd:option('-optim', 'adam', 'what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate', 4e-4, 'learning rate')
cmd:option('-learning_rate_decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = don\'t)')
cmd:option('-learning_rate_decay_every', 50000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_alpha', 0.8, 'alpha for adagrad/rmsprop/mementum/adam')
cmd:option('-optim_beta', 0.999, 'beta used for adam')
cmd:option('-optim_epsilon', 1e-8, 'epsilon that goes into denominator for smoothing')
--optimization cnn
--cmd:option('-cnn_optim', 'adam', 'optimization to use for CNN')
--cmd:option('-cnn_optim_alpha', 0.8, 'alpha for momentum of CNN')
--cmd:option('-cnn_optim_beta', 0.999, 'alpha for mementum of CNN')
--cmd:option('-cnn_learning_rate', 1e-5, 'learning rate for the CNN')
--cmd:option('-cnn_weight_decay', 0, 'L2 weight decay just for the CNN')
--evaluation/checkpointing

cmd:option('-g_pre_nepoch', 5, '')
cmd:option('-d_pre_nepoch', 5, '')
cmd:option('-g_rl_niter', 1, '')
cmd:option('-d_rl_niter', 1, '')
cmd:option('-rl_niter', 100000, '')
cmd:option('-rollout_num', 16, '')

cmd:option('-save_checkpoint_every', 2500, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', '', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', '', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 2016, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')

cmd:text()

local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')

if opt.gpuid >= 0 then
	require 'cutorch'
	require 'cunn'
	require 'cudnn'
	cutorch.manualSeed(opt.seed)
	cutorch.setDevice(opt.gpuid + 1)
end

local loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json}
local val_loader = DataLoader{h5_file = opt.val_input_h5, json_file = opt.input_json, random = false}

local protos = {}
protos.gan_crit = nn.BCECriterion()
protos.d = netUtils.initDModel(opt, loader)
protos.g = netUtils.initGModel(opt, loader)
protos.policy_crit = nn.PolicyCrit(loader:getWordSize() + 1)
protos.g_seq_crit = nn.GSeqCrit()
protos.max_seq_length = loader:getWordSeqLength()

local label = torch.zeros(opt.batch_size, 1)

if opt.gpuid >= 0 then
--	cudnn.convert(protos.d.d.core, cudnn)
--	cudnn.convert(protos.d.d.lookup_table, cudnn)
--	cudnn.convert(protos.d.d.decoder, cudnn)
--	cudnn.convert(protos.g.g.core, cudnn)
--	cudnn.convert(protos.g.g.lookup_table, cudnn)
	for k, v in pairs(protos.d) do v:cuda() end
	for k, v in pairs(protos.g) do v:cuda() end
	protos.gan_crit = protos.gan_crit:cuda()
	protos.policy_crit = protos.policy_crit:cuda()	
	protos.g_seq_crit = protos.g_seq_crit:cuda()
	label = label:cuda()
end

local g_params, g_grad_params = protos.g.g:getParameters()
local d_params, d_grad_params = protos.d.d:getParameters()
--local g_cnn_params, g_cnn_grad_params = protos.g.cnn:getParameters()	
--local d_cnn_params, d_cnn_grad_params = protos.d.cnn:getParameters()
--local g_cnn_proj_params, g_cnn_proj_grad_params = protos.g.cnn_proj:getParameters()
--local d_cnn_proj_params, d_cnn_proj_grad_params = protos.d.cnn_proj:getParameters()

--print('total number of parameters in G:', g_params:nElement())
--print('total number of parameters in D:', d_params:nElement())
assert(g_params:nElement() == g_grad_params:nElement())
assert(d_params:nElement() == d_grad_params:nElement())

local policy_g = protos.g.g:clone()
local policy_g_params, policy_g_grad_params = policy_g:getParameters()
--thin_g.core:share(protos.g.g.core, 'weight', 'bias')
--netUtils.sanitizeGradients(thin_g.core)
--local thin_g_cnn = protos.g.cnn:clone('weight', 'bias')
--netUtils.sanitizeGradients(thin_g_cnn)

--local thin_d = protos.d.d:clone()
--thin_d.core:share(protos.d.d.core, 'weight', 'bias')
--netUtils.sanitizeGradients(thin_d.core)
--thin_d.decoder:share(protos.d.d.decoder, 'weight', 'bias')
--netUtils.sanitizeGradients(thin_d.decoder)
--thin_d.encoding:share(protos.d.d.encoding, 'weight')
--netUtils.sanitizeGradients(thin_d.encoding)
--local thin_d_cnn = protos.d.cnn:clone('weight', 'bias')
--netUtils.sanitizeGradients(thin_d_cnn)

protos.g.g:createClones()
protos.d.d:createClones()

collectgarbage()

local num_sample = loader:getNumSample()
local num_batch = torch.floor(num_sample / opt.batch_size)
-------------------------------------------
-------------------------------------------
--	local learning_rate = opt.learning_rate
--	local cnn_learning_rate = opt.cnn_learning_rate
--	if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
--		local frac = (iter - opt.finetune_cnn_after) / opt.learning_rate_decay_every
--		local decay_factor = math.pow(0.5, frac)
--		cnn_learning_rate = cnn_learning_rate * decay_factor
--		frac = (iter - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
--		decay_factor = math.pow(0.5, frac)
--		learning_rate = learning_rate * decay_factor
--	end
learning_rate = opt.learning_rate
if string.len(opt.g_start_from) == 0 then
	print("pretraining g...")
	local g_optim_state = {}
	local loss
	for epoch = 1, opt.g_pre_nepoch do
		protos.g.g:training()
		for i = 1, num_batch do
			g_grad_params:zero()
			loss = netTrain.gTrain(loader, opt, protos)
			if i % 10 == 0 then 
				print("epoch: " .. epoch .. ' / ' .. opt.d_pre_nepoch .. ', ' .. i .. ' / ' .. num_batch .. ', g loss: ' .. loss) 
				collectgarbage()
			end 	
			g_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
			netTrain.updateParams(opt.optim, g_params, g_grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, g_optim_state)
--		if i % 1000 == 0 then
--			loss = 0
--			protos.g.g:evaluate()
--			val_loader:resetIterator()
--			for i = 1, 10 do
--				loss = loss + netTrain.gTrain(val_loader, opt, protos)	
--			end			 
--			print("epoch: " .. epoch .. ' val g loss: ' .. loss / 10)
--			protos.g.g:training()
--		end
		end
	--val--
		loss = 0
		protos.g.g:evaluate()
		val_loader:resetIterator()
		for i = 1, 10 do
			loss = loss + netTrain.gTrain(val_loader, opt, protos)	
		end			 
		print("epoch: " .. epoch .. ' val g loss: ' .. loss / 10)
	end
	checkpoint_path = path.join(opt.checkpoint_path, 'model_' .. opt.id .. '_pre_g.t7')
	netTrain.saveGCheckpoint(protos, opt, checkpoint_path)
end

if string.len(opt.d_start_from) == 0 then
	print("pretraining d...")
	local d_optim_state = {}
	for epoch = 1, opt.d_pre_nepoch do
		protos.g.g:evaluate()
		protos.d.d:training()
		for i = 1, num_batch do
			d_grad_params:zero()
			loss = netTrain.dTrain(loader, opt, protos, label)
			if i % 10 == 0 then 
				print("epoch: " .. epoch .. ' / ' .. opt.d_pre_nepoch .. ', ' .. i .. ' / ' .. num_batch .. ', d loss: ' .. loss) 
				collectgarbage()
			end 	
			d_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
			netTrain.updateParams(opt.optim, d_params, d_grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, d_optim_state)
		end
		--val--
		loss = 0
		protos.d.d:evaluate()
		val_loader:resetIterator()
		for i = 1, 10 do
			loss = loss + netTrain.dEval(val_loader, opt, protos, label)
		end
		print("epoch: " .. epoch .. ' val d loss: ' .. loss / 10)
	end
	checkpoint_path = path.join(opt.checkpoint_path, 'model_' .. opt.id .. '_pre_d.t7')
	netTrain.saveDCheckpoint(protos, opt, checkpoint_path)
end

print("rl training...")
local g_optim_state = {}
local d_optim_state = {}
policy_g:evaluate()
policy_g_params:copy(g_params)
local gen_x
for iter = 1, opt.rl_niter do
	protos.g.g:training()
	protos.d.d:evaluate()
	for i = 1, opt.g_rl_niter do
		g_grad_params:zero()
		loss = netTrain.rlLearning(loader, opt, protos, policy_g)
		g_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
		netTrain.updateParams(opt.optim, g_params, g_grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, g_optim_state)
		print("iter: " .. iter .. ' / ' .. opt.rl_niter .. ', g loss: ' .. loss)
	end
	--val--
	loss = 0	
	protos.g.g:evaluate()
	val_loader:resetIterator()
	for i = 1, 10 do
		loss = loss + netTrain.gEval(val_loader, opt, protos, label)
	end
	print("epoch: " .. iter .. ' val g loss: ' .. loss / 10)
	-------
	policy_g_params:copy(g_params)
--	protos.g.g:evaluate()
	protos.d.d:training()
	for i = 1, opt.d_rl_niter do
		d_grad_params:zero()
		loss = netTrain.dTrain(loader, opt, protos, label)
		if i % 10 == 0 then 
			print("iter: " .. i .. ' / ' .. opt.d_rl_niter .. ',  d loss: ' .. loss) 
			collectgarbage()
		end 	
		d_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
		netTrain.updateParams(opt.optim, d_params, d_grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, d_optim_state)
	end	
	loss = 0	
	protos.d.d:evaluate()
	val_loader:resetIterator()
	for i = 1, 10 do
		loss = loss + netTrain.dEval(val_loader, opt, protos, label)
	end
	print("epoch: " .. iter .. ' val d loss: ' .. loss / 10)
	if iter % 1000 == 0 then
		checkpoint_path = path.join(opt.checkpoint_path, 'model_' .. opt.id .. '_g_iter' .. iter .. '.t7')
		netTrain.saveGCheckpoint(protos, opt, checkpoint_path)
		checkpoint_path = path.join(opt.checkpoint_path, 'model_' .. opt.id .. '_d_iter' .. iter .. '.t7')
		netTrain.saveDCheckpoint(protos, opt, checkpoint_path)
	end
end
