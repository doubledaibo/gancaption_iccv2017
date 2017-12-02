require 'torch'
require 'nn'
require 'nngraph'

require 'loadcaffe'

local misc = require 'utils.misc'
require 'utils.DataLoader'
local netUtils = require 'utils.netUtils'
local netTrain = require 'utils.netTrain'
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
cmd:option('-input_feat', '', '')
cmd:option('-val_input_feat', '', '')
cmd:option('-input_json', 'coco_cap_mappings.json', 'path to json file storing dataset stats')
--cnn setting
cmd:option('-cnn_proto', 'vgg_deploy.prototxt', 'path to cnn prototxt file in Caffe format.')
cmd:option('-cnn_model', 'vgg_final.caffemodel', 'path to cnn model file, Caffe format.') 
--data setting
cmd:option('-d_start_from', '', 'path to a d checkpoint, Empty = don\'t')
cmd:option('-g_start_from', '', 'path to a g checkpoint')
--model setting
cmd:option('-num_layers', 1, 'number of hidden layers in rnn units')
cmd:option('-rnn_size', 512, 'size of the rnn in number of hidden nodes in each layer')
cmd:option('-hidden_sizes', {64}, 'sizes of hidden layers in decoder')
cmd:option('-input_encoding_size', 512, 'the encoding size of each node')
cmd:option('-cnn_input_size', 224, 'input size of image')
cmd:option('-cnn_output_size', 4096, 'length of vector outputed by cnn')
cmd:option('-noise_size', 100, 'length of noise vector')
--general
cmd:option('-g_pre_nepoch', 2, '')
cmd:option('-d_pre_nepoch', 2, '')
cmd:option('-rl_single', 1, '')
cmd:option('-g_rl_niter', 1, '')
cmd:option('-d_rl_niter', 50, '')
cmd:option('-max_iters', 100000, 'max number of iterations to run for')
cmd:option('-batch_size', 16, 'what is the batch size for updating parameter')
cmd:option('-iter_size', 4, 'total samples per iter is iter_size * batch_size')
cmd:option('-grad_clip', 5, 'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
cmd:option('-dropout', 0.5, 'strength of dropout in the Language Model RNN')
cmd:option('-finetune_cnn', 0, 'after what iteration do we start finetuning the CNN? (0 = disable; 1 = finetune from start)')
cmd:option('-rollout_momentum', 0, '')
cmd:option('-rollout_num', 64, '')
--optimization
cmd:option('-optim', 'adam', 'what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate', 4e-4, 'learning rate')
cmd:option('-learning_rate_decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = don\'t)')
cmd:option('-learning_rate_decay_every', 50000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_alpha', 0.8, 'alpha for adagrad/rmsprop/mementum/adam')
cmd:option('-optim_beta', 0.999, 'beta used for adam')
cmd:option('-optim_epsilon', 1e-8, 'epsilon that goes into denominator for smoothing')
--optimization cnn
cmd:option('-cnn_optim', 'adam', 'optimization to use for CNN')
cmd:option('-cnn_optim_alpha', 0.8, 'alpha for momentum of CNN')
cmd:option('-cnn_optim_beta', 0.999, 'alpha for mementum of CNN')
cmd:option('-cnn_learning_rate', 1e-5, 'learning rate for the CNN')
cmd:option('-cnn_weight_decay', 0, 'L2 weight decay just for the CNN')
--evaluation/checkpointing
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

local rl_loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json, feat_file = opt.input_feat}
local loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json, feat_file = opt.input_feat}
local val_loader = DataLoader{h5_file = opt.val_input_h5, json_file = opt.input_json, feat_file = opt.val_input_feat, random = false}

local protos = {}
protos.gan_crit = nn.BCECriterion()
protos.d = netUtils.initDModel(opt, loader)
protos.g = netUtils.initGModelNoise(opt, loader)
protos.policy_crit = nn.PolicyCrit(loader:getWordSize() + 1)
protos.g_seq_crit = nn.GSeqCrit()
protos.max_seq_length = loader:getWordSeqLength()

local label = torch.zeros(opt.batch_size, 1)
local noise = torch.zeros(opt.batch_size, opt.noise_size)

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
	noise = noise:cuda()
end

local val_noise = noise:clone()
val_noise:uniform(-1, 1)

local g_params, g_grad_params = protos.g.g:getParameters()
local d_params, d_grad_params = protos.d.d:getParameters()
--local g_cnn_params, g_cnn_grad_params = protos.g.cnn:getParameters()	
--local d_cnn_params, d_cnn_grad_params = protos.d.cnn:getParameters()

--print('total number of parameters in G:', g_params:nElement())
--print('total number of parameters in D:', d_params:nElement())
assert(g_params:nElement() == g_grad_params:nElement())
assert(d_params:nElement() == d_grad_params:nElement())
--assert(g_cnn_params:nElement() == g_cnn_grad_params:nElement())
--assert(d_cnn_params:nElement() == d_cnn_grad_params:nElement())

local policy = {}
policy.g = protos.g.g:clone()
--policy.cnn = protos.g.cnn:clone()
local policy_g_params, policy_g_grad_params = policy.g:getParameters()
--local policy_cnn_params, policy_cnn_grad_params = policy.cnn:getParameters()

protos.thin_d = netUtils.getThinD(protos.d)
protos.thin_g = netUtils.getThinGNoise(protos.g)
assert(protos.thin_d ~= nil)
assert(protos.thin_g ~= nil)

protos.g.g:createClones()
protos.d.d:createClones()

collectgarbage()

local num_sample = loader:getNumSample()
local num_batch = torch.floor(num_sample / (opt.batch_size * opt.iter_size))
--local num_batch = 1
-------------------------------------------
opt.d_report_interval = torch.ceil(opt.d_rl_niter / 3)
--opt.rollout_num = 64 
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
--cnn_learning_rate = opt.cnn_learning_rate
local d_optim_state = {}
--local d_cnn_optim_state = {}
local g_optim_state = {}
--local g_cnn_optim_state = {}
local loss

if string.len(opt.g_start_from) == 0 then
	print("pretraining g...")
	g_optim_state = {}
--	g_cnn_optim_state = {}
	for epoch = 1, opt.g_pre_nepoch do
		protos.g.g:training()
--		protos.g.cnn:training()
		for i = 1, num_batch do
			g_grad_params:zero()
--			g_cnn_grad_params:zero()
			loss = 0
			for k = 1, opt.iter_size do
				loss = loss + netTrain.gTrainNoise(loader, opt, protos, noise)	
			end
			loss = loss / opt.iter_size
			g_grad_params:div(opt.iter_size)
--			g_cnn_grad_params:div(opt.iter_size)
			if i % 10 == 0 then 
				print("epoch: " .. epoch .. ' / ' .. opt.g_pre_nepoch .. ', ' .. i .. ' / ' .. num_batch .. ', g loss: ' .. loss) 
				collectgarbage()
			end 	
--			g_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
			g_grad_params:mul(opt.grad_clip):div(math.max(g_grad_params:norm(), opt.grad_clip))
			netTrain.updateParams(opt.optim, g_params, g_grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, g_optim_state)
--			if opt.cnn_weight_decay > 0 then
--				g_cnn_grad_params:add(opt.cnn_weight_decay, g_cnn_params)
--			end
--			g_cnn_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
--			if opt.finetune_cnn == 1 then
--				netTrain.updateParams(opt.optim, g_cnn_params, g_cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, opt.cnn_optim_beta, opt.optim_epsilon, g_cnn_optim_state)
--			end	
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
--		protos.g.cnn:evaluate()
		val_loader:resetIterator()
		for i = 1, 10 do
			loss = loss + netTrain.gTrainNoise(val_loader, opt, protos, val_noise)	
		end			 
		print("epoch: " .. epoch .. ' val g loss: ' .. loss / 10)
	end
	checkpoint_path = path.join(opt.checkpoint_path, 'model_' .. opt.id .. '_pre_g.t7')
	netTrain.saveGCheckpoint(protos, opt, checkpoint_path)
end

if string.len(opt.d_start_from) == 0 then
	print("pretraining d...")
	d_optim_state = {}
--	d_cnn_optim_state = {}
	for epoch = 1, opt.d_pre_nepoch do
		protos.g.g:evaluate()
--		protos.g.cnn:evaluate()
		protos.d.d:training()
--		protos.d.cnn:evaluate()
		for i = 1, num_batch do
			d_grad_params:zero()
--			d_cnn_grad_params:zero()
			loss = 0
			for k = 1, opt.iter_size do
				loss = loss + netTrain.dTrainMMNoise(loader, opt, protos, label, noise)
			end
			loss = loss / opt.iter_size
			d_grad_params:div(opt.iter_size)
--			d_cnn_grad_params:div(opt.iter_size)
			if i % 10 == 0 then 
				print("epoch: " .. epoch .. ' / ' .. opt.d_pre_nepoch .. ', ' .. i .. ' / ' .. num_batch .. ', d loss: ' .. loss) 
				collectgarbage()
			end 	
--			d_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
			d_grad_params:mul(opt.grad_clip):div(math.max(opt.grad_clip, d_grad_params:norm()))
			netTrain.updateParams(opt.optim, d_params, d_grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, d_optim_state)
--			if opt.cnn_weight_decay > 0 then
--				d_cnn_grad_params:add(opt.cnn_weight_decay, d_cnn_params)
--			end
--			d_cnn_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
--			if opt.finetune_cnn == 1 then
--				netTrain.updateParams(opt.optim, d_cnn_params, d_cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, opt.cnn_optim_beta, opt.optim_epsilon, d_cnn_optim_state)
--			end	

		end
		--val--
		loss = 0
		protos.d.d:evaluate()
--		protos.d.cnn:evaluate()
		val_loader:resetIterator()
		for i = 1, 10 do
			loss = loss + netTrain.dEvalMMNoise(val_loader, opt, protos, label, val_noise)
		end
		print("epoch: " .. epoch .. ' val d loss: ' .. loss / 10)
	end
	checkpoint_path = path.join(opt.checkpoint_path, 'model_' .. opt.id .. '_pre_d.t7')
	netTrain.saveDCheckpoint(protos, opt, checkpoint_path)
end

print("rl training...")
g_optim_state = {}
--g_cnn_optim_state = {}
d_optim_state = {}
--d_cnn_optim_state = {}
policy.g:evaluate()
--policy.cnn:evaluate()
policy_g_params:copy(g_params)
--policy_cnn_params:copy(g_cnn_params)
local gen_x
for iter = 1, opt.max_iters do
	protos.g.g:training()
--	protos.g.cnn:training()
	protos.d.d:evaluate()
--	protos.d.cnn:evaluate()
	for i = 1, opt.g_rl_niter do
		g_grad_params:zero()
--		g_cnn_grad_params:zero()
		loss = 0
		for k = 1, opt.iter_size do
			loss = loss + netTrain.rlLearningNoise(rl_loader, opt, protos, policy, noise)
		end
		loss = loss / opt.iter_size
		g_grad_params:div(opt.iter_size)
--		g_cnn_grad_params:div(opt.iter_size)
--		g_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
		g_grad_params:mul(opt.grad_clip):div(math.max(g_grad_params:norm(), opt.grad_clip))
		netTrain.updateParams(opt.optim, g_params, g_grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, g_optim_state)
--		if opt.cnn_weight_decay > 0 then
--			g_cnn_grad_params:add(opt.cnn_weight_decay, g_cnn_params)
--		end
--		g_cnn_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
--		if opt.finetune_cnn == 1 then
--			netTrain.updateParams(opt.optim, g_cnn_params, g_cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, opt.cnn_optim_beta, opt.optim_epsilon, g_cnn_optim_state)
--		end	
		print("iter: " .. iter .. ' / ' .. opt.max_iters .. ', g loss: ' .. loss)
	end
	--val--
	if iter % 10 == 0 then
		loss = 0	
		protos.g.g:evaluate()
	--	protos.g.cnn:evaluate()
		val_loader:resetIterator()
		for i = 1, 1 do
			loss = loss + netTrain.gEval2Noise(val_loader, opt, protos, policy, val_noise)
		end
		print("epoch: " .. iter .. ' (expected reward) val g loss: ' .. loss / 1)
		loss = 0
		val_loader:resetIterator()
		for i = 1, 1 do
			loss = loss + netTrain.gEvalNoise(val_loader, opt, protos, label, val_noise)
		end
		print("epoch: " .. iter .. ' (best caption score) val g loss: ' .. loss / 1)
	end
	-------
	policy_g_params:mul(opt.rollout_momentum):add(1 - opt.rollout_momentum, g_params)
--	policy_cnn_params:mul(opt.rollout_momentum):add(1 - opt.rollout_momentum, g_cnn_params)
--	policy_g_params = policy_g_params * opt.rollout_momentum + g_params * (1 - opt.rollout_momentum)
--	policy_cnn_params = policy_cnn_params * opt.rollout_momentum + g_cnn_params * (1 - opt.rollout_momentum)
--	protos.g.g:evaluate()
	protos.d.d:training()
--	protos.d.cnn:training()
	for i = 1, opt.d_rl_niter do
		d_grad_params:zero()
--		d_cnn_grad_params:zero()
		loss = 0
		for k = 1, opt.iter_size do
			loss = loss + netTrain.dTrainMMNoise(loader, opt, protos, label, noise)
		end
		loss = loss / opt.iter_size
		d_grad_params:div(opt.iter_size)
--		d_cnn_grad_params:div(opt.iter_size)
		if i % opt.d_report_interval == 0 then 
			print("iter: " .. i .. ' / ' .. opt.d_rl_niter .. ',  d loss: ' .. loss) 
			collectgarbage()
		end 	
	--	d_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
		d_grad_params:mul(opt.grad_clip):div(math.max(opt.grad_clip, d_grad_params:norm()))
		netTrain.updateParams(opt.optim, d_params, d_grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, d_optim_state)
--		if opt.cnn_weight_decay > 0 then
--			d_cnn_grad_params:add(opt.cnn_weight_decay, d_cnn_params)
--		end
--		d_cnn_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
--		if opt.finetune_cnn == 1 then
--			netTrain.updateParams(opt.optim, d_cnn_params, d_cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, opt.cnn_optim_beta, opt.optim_epsilon, d_cnn_optim_state)
--		end	
	end	
	loss = 0	
	protos.d.d:evaluate()
--	protos.d.cnn:evaluate()
	val_loader:resetIterator()
	for i = 1, 10 do
		loss = loss + netTrain.dEvalMMNoise(val_loader, opt, protos, label, val_noise)
	end
	print("epoch: " .. iter .. ' val d loss: ' .. loss / 10)
	if iter % opt.save_checkpoint_every == 0 then
		checkpoint_path = path.join(opt.checkpoint_path, 'model_' .. opt.id .. '_g_iter' .. iter .. '.t7')
		netTrain.saveGCheckpoint(protos, opt, checkpoint_path)
		checkpoint_path = path.join(opt.checkpoint_path, 'model_' .. opt.id .. '_d_iter' .. iter .. '.t7')
		netTrain.saveDCheckpoint(protos, opt, checkpoint_path)
	end
end
