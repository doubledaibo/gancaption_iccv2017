local misc = require 'utils.misc'
local netTrain = {}

function netTrain.updateParams(optim_type, params, grad_params, learning_rate, optim_alpha, optim_beta, optim_epsilon, optim_state) 
	if optim_type == 'rmsprop' then
		rmsprop(params, grad_params, learning_rate, optim_alpha, optim_epsilon, optim_state)
	elseif optim_type == 'adagrad' then
		adagrad(params, grad_params, learning_rate, optim_epsilon, optim_state)
	elseif optim_type == 'sgd' then
		sgd(params, grad_params, learning_rate)
	elseif optim_type == 'sgdm' then
		sgdm(params, grad_params, learning_rate, optim_alpha, optim_state)
	elseif optim_type == 'sgdmom' then
		sgdmom(params, grad_params, learning_rate, optim_alpha, optim_state)
	elseif optim_type == 'adam' then
		adam(params, grad_params, learning_rate, optim_alpha, optim_beta, optim_epsilon, optim_state)
	else
		error('bad option optim type')		
	end
end

function netTrain.gTrain(loader, opt, protos)
	local data = loader:getSeqBatch({batch_size = opt.batch_size})
	local pred_g, dummy
	local grad_pred_g
	local loss
	if opt.gpuid >= 0 then data.seqs = data.seqs:cuda() end
	pred_g, dummy = unpack(protos.g.g:forward({data.seqs, protos.max_seq_length}))
	loss = protos.g_seq_crit:forward(pred_g, data.seqs)
	grad_pred_g = protos.g_seq_crit:backward(pred_g, data.seqs)
	dummy = protos.g.g:backward({data.seqs, protos.max_seq_length}, grad_pred_g)
	return loss
end

function netTrain.dTrain(loader, opt, protos, label)
	local data = loader:getSeqBatch({batch_size = opt.batch_size})
	if opt.gpuid >= 0 then data.seqs = data.seqs:cuda() end
	local dummy, gen_seq, pred_d, grad_pred_d
	local loss = 0
	dummy, gen_seq = unpack(protos.g.g:forward({opt.batch_size}))
	label:fill(1)
	pred_d = protos.d.d:forward(data.seqs)
	loss = loss + protos.gan_crit:forward(pred_d, label)
	grad_pred_d = protos.gan_crit:backward(pred_d, label)
	dummy = protos.d.d:backward(data.seqs, grad_pred_d)
	label:fill(0)
	pred_d = protos.d.d:forward(gen_seq)
	loss = loss + protos.gan_crit:forward(pred_d, label)
	grad_pred_d = protos.gan_crit:backward(pred_d, label)
	dummy = protos.d.d:backward(gen_seq, grad_pred_d)
	return loss		
end 

function netTrain.dEval(loader, opt, protos, label)
	local data = loader:getSeqBatch({batch_size = opt.batch_size})
	if opt.gpuid >= 0 then data.seqs = data.seqs:cuda() end
	local dummy, gen_seq, pred_d
	local loss = 0
	dummy, gen_seq = unpack(protos.g.g:forward({opt.batch_size}))
	label:fill(1)
	pred_d = protos.d.d:forward(data.seqs)
	loss = loss + protos.gan_crit:forward(pred_d, label)
	label:fill(0)
	pred_d = protos.d.d:forward(gen_seq)
	loss = loss + protos.gan_crit:forward(pred_d, label)
	return loss		
end 

function netTrain.gEval(loader, opt, protos, label)
	local dummy, gen_seq, pred_d
	local loss = 0
	dummy, gen_seq = unpack(protos.g.g:forward({opt.batch_size}))
	label:fill(1)
	pred_d = protos.d.d:forward(gen_seq)
	loss = loss + protos.gan_crit:forward(pred_d, label)
	return loss		
end 

function netTrain.getRolloutReward(rollout_policy, input_x, rollout_num, discriminator)
	local max_seq_length = input_x:size(1)
	local batch_size = input_x:size(2)
	local reward = input_x:clone():zero()
	for i = 1, rollout_num do
		for t = 1, max_seq_length - 1 do
			local dummy, rollout_samples = unpack(rollout_policy:forward({input_x, t}))
			local scores = discriminator:forward(rollout_samples)
			reward[t] = reward[t] + scores	
		end
		local scores = discriminator:forward(input_x)
		reward[max_seq_length] = reward[max_seq_length] + scores
	end 		
	reward = reward:div(rollout_num)
	return reward
end 

function netTrain.rlLearning(loader, opt, protos, rollout_policy)
	local prob_seq, gen_seq
	local rewards, loss
	local grad_prob_seq, dummy1, dummy2
	prob_seq, gen_seq = unpack(protos.g.g:forward({opt.batch_size}))
	rewards = netTrain.getRolloutReward(rollout_policy, gen_seq, opt.rollout_num, protos.d.d)
	loss = protos.policy_crit:forward({gen_seq, prob_seq, rewards}, nil)
	dummy1, grad_prob_seq, dummy2 = unpack(protos.policy_crit:backward({gen_seq, prob_seq, rewards}, nil))
	dummy1 = protos.g.g:backward(opt.batch_size, grad_prob_seq)
	return loss			
end

function netTrain.saveGCheckpoint(protos, opt, path)
	local checkpoint = {}
	checkpoint.protos = protos.g
	print("save g to " .. path)
	torch.save(path, checkpoint)
end

function netTrain.saveDCheckpoint(protos, opt, path)
	local checkpoint = {}
	checkpoint.protos = protos.d
	print("save d to " .. path)
	torch.save(path, checkpoint)
end
return netTrain 
