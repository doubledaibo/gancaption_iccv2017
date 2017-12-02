local misc = require 'utils.misc'
local utils = require 'utils.netUtils'
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

function netTrain.gTrainNoise(loader, opt, protos, noise)
	noise:fill(0)

--	local data = loader:getDataBatch({batch_size = opt.batch_size})
	local data = loader:getFeatBatch({batch_size = opt.batch_size})
--	data.images = utils.prepro(data.images, {cnn_input_size = opt.cnn_input_size}, true, opt.gpuid >= 0)
	local img_feat, pred_g, dummy
	local grad_pred_g, grad_img_feat
	local loss
	if opt.gpuid >= 0 then 
		data.seqs = data.seqs:cuda() 
		data.feats = data.feats:cuda()
	end
--	img_feat = protos.g.cnn:forward(data.images)
	img_feat = data.feats
	pred_g, dummy = unpack(protos.g.g:forward({{img_feat, noise}, data.seqs, protos.max_seq_length + 1}))
	loss = protos.g_seq_crit:forward(pred_g, data.seqs)
	grad_pred_g = protos.g_seq_crit:backward(pred_g, data.seqs)
	grad_img_feat, dummy = unpack(protos.g.g:backward({{img_feat, noise}, data.seqs, protos.max_seq_length}, grad_pred_g))
--	dummy = protos.g.cnn:backward(data.images, grad_img_feat)
	return loss
end

function netTrain.dTrainMMNoise(loader, opt, protos, label, noise) --additional mismatch batch
	noise:uniform(-1, 1)

--	local data = loader:getDataBatch({batch_size = opt.batch_size})
	local data = loader:getFeatBatch({batch_size = opt.batch_size})
--	data.images = utils.prepro(data.images, {cnn_input_size = opt.cnn_input_size}, true, opt.gpuid >= 0)
	if opt.gpuid >= 0 then 
		data.seqs = data.seqs:cuda() 
		data.feats = data.feats:cuda()
		data.mmseqs = data.mmseqs:cuda()
	end
	local dummy, gen_seq, pred_d, grad_pred_d
	local g_img_feat, d_img_feat, grad_d_img_feat, grad_g_img_feat
	local loss = 0
--	g_img_feat = protos.g.cnn:forward(data.images)
	g_img_feat = data.feats
	dummy, gen_seq = unpack(protos.g.g:forward({{g_img_feat, noise}, opt.batch_size}))
	label:fill(1)
--	d_img_feat = protos.d.cnn:forward(data.images)
	d_img_feat = data.feats
	pred_d = protos.d.d:forward({d_img_feat, data.seqs})
	loss = loss + protos.gan_crit:forward(pred_d, label)
	grad_pred_d = protos.gan_crit:backward(pred_d, label)
	grad_d_img_feat = protos.d.d:backward({d_img_feat, data.seqs}, grad_pred_d)
--	dummy = protos.d.cnn:backward(data.images, grad_d_img_feat)
	label:fill(0)
--	d_img_feat = protos.d.cnn:forward(data.images)
	d_img_feat = data.feats
	pred_d = protos.d.d:forward({d_img_feat, gen_seq})
	loss = loss + protos.gan_crit:forward(pred_d, label)
	grad_pred_d = protos.gan_crit:backward(pred_d, label)
	grad_d_img_feat = protos.d.d:backward({d_img_feat, gen_seq}, grad_pred_d)
--	dummy = protos.d.cnn:backward(data.images, grad_d_img_feat)
	label:fill(0)
--	d_img_feat = protos.d.cnn:forward(data.images)
	d_img_feat = data.feats
	pred_d = protos.d.d:forward({d_img_feat, data.mmseqs})
	loss = loss + protos.gan_crit:forward(pred_d, label)
	grad_pred_d = protos.gan_crit:backward(pred_d, label)
	grad_d_img_feat = protos.d.d:backward({d_img_feat, data.mmseqs}, grad_pred_d)
--	dummy = protos.d.cnn:backward(data.images, grad_d_img_feat)
	return loss		
end 

function netTrain.dEvalMMNoise(loader, opt, protos, label, noise)
--	local data = loader:getDataBatch({batch_size = opt.batch_size})
	local data = loader:getFeatBatch({batch_size = opt.batch_size})
--	data.images = utils.prepro(data.images, {cnn_input_size = opt.cnn_input_size}, true, opt.gpuid >= 0)
	if opt.gpuid >= 0 then 
		data.seqs = data.seqs:cuda() 
		data.feats = data.feats:cuda()
		data.mmseqs = data.mmseqs:cuda()
	end
	local dummy, gen_seq, pred_d, g_img_feat, d_img_feat
	local loss = 0
--	g_img_feat = protos.g.cnn:forward(data.images)
	g_img_feat = data.feats
	dummy, gen_seq = unpack(protos.g.g:forward({{g_img_feat, noise}, opt.batch_size}))
	label:fill(1)
--	d_img_feat = protos.d.cnn:forward(data.images)
	d_img_feat = data.feats
	pred_d = protos.d.d:forward({d_img_feat, data.seqs})
	loss = loss + protos.gan_crit:forward(pred_d, label)
	label:fill(0)
	pred_d = protos.d.d:forward({d_img_feat, gen_seq})
	loss = loss + protos.gan_crit:forward(pred_d, label)
	label:fill(0)
	pred_d = protos.d.d:forward({d_img_feat, data.mmseqs})
	loss = loss + protos.gan_crit:forward(pred_d, label)
	return loss		
end 

function netTrain.gEvalNoise(loader, opt, protos, label, noise)
--	local data = loader:getGenBatch({batch_size = opt.batch_size})
	local data = loader:getFeatBatch({batch_size = opt.batch_size})
--	data.images = utils.prepro(data.images, {cnn_input_size = opt.cnn_input_size}, true, opt.gpuid >= 0)
	local dummy, gen_seq, pred_d, d_img_feat, g_img_feat
	local loss = 0
	if opt.gpuid >= 0 then
		data.feats = data.feats:cuda()
	end
--	g_img_feat = protos.g.cnn:forward(data.images)
	g_img_feat = data.feats
--	dummy, gen_seq = unpack(protos.g.g:forward({g_img_feat, opt.batch_size}))
	gen_seq, dummy = protos.g.g:sampleBeam({g_img_feat, noise}, {beam_size = 1})
	label:fill(1)
--	d_img_feat = protos.d.cnn:forward(data.images)
	d_img_feat = data.feats
	pred_d = protos.d.d:forward({d_img_feat, gen_seq})
	loss = loss + protos.gan_crit:forward(pred_d, label)
	return loss		
end 

function netTrain.gEval2Noise(loader, opt, protos, rollout_policy, noise)
	local prob_seq, gen_seq, img_feat
	local rewards, loss
	local batch_size = 1
	if opt.rl_single ~= nil and opt.rl_single ~= 1 then
		batch_size = opt.batch_size
	end
--	local data = loader:getGenBatch({batch_size = opt.batch_size})
	local data = loader:getFeatBatch({batch_size = batch_size})
--	data.images = utils.prepro(data.images, {cnn_input_size = opt.cnn_input_size}, true, opt.gpuid >= 0)
--	img_feat = protos.g.cnn:forward(data.images)
	if opt.gpuid >= 0 then
		data.feats = data.feats:cuda()
	end
	img_feat = data.feats
	prob_seq, gen_seq = unpack(protos.g.g:forward({{img_feat, noise}, batch_size}))
--	rewards = netTrain.getRolloutReward(rollout_policy, gen_seq, data.images, opt.rollout_num, protos.d)
	rewards = netTrain.getRolloutRewardNoise(rollout_policy, gen_seq, data.feats, noise, opt.rollout_num, protos.d)
	loss = protos.policy_crit:forward({gen_seq, prob_seq, rewards}, nil)
	return loss			
end

function netTrain.getRolloutRewardNoise(rollout_policy, input_x, input_img, input_noise, rollout_num, discriminator)
	local max_seq_length = input_x:size(1)
	local batch_size = input_x:size(2)
	local reward = input_x:clone():zero()
--	local policy_img_feat = rollout_policy.cnn:forward(input_img)
	local policy_img_feat = input_img
--	local dis_img_feat = discriminator.cnn:forward(input_img)
	local dis_img_feat = input_img
	for i = 1, rollout_num do
		for t = 1, max_seq_length - 1 do
			local dummy, rollout_samples = unpack(rollout_policy.g:forward({{policy_img_feat, input_noise}, input_x, t}))
			local scores = discriminator.d:forward({dis_img_feat, rollout_samples})
			reward[t] = reward[t] + scores	
		end
		local scores = discriminator.d:forward({dis_img_feat, input_x})
		reward[max_seq_length] = reward[max_seq_length] + scores
	end 		
	reward = reward:div(rollout_num)
	return reward
end 

function netTrain.rlLearningNoise(loader, opt, protos, rollout_policy, noise)
	noise:uniform(-1, 1)

	local prob_seq, gen_seq, img_feat
	local rewards, loss
	local grad_prob_seq, dummy1, dummy2, grad_img_feat
	local batch_size = 1
	if opt.rl_single ~= nil and opt.rl_single ~= 1 then
		batch_size = opt.batch_size
	end
--	local data = loader:getGenBatch({batch_size = opt.batch_size})
	local data = loader:getFeatBatch({batch_size = batch_size})
--	data.images = utils.prepro(data.images, {cnn_input_size = opt.cnn_input_size}, true, opt.gpuid >= 0)
--	img_feat = protos.g.cnn:forward(data.images)
	if opt.gpuid >= 0 then
		data.feats = data.feats:cuda()
	end
	img_feat = data.feats
	prob_seq, gen_seq = unpack(protos.g.g:forward({{img_feat, noise}, batch_size}))
--	rewards = netTrain.getRolloutReward(rollout_policy, gen_seq, data.images, opt.rollout_num, protos.d)
	rewards = netTrain.getRolloutRewardNoise(rollout_policy, gen_seq, data.feats, noise, opt.rollout_num, protos.d)
	loss = protos.policy_crit:forward({gen_seq, prob_seq, rewards}, nil)
	dummy1, grad_prob_seq, dummy2 = unpack(protos.policy_crit:backward({gen_seq, prob_seq, rewards}, nil))
	grad_img_feat, dummy1 = protos.g.g:backward({{img_feat, noise}, opt.batch_size}, grad_prob_seq)
--	dummy1 = protos.g.cnn:backward(data.images, grad_img_feat)
	return loss			
end

function netTrain.saveGCheckpoint(protos, opt, path)
	local checkpoint = {}
	checkpoint.protos = protos.thin_g
	print("save g to " .. path)
	torch.save(path, checkpoint)
end

function netTrain.saveDCheckpoint(protos, opt, path)
	local checkpoint = {}
	checkpoint.protos = protos.thin_d
	print("save d to " .. path)
	torch.save(path, checkpoint)
end
return netTrain 
