require 'nn'
require 'nngraph'
require 'utils.OneHot'
require 'utils.Clip'

local crit, parent = torch.class('nn.PolicyCrit', 'nn.Criterion')
function crit:__init(output_size)
	parent.__init(self)
	self.loss_net = self:create_loss_net(output_size)
	self.grad_loss = torch.Tensor(1)
	self.zero_mask = torch.Tensor()
end

function crit:updateOutput(input, target)
	self.batch_size = input[1]:size(2)
	self.zero_mask:resizeAs(input[1]):zero()
	self.zero_mask = self.zero_mask + 1
	self.masked_seq = input[1]:clone()
	self.zero_mask[torch.eq(input[1], 0)] = 0
	self.masked_seq[torch.eq(input[1], 0)] = input[2]:size(3)
	self.loss_inputs = {self.masked_seq, input[2], input[3], self.zero_mask}
--	self.a = self.loss_net:forward(self.loss_inputs)
--	self.output = - self.a / self.batch_size
	self.output = - self.loss_net:forward(self.loss_inputs) / self.batch_size
	return self.output[1]
end

function crit:updateGradInput(input, target)
	self.grad_loss[1] = -1.0 / self.batch_size
	local grad_seq, grad_prob, grad_reward, dummy = unpack(self.loss_net:backward(self.loss_inputs, self.grad_loss))
	return {grad_seq, grad_prob, grad_reward}
end

function crit:create_loss_net(output_size)
	local inputs = {}
	table.insert(inputs, nn.Identity()())
	table.insert(inputs, nn.Identity()())
	table.insert(inputs, nn.Identity()())
	table.insert(inputs, nn.Identity()())
	local seq = inputs[1]
	local prob = inputs[2]
	local reward = inputs[3]
	local mask = inputs[4]
	local reshape_seq = nn.View(-1)(seq)
	local onehot = nn.OneHot(output_size)(reshape_seq)
	local reshape_prob = nn.View(-1, output_size)(prob)
	local clip_prob = nn.Clip(1e-20, 1.0)(reshape_prob)
	local log_prob = nn.Log()(clip_prob)
	local log_prob_seq = nn.CMulTable()({log_prob, onehot})
	local sum_log_prob_seq = nn.Sum(2, 2)(log_prob_seq)
	local reshape_reward = nn.View(-1)(reward)
	local reshape_mask = nn.View(-1)(mask)
	local sum_reward_seq = nn.CMulTable()({sum_log_prob_seq, reshape_reward, reshape_mask})
	local loss = nn.Sum(1, 1)(sum_reward_seq)
	local outputs = {}
	table.insert(outputs, loss)
	return nn.gModule(inputs, outputs)
end

