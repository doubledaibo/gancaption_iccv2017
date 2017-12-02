require 'nn'

local misc = require 'utils.misc'
local netUtils = require 'utils.netUtils'
local LSTM = require 'gLSTM'

local layer, parent = torch.class('nn.G', 'nn.Module')

function layer:__init(opt)
	parent.__init(self)
	self.vocab_size = misc.getOpt(opt, 'vocab_size')	
	self.input_encoding_size = misc.getOpt(opt, 'input_encoding_size')
	self.rnn_size = misc.getOpt(opt, 'rnn_size')
	self.num_layers = misc.getOpt(opt, 'num_layers', 1)
	self.dropout = misc.getOpt(opt, 'dropout', 0.5)
	self.max_seq_length = misc.getOpt(opt, 'max_seq_length')
	self.on_gpu = misc.getOpt(opt, 'gpuid', -1) >= 0
	self.core = LSTM.lstm(self.input_encoding_size, self.vocab_size + 1, self.rnn_size, self.num_layers, self.dropout, false)
	self.lookup_table = nn.LookupTable(self.vocab_size + 1, self.input_encoding_size)
	self.sample_seq = torch.Tensor()
	self.prob_seq = torch.Tensor()
end

function layer:_createInitState(batch_size)
	if not self.init_state then self.init_state = {} end
	local times = 2
	for h = 1, self.num_layers*times do
		self.init_state[h] = torch.zeros(batch_size, self.rnn_size)
		if self.on_gpu then
			self.init_state[h] = self.init_state[h]:cuda()
		end
	end
	self.num_state = #self.init_state
	self.end_mask = torch.ones(batch_size)
	if self.on_gpu then
		self.end_mask = self.end_mask:cuda()
	end
end 

function layer:createClones()
	print('constructing clones inside the G model')
	self.clones = {self.core}	
	self.lookup_tables = {self.lookup_table}
	for t = 2, self.max_seq_length + 1 do
		self.clones[t] = self.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
		self.lookup_tables[t] = self.lookup_table:clone('weight', 'gradWeight')		
	end
end

function layer:getModulesList()
	return {self.core, self.lookup_table}
end

function layer:parameters()
	local p1, g1 = self.core:parameters()
	local p2, g2 = self.lookup_table:parameters()
	local params = {}
	for k, v in pairs(p1) do table.insert(params, v) end
	for k, v in pairs(p2) do table.insert(params, v) end
	local grad_params = {}
	for k, v in pairs(g1) do table.insert(grad_params, v) end
	for k, v in pairs(g2) do table.insert(grad_params, v) end
	return params, grad_params
end

function layer:training() 
	if self.clones == nil then self:createClones() end 
	for k, v in pairs(self.clones) do v:training() end
	for k, v in pairs(self.lookup_tables) do v:training() end
end

function layer:evaluate()
	if self.clones == nil then self:createClones() end
	for k, v in pairs(self.clones) do v:evaluate() end
	for k, v in pairs(self.lookup_tables) do v:evaluate() end
end

function layer:updateOutput(input)
	if self.clones == nil then self.createClones() end
	local batch_size
	if #input == 1 then
		batch_size = input[1]
	else
		batch_size = input[1]:size(2)
	end
	self.sample_seq:resize(self.max_seq_length + 1, batch_size):zero()
	self.prob_seq:resize(self.max_seq_length + 1, batch_size, self.vocab_size + 1):zero()
	local fix_num = 0
	if #input == 2 then
		fix_num = input[2]
		if fix_num > 0 then self.sample_seq[{{1, fix_num}, {}}] = input[1][{{1, fix_num}, {}}] end
	end			
	self:_createInitState(batch_size)
	self.state = {[0] = self.init_state}
	self.lookup_tables_inputs = {}
	self.inputs = {}
	self.tmax = 0
	for t = 1, self.max_seq_length + 1 do
		local xt, it
		local can_skip = false
		if t == 1 then
			it = torch.LongTensor(batch_size):fill(self.vocab_size + 1)
			self.lookup_tables_inputs[t] = it
			xt = self.lookup_tables[t]:forward(it)
		else
			it = self.sample_seq[t - 1]:clone()
			if torch.sum(it) == 0 then
				can_skip = true
			else
				it[torch.eq(it, 0)] = 1
				self.lookup_tables_inputs[t] = it
				xt = self.lookup_tables[t]:forward(it)
			end
		end
		if not can_skip then
			self.inputs[t] = {xt, unpack(self.state[t - 1])}
			local out = self.clones[t]:forward(self.inputs[t])
			self.prob_seq[t] = out[self.num_state + 1]
			if t > fix_num then
				--sampling
				it = torch.multinomial(out[self.num_state + 1], 1):view(-1)
				it = torch.cmul(it, self.end_mask)
				self.sample_seq[t] = it:clone()
			else
				it = self.sample_seq[t]:clone()
			end
			self.end_mask[torch.eq(it, self.vocab_size + 1)] = 0
			self.tmax = t
			self.state[t] = {}
			for i = 1, self.num_state do table.insert(self.state[t], out[i]) end
		else	
			break
		end
	end
	self.output = {self.prob_seq, self.sample_seq}
	return self.output 	 						 
end

function layer:updateGradInput(input, gradOutput)
--	local dimg
	local dstate = {[self.tmax] = self.init_state}
	for t = self.tmax, 1, -1 do
		local dout = {}
		for k = 1, self.num_state do table.insert(dout, dstate[t][k]) end
		table.insert(dout, gradOutput[t])
		local dinputs = self.clones[t]:backward(self.inputs[t], dout)
		local dxt = dinputs[1]	
		if t ~= 1 then
			dstate[t - 1] = {}
			for k = 2, self.num_state + 1 do table.insert(dstate[t - 1], dinputs[k]) end
		end
--		if t == 1 then
--			dimg = dxt
--		else
			local it = self.lookup_tables_inputs[t]
			self.lookup_tables[t]:backward(it, dxt) 
--		end
	end
--	self.gradInput = dimg 
	self.gradInput = torch.Tensor()
	return self.gradInput	
end

