require 'nn'

local misc = require 'utils.misc'
local netUtils = require 'utils.netUtils'
local dLSTM = require 'dLSTM'
local decoder = require 'decoderHn'

local layer, parent = torch.class('nn.D', 'nn.Module')

function layer:__init(opt)
	parent.__init(self)
	self.input_encoding_size = misc.getOpt(opt, 'input_encoding_size')
	self.vocab_size = misc.getOpt(opt, 'vocab_size')
	self.rnn_size = misc.getOpt(opt, 'rnn_size')
	self.num_layers = misc.getOpt(opt, 'num_layers', 1)
	local dropout = misc.getOpt(opt, 'dropout', 0)
	self.seq_length = misc.getOpt(opt, 'max_seq_length')
	self.on_gpu = misc.getOpt(opt, 'gpuid', -1) >= 0
	self.core = dLSTM.lstm(self.input_encoding_size, self.rnn_size, self.num_layers, dropout)
	self.lookup_table = nn.LookupTable(self.vocab_size + 1, self.input_encoding_size)
	self.decoder = decoder.decode(self.rnn_size, misc.getOpt(opt, 'hidden_sizes', {128}), dropout)
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
	self.decoder_input = torch.zeros(batch_size, self.rnn_size)
	if self.on_gpu then
		self.decoder_input = self.decoder_input:cuda()
	end
end 

function layer:createClones()
	print('constructing clones inside the D model')
	self.clones = {self.core}	
	self.lookup_tables = {self.lookup_table}
	for t = 2, self.seq_length + 1 do
		self.clones[t] = self.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
		self.lookup_tables[t] = self.lookup_table:clone('weight', 'gradWeight')		
	end
end

function layer:getModulesList()
	return {self.core, self.lookup_table, self.decoder}
end

function layer:parameters()
	local p1, g1 = self.core:parameters()
	local p2, g2 = self.lookup_table:parameters()
	local p3, g3 = self.decoder:parameters()
	local params = {}
	for k, v in pairs(p1) do table.insert(params, v) end
	for k, v in pairs(p2) do table.insert(params, v) end
	for k, v in pairs(p3) do table.insert(params, v) end
	local grad_params = {}
	for k, v in pairs(g1) do table.insert(grad_params, v) end
	for k, v in pairs(g2) do table.insert(grad_params, v) end
	for k, v in pairs(g3) do table.insert(grad_params, v) end 

	return params, grad_params
end

function layer:training() 
	if self.clones == nil then self:createClones() end 
	for k, v in pairs(self.clones) do v:training() end
	for k, v in pairs(self.lookup_tables) do v:training() end
	self.decoder:training()
end

function layer:evaluate()
	if self.clones == nil then self:createClones() end
	for k, v in pairs(self.clones) do v:evaluate() end
	for k, v in pairs(self.lookup_tables) do v:evaluate() end
	self.decoder:evaluate()
end

function layer:updateOutput(input)
	--in d, self.vocab_size + 1 meaning <eos>: end of sentence
	local seq = input
	self.batch_size = seq:size(2)
	if self.clones == nil then self.createClones() end	
	assert(seq:size(1) == self.seq_length + 1)
	self.output:resize(self.batch_size, 1)
	self:_createInitState(self.batch_size)
	
	self.state = {[0] = self.init_state}
	self.inputs = {}
	self.last_token = torch.LongTensor(self.batch_size):fill(0)
	for b = 1, self.batch_size do 
		self.last_token[b] = self.seq_length + 1
		for t = 2, self.seq_length + 1 do
			if seq[t][b] == 0 then
				self.last_token[b] = t - 1
				assert(seq[t - 1][b] == self.vocab_size + 1)
				break
			end
		end
	end
	self.lookup_tables_inputs = {}
	self.tmax = 0
	for t = 1, self.seq_length + 1 do
		local xt
		local can_skip = false
		local it = seq[t]:clone() 
--		it[torch.eq(self.last_token, t)] = self.vocab_size + 1
		if torch.sum(it) == 0 then
			can_skip = true
		else
			it[torch.eq(it, 0)] = self.vocab_size + 1
			self.lookup_tables_inputs[t] = it
			xt = self.lookup_tables[t]:forward(it)	
		end
		if not can_skip then
			self.tmax = t
			self.inputs[t] = {xt, unpack(self.state[t - 1])}
			local out = self.clones[t]:forward(self.inputs[t])
			self.state[t] = {}
			for i = 1, self.num_state do table.insert(self.state[t], out[i]) end
		else
			break
		end
	end
	for b = 1, self.batch_size do
		self.decoder_input[b] = self.state[self.last_token[b]][self.num_state][b]
	end
	local decoder_out = self.decoder:forward(self.decoder_input)
	self.output[{{}, {1, 1}}] = decoder_out[{{}, {1, 1}}]
	return self.output 				 
end

function layer:updateGradInput(input, gradOutput)
	local dstate = {[self.tmax] = self.init_state}
	local ddecoder = self.decoder:backward(self.decoder_input, gradOutput)	
	for t = self.tmax, 1, -1 do
		for b = 1, self.batch_size do
			if self.last_token[b] == t then
				dstate[t][self.num_state][b] = ddecoder[b]
			end
		end
		local dinputs = self.clones[t]:backward(self.inputs[t], dstate[t])
		local dxt = dinputs[1]	
		if t ~= 1 then
			dstate[t - 1] = {}
			for k = 2, self.num_state + 1 do table.insert(dstate[t - 1], dinputs[k]) end
		end
		local it = self.lookup_tables_inputs[t]
		self.lookup_tables[t]:backward(it, dxt)
	end
	self.gradInput = torch.Tensor()
	return self.gradInput	
end

