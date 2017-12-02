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
	self.cnn_output_size = misc.getOpt(opt, 'cnn_output_size')
	self.noise_size = misc.getOpt(opt, 'noise_size')
	self.core = LSTM.lstm(self.input_encoding_size, self.vocab_size + 1, self.rnn_size, self.num_layers, self.dropout, false)
	self.lookup_table = nn.LookupTable(self.vocab_size + 1, self.input_encoding_size)
	self.proj = nn.Sequential()
	self.proj:add(nn.JoinTable(2, 2))
	self.proj:add(nn.Linear(self.cnn_output_size + self.noise_size, self.input_encoding_size))
	self.proj:add(nn.ReLU(true))
	self.sample_seq = torch.Tensor()
	self.prob_seq = torch.Tensor()
	self.zero_grad = torch.Tensor()
	self.end_mask = torch.Tensor()
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
end 

function layer:createClones()
	print('constructing clones inside the G model')
	self.clones = {self.core}	
	self.lookup_tables = {self.lookup_table}
	for t = 2, self.max_seq_length + 2 do
		self.clones[t] = self.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
		self.lookup_tables[t] = self.lookup_table:clone('weight', 'gradWeight')		
	end
end

function layer:getModulesList()
	return {self.core, self.lookup_table, self.proj}
end

function layer:parameters()
	local p1, g1 = self.core:parameters()
	local p2, g2 = self.lookup_table:parameters()
	local p3, g3 = self.proj:parameters()
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
	self.proj:training()
end

function layer:evaluate()
	if self.clones == nil then self:createClones() end
	for k, v in pairs(self.clones) do v:evaluate() end
	for k, v in pairs(self.lookup_tables) do v:evaluate() end
	self.proj:evaluate()
end

function layer:updateOutput(input)
	if self.clones == nil then self.createClones() end
	local batch_size
	local guidance = self.proj:forward(input[1]) --input[1] = {img, noise}
	if #input == 2 then
		batch_size = input[2]
	else
		batch_size = input[2]:size(2)
	end
	self.end_mask:resize(batch_size):fill(1)
	self.zero_grad:resize(batch_size, self.vocab_size + 1):zero()
	self.sample_seq:resize(self.max_seq_length + 1, batch_size):zero()
	self.prob_seq:resize(self.max_seq_length + 1, batch_size, self.vocab_size + 1):zero()
	local fix_num = 0
	if #input == 3 then
		fix_num = input[3]
		if fix_num > 0 then self.sample_seq[{{1, fix_num}, {}}] = input[2][{{1, fix_num}, {}}] end
	end			
	self:_createInitState(batch_size)
	self.state = {[0] = self.init_state}
	self.lookup_tables_inputs = {}
	self.inputs = {}
	self.tmax = 0
	for t = 1, self.max_seq_length + 2 do
		local xt, it, dummy
		local can_skip = false
		if t == 1 then
			xt = guidance
		elseif t == 2 then
			it = torch.LongTensor(batch_size):fill(self.vocab_size + 1)
			self.lookup_tables_inputs[t] = it
			xt = self.lookup_tables[t]:forward(it)
		else
			it = self.sample_seq[t - 2]:clone()
			if torch.sum(it) == 0 then
				can_skip = true
			else
				it[torch.eq(it, 0)] = self.vocab_size + 1
				self.lookup_tables_inputs[t] = it
				xt = self.lookup_tables[t]:forward(it)
			end
		end
		if not can_skip then
			self.inputs[t] = {xt, unpack(self.state[t - 1])}
			local out = self.clones[t]:forward(self.inputs[t])
			if t > 1 then
				self.prob_seq[t - 1] = out[self.num_state + 1]
				if t - 1 > fix_num then
					--sampling
					it = torch.multinomial(out[self.num_state + 1], 1):view(-1)
					it = torch.cmul(it, self.end_mask)
					self.sample_seq[t - 1] = it:clone()
				else
					it = self.sample_seq[t - 1]:clone()
				end
				self.end_mask[torch.eq(it, self.vocab_size + 1)] = 0
				self.tmax = t
			end
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
	local dguidance
	local dstate = {[self.tmax] = self.init_state}
	for t = self.tmax, 1, -1 do
		local dout = {}
		for k = 1, self.num_state do table.insert(dout, dstate[t][k]) end
		if t ~= 1 then
			table.insert(dout, gradOutput[t - 1])
		else
			table.insert(dout, self.zero_grad)
		end
		local dinputs = self.clones[t]:backward(self.inputs[t], dout)
		local dxt = dinputs[1]	
		if t ~= 1 then
			dstate[t - 1] = {}
			for k = 2, self.num_state + 1 do table.insert(dstate[t - 1], dinputs[k]) end
		end
		if t == 1 then
			dguidance = dxt
		else
			local it = self.lookup_tables_inputs[t]
			self.lookup_tables[t]:backward(it, dxt) 
		end
	end
	self.gradInput = self.proj:backward(input[1], dguidance)
	return self.gradInput	
end

function layer:sample(input, opt)
	local temperature = misc.getOpt(opt, 'temperature', 1.0)
	local epsilon = misc.getOpt(opt, 'epsilon', 0.5)
	local guidance = self.proj:forward(input)
	local batch_size, feat_dim = guidance:size(1), guidance:size(2)
	self:_createInitState(batch_size)
	local seq = torch.LongTensor(self.max_seq_length + 1, batch_size):zero()
	local seq_log_probs = torch.FloatTensor(self.max_seq_length, batch_size)
	local state = self.init_state
	local log_probs
	local end_mask = torch.LongTensor(batch_size):fill(1)	
	for t = 1, self.max_seq_length + 2 do
		local xt, it, sample_log_probs
		if t == 1 then
			xt = guidance
		elseif t== 2 then
			it = torch.LongTensor(batch_size):fill(self.vocab_size + 1)
			xt = self.lookup_table:forward(it)
		else
			local greedy = torch.rand(1)[1] < epsilon
			if greedy then
				sample_log_probs, it = torch.max(log_probs, 2)
				it = it:view(-1):long()
			else
				local prob = torch.exp(torch.div(log_probs, temperature))
				it = torch.multinomial(prob, 1)
				sample_log_probs = log_probs:gather(2, it)
				it = it:view(-1):long()
			end
			xt = self.lookup_table:forward(it)
		end
		if t >= 3 then
			it = torch.cmul(it, end_mask)
			sample_log_probs = sample_log_probs:view(-1):float()
			sample_log_probs[torch.eq(end_mask, 0)] = 0
			end_mask[torch.eq(it, self.vocab_size + 1)] = 0
			seq[t - 2] = it
			seq_log_probs[t - 2] = sample_log_probs
		end
		local inputs = {xt, unpack(state)}
		local out = self.core:forward(inputs)
		log_probs = torch.log(out[self.num_state + 1])
		state = {}
		for i = 1, self.num_state do table.insert(state, out[i]) end
	end	
	seq[self.max_seq_length + 1][torch.eq(end_mask, 1)] = self.vocab_size + 1
	return seq, seq_log_probs
end

function layer:sampleBeam(input, opt)
	local function compare(a, b) return a.p > b.p end
	local guidance = self.proj:forward(input) 
	local beam_size = misc.getOpt(opt, 'beam_size', 10)
	local batch_size, feat_dim = guidance:size(1), guidance:size(2)
--	assert(beam_size <= self.vocab_size + 1)			
	local seq = torch.LongTensor(self.max_seq_length + 1, batch_size):zero()	
	local new_seq = torch.LongTensor(self.max_seq_length + 1, 1):zero()
	local seq_log_probs = torch.FloatTensor(self.max_seq_length, batch_size)	
	self:_createInitState(beam_size)
	local beam_seq = torch.LongTensor(self.max_seq_length, beam_size)
	local beam_seq_log_probs = torch.FloatTensor(self.max_seq_length, beam_size)
	local beam_log_probs_sum = torch.zeros(beam_size)
	for k = 1, batch_size do
		beam_seq:zero()
		beam_seq_log_probs:zero()
		beam_log_probs_sum:zero()
		local state = self.init_state
		local log_probs
		local done_beams = {}
		local max_row = 1
		for t = 1, self.max_seq_length + 2 do
			local xt, it, sample_log_probs
			local new_state
			if t == 1 then
				local guidancek = guidance[{{k, k}}]:expand(beam_size, feat_dim)
				xt = guidancek
			elseif t == 2 then
				it = torch.LongTensor(beam_size):fill(self.vocab_size + 1)
				xt = self.lookup_table:forward(it)
			else
				local log_probs_f = log_probs:float()
				local ys, ix = torch.sort(log_probs_f, 2, true)
				local candidates = {}
				local cols = math.min(beam_size + 1, ys:size(2))
				local rows = max_row
				for c = 1, cols do
					for q = 1, rows do
--						if ix[{q, c}] ~= self.vocab_size then --Ignore UNK Token
							local local_log_prob = ys[{q, c}]
							local candidate_log_prob = beam_log_probs_sum[q] + local_log_prob
							table.insert(candidates, {c = ix[{q, c}], q = q, p = candidate_log_prob, r = local_log_prob})
--						end
					end
				end
				table.sort(candidates, compare)
				new_state = netUtils.cloneList(state)
				local beam_seq_prev, beam_seq_log_probs_prev
				if t > 3 then
					beam_seq_prev = beam_seq[{{1, t - 3}, {}}]:clone()
					beam_seq_log_probs_prev = beam_seq_log_probs[{{1, t - 3}, {}}]:clone()
				end
				max_row = 0
				for vix = 1, #candidates do
					local v = candidates[vix]
					local sv = max_row + 1
					if t > 3 then
						beam_seq[{{1, t - 3}, sv}] = beam_seq_prev[{{}, v.q}]
						beam_seq_log_probs[{{1, t - 3}, sv}] = beam_seq_log_probs_prev[{{}, v.q}]
					end
					for state_ix = 1, #new_state do
						new_state[state_ix][sv] = state[state_ix][v.q]
					end
					beam_seq[{t - 2, sv}] = v.c
					beam_seq_log_probs[{t - 2, sv}] = v.r
					beam_log_probs_sum[sv] = v.p
					if v.c == self.vocab_size + 1 or t == self.max_seq_length + 2 then
						new_seq:zero()
						new_seq[{{1, self.max_seq_length}, {}}] = beam_seq[{{}, sv}]:clone()
						new_seq[{t - 1, 1}] = self.vocab_size + 1
						table.insert(done_beams, 
							{seq = new_seq:clone(),
							log_p_seq = beam_seq_log_probs[{{}, sv}]:clone(),
							p = beam_log_probs_sum[sv]})
					else
						max_row = max_row + 1
					end
					if max_row == beam_size then break end
				end					
				it = beam_seq[t - 2]
				xt = self.lookup_table:forward(it)
			end
			if t ~= self.max_seq_length + 2 then
				if new_state then state = new_state end
				local inputs = {xt, unpack(state)}
				local out = self.core:forward(inputs)
				log_probs = torch.log(out[self.num_state + 1])
				state = {}
				for i = 1, self.num_state do table.insert(state, out[i]) end
			end 
		end	
		table.sort(done_beams, compare)
		seq[{{}, k}] = done_beams[1].seq
		seq_log_probs[{{}, k}] = done_beams[1].log_p_seq			
	end
	return seq, seq_log_probs
end


