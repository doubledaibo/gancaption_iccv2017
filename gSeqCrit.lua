require 'nn'

local misc = require 'utils.misc'
local netUtils = require 'utils.netUtils'

local crit, parent = torch.class('nn.GSeqCrit', 'nn.Criterion')
function crit:__init()
	parent.__init(self)
	self.loger = nn.Sequential()
	self.loger:add(nn.Log())
	self.gradLog = torch.Tensor()
end

function crit:updateOutput(input, target)	
	self.gradInput:resizeAs(input):zero()
	local logInput = self.loger:forward(input)
	self.gradLog:resizeAs(input):zero()
	local L, batch_size, end_token = input:size(1), input:size(2), input:size(3)
	local D = target:size(1)
	assert(D == L)
	local loss = 0
	local n = 0
	for b = 1, batch_size do
		for t = 1, L do
			local target_index = target[{t, b}]
			if target_index ~= 0 then
				n = n + 1
				loss = loss - logInput[{t, b, target_index}]
				self.gradLog[{t, b, target_index}] = - 1			
			else
				break
			end
		end
	end
	self.output = loss / n
	self.gradLog:div(n)
	self.gradInput = self.loger:backward(input, self.gradLog)
	return self.output
end

function crit:updateGradInput(input, target)
	return self.gradInput
end
