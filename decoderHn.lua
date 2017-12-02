require 'nn'
require 'nngraph'

local decoder = {}
function decoder.decode(input_size, hidden_sizes, dropout)
	dropout = dropout or 0
	local inputs = {}
	local outputs = {}
	table.insert(inputs, nn.Identity()())
	local x = inputs[1]
	local prev_size = input_size
	for i = 1, #hidden_sizes do
		if dropout > 0 then x = nn.Dropout(dropout)(x) end
		local fc = nn.Linear(prev_size, hidden_sizes[i])(x)
		x = nn.Tanh()(fc)		
		prev_size = hidden_sizes[i]
	end
	local prob = nn.Linear(prev_size, 1)(x)
	prob = nn.Sigmoid()(prob)	
	table.insert(outputs, prob)
	return nn.gModule(inputs, outputs)
end
return decoder
