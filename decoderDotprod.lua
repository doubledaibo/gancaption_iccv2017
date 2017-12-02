require 'nn'
require 'nngraph'

local decoder = {}
function decoder.decode()
	dropout = dropout or 0
	local inputs = {}
	local outputs = {}
	table.insert(inputs, nn.Identity()())
	table.insert(inputs, nn.Identity()())
	local img_feat = inputs[1]
	local cap_feat = inputs[2]
	local score = nn.DotProduct()({img_feat, cap_feat})
	score = nn.Sigmoid()(score)
	table.insert(outputs, score)
	return nn.gModule(inputs, outputs)
end
return decoder
