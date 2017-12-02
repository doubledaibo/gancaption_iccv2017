local cjson = require 'cjson'
local misc = {}

function misc.computeEditDist(seq1, seq2)
	local l = seq1:size(1)
	local dist = torch.LongTensor(l + 1, l + 1):zero()
	local l1 = l
	for i = l, 1, -1 do
		if seq1[i][1] ~= 0 then
			l1 = i
			break
		end
	end 
	local l2 = l
	for i = l, 1, -1 do
		if seq2[i][1] ~= 0 then
			l2 = i
			break
		end
	end
	for i = 1, l + 1 do
		dist[i][1] = i - 1
		dist[1][i] = i - 1
	end
	for i = 1, l1 do
		for j = 1, l2 do
			if seq1[i][1] == seq2[j][1] then
				dist[i + 1][j + 1] = dist[i][j]
			else
				dist[i + 1][j + 1] = math.min(dist[i][j + 1] + 1, dist[i + 1][j] + 1, dist[i][j] + 1)
			end
		end
	end
	return dist[l1 + 1][l2 + 1]	
end

function misc.getOpt(opt, key, default_value)
	if default_value == nil and (opt == nil or opt[key] == nil) then
		error('error: required key ' .. key .. ' was not provided in an opt.')
	end
	if opt == nil then return default_value end
	local v = opt[key]
	if v == nil then v = default_value end
	return v
end

function misc.readJson(path)
	local file = io.open(path, 'r')
	local text = file:read()
	file:close()
	local info = cjson.decode(text)
	return info
end

function misc.countKeys(t)
	local n = 0
	for k, v in pairs(t) do
		n = n + 1
	end
	return n
end

function misc.writeJson(path, j)
	cjson.encode_sparse_array(true, 2, 10)
	local text = cjson.encode(j)
	local file = io.open(path, 'w')	
	file:write(text)
	file:close()
end

return misc

