require 'hdf5'
require 'math'

local misc = require 'utils.misc'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)
	print('DataLoader loading json file: ', opt.json_file)
	self.info = misc.readJson(opt.json_file) 
	self.word_size = misc.countKeys(self.info.wtoi)
	print('word size is ' .. self.word_size)
	
	print('DataLoader loading h5 file: ', opt.h5_file)
	self.h5_file = hdf5.open(opt.h5_file, 'r')
	local word_seq_size = self.h5_file:read('/word'):dataspaceSize()
	self.num_seq = word_seq_size[1]
	self.word_seq_length = word_seq_size[2]
	print('max word sequence length in data is ' .. self.word_seq_length)
	local image_shape = self.h5_file:read('/img'):dataspaceSize()
	assert(#image_shape == 4, '/images shuold be a 4D tensor')
	assert(image_shape[3] == image_shape[4], 'width and height must match')
	self.image_channels = image_shape[2]
	self.image_size = image_shape[3]		   
	self.num_img = image_shape[1]
	if opt.random == nil then
		self.random = true
	else
		self.random = opt.random
	end
	self:resetIterator()
end

function DataLoader:getDict()
	return self.info.itow
end

function DataLoader:resetIterator()
	self.iterator = 1
	if self.random then
		self.perm = torch.randperm(self.num_seq)
	else
		self.perm = torch.range(1, self.num_seq)
	end
end

function DataLoader:getWordSize()
	return self.word_size
end

function DataLoader:getWordSeqLength()
	return self.word_seq_length
end

function DataLoader:getNumSample()
	return self.num_seq
end

function DataLoader:getSeqBatch(opt)
	local batch_size = misc.getOpt(opt, 'batch_size', 16)
	local seq_batch = torch.LongTensor(batch_size, self.word_seq_length + 1):zero()
	for i = 1, batch_size do
		if self.iterator > self.num_seq then
			self:resetIterator()
		end
		local idx = self.perm[self.iterator]
		self.iterator = self.iterator + 1
		seq_batch[{{i, i}, {1, self.word_seq_length}}] = self.h5_file:read('/word'):partial({idx, idx}, {1, self.word_seq_length})
		for j = 2, self.word_seq_length + 1 do
			if seq_batch[i][j] == 0 then
				seq_batch[i][j] = self.word_size + 1
				break
			end
		end
	end
	local data = {}
	data.seqs = seq_batch:transpose(1, 2):contiguous()
	return data	
end
