# Code of [Towards Diverse and Natural Image Descriptions via a Conditional GAN](https://arxiv.org/abs/1703.06029)

Based on [Neuraltalk2](https://github.com/karpathy/neuraltalk2) and [SeqGAN](https://github.com/LantaoYu/SeqGAN). Special thanks to the authors!

Testing codes and related codes will be added gradually.

## Brief Explanation 

- Without images as conditions
	- utils/DataLoaderSeq.lua
	- utils/netUtilsSeq.lua
	- utils/netTrainSeq.lua
	- dSeq.lua
	- gSeq.lua
	- trainGANSeqPolicy.lua
	- Can be used to validate the codes for rollouts. If there are bugs, g will eventually generate nonsense

- With images as conditions, without finetuning the image encoder
	- utils/DataLoader.lua
	- utils/netUtils.lua
	- utils/netTrain.lua
	- dDotprod.lua
	- gNoise.lua
	- trainGANPolicyFeatNoise.lua


	 
