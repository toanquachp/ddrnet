import math
import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH_NORM_MOM = 0.1

def conv3x3(in_channels, out_channels, stride=1):
	return nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)


class Block(nn.Module):

	EXPANSION = 1

	def __init__(self, in_channels, out_channels, stride=1, downsample=None, no_relu=False):
		super(Block, self).__init__()
		self.conv_1 = conv3x3(in_channels, out_channels, stride)
		self.bn_1 = nn.BatchNorm2d(out_channels, momentum=BATCH_NORM_MOM)
		self.relu = nn.ReLU(inplace=True)
		self.conv_2 = conv3x3(out_channels, out_channels)
		self.bn_2 = nn.BatchNorm2d(out_channels, momentum=BATCH_NORM_MOM)
		self.downsample = downsample
		self.stride = stride
		self.no_relu = no_relu
	
	def forward(self, x):
		residual = x

		out = self.conv_1(x)
		out = self.bn_1(out)
		out = self.relu(out)
  
		out = self.conv_2(out)
		out = self.bn_2(out)
  
		if self.downsample is not None:
			residual = self.downsample(x)
	
		out += residual
  
		if self.no_relu:
			return out
		else:
			return self.relu(out)


class BottleNeck(nn.Module):
	EXPANSION = 2
	def __init__(self, in_channels, out_channels, stride=1, downsample=None, no_relu=True):
		'''
		Layer conv5_1 (2nd half)
		'''
		super(BottleNeck, self).__init__()
		
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channels, momentum=BATCH_NORM_MOM)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channels, momentum=BATCH_NORM_MOM)
		self.conv3 = nn.Conv2d(out_channels, out_channels * self.EXPANSION, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(out_channels * self.EXPANSION, momentum=BATCH_NORM_MOM)
		
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride
		self.no_relu = no_relu
	
	def forward(self, x):
		residual = x

		out = self.conv1(residual)
		out = self.bn1(out)
		out = self.relu(out)
  
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)
	
		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)
   
		out += residual
  
		if self.no_relu:
			return out
		else:
			return self.relu(out)


class DAPPM(nn.Module):
	def __init__(self, in_channels, branch_channels, out_channels):
		super(DAPPM, self).__init__()
  
		self.scale_1 = nn.Sequential(
			nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
			nn.BatchNorm2d(in_channels, momentum=BATCH_NORM_MOM),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False)
		)
  
		self.scale_2 = nn.Sequential(
			nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
			nn.BatchNorm2d(in_channels, momentum=BATCH_NORM_MOM),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False)
		)
  
		self.scale_3 = nn.Sequential(
			nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
			nn.BatchNorm2d(in_channels, momentum=BATCH_NORM_MOM),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False)
		)
  
		self.scale_4 = nn.Sequential(
			# nn.AdaptiveAvgPool2d((1, 1)),
			nn.BatchNorm2d(in_channels, momentum=BATCH_NORM_MOM),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False)
		)
  
		self.scale_0 = nn.Sequential(
			nn.BatchNorm2d(in_channels, momentum=BATCH_NORM_MOM),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False)
		)
  
		self.process_1 = nn.Sequential(
			nn.BatchNorm2d(branch_channels, momentum=BATCH_NORM_MOM),
			nn.ReLU(inplace=True),
			nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1, bias=False)
		)
  
		self.process_2 = nn.Sequential(
			nn.BatchNorm2d(branch_channels, momentum=BATCH_NORM_MOM),
			nn.ReLU(inplace=True),
			nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1, bias=False)
		)
  
		self.process_3 = nn.Sequential(
			nn.BatchNorm2d(branch_channels, momentum=BATCH_NORM_MOM),
			nn.ReLU(inplace=True),
			nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1, bias=False)
		)
  
		self.process_4 = nn.Sequential(
			nn.BatchNorm2d(branch_channels, momentum=BATCH_NORM_MOM),
			nn.ReLU(inplace=True),
			nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1, bias=False)
		)
  
		self.compression = nn.Sequential(
			nn.BatchNorm2d(branch_channels * 5, momentum=BATCH_NORM_MOM),
			nn.ReLU(inplace=True),
			nn.Conv2d(branch_channels * 5, out_channels, kernel_size=1, bias=False)
		)
  
		self.shortcut = nn.Sequential(
			nn.BatchNorm2d(in_channels, momentum=BATCH_NORM_MOM),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
		)
  
	def forward(self, x):
		width = x.shape[-1]
		height = x.shape[-2]

		x_list = []
		
		x_list.append(self.scale_0(x))
		
		x_list.append(self.process_1((F.interpolate(
			self.scale_1(x),
			size=[height, width],
			mode='bilinear')+x_list[0])))
  
		x_list.append(self.process_2((F.interpolate(
			self.scale_2(x),
			size=[height, width],
			mode='bilinear') + x_list[1])))
		
		x_list.append(self.process_3((F.interpolate(
			self.scale_3(x),
			size=[height, width],
			mode='bilinear') + x_list[2])))

		x_list.append(self.process_4((F.interpolate(
			self.scale_4(x),
			size=[height, width],
			mode='bilinear') + x_list[3])))

		out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)

		return out


class SegmentationHead(nn.Module):
	def __init__(self, in_channels, inter_channels, out_channels, scale_factor=None):
		super(SegmentationHead, self).__init__()
	
		self.bn1 = nn.BatchNorm2d(in_channels, momentum=BATCH_NORM_MOM)
		self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(inter_channels, momentum=BATCH_NORM_MOM)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(inter_channels, out_channels, kernel_size=1, padding=0, bias=True)
		self.scale_factor = scale_factor
		
	def forward(self, x):
		x = self.conv1(self.relu(self.bn1(x)))
		out = self.conv2(self.relu(self.bn2(x)))

		if self.scale_factor is not None:
			height = x.shape[-2] * self.scale_factor
			width = x.shape[-1] * self.scale_factor
			out = F.interpolate(out, size=[height, width], mode='bilinear')
		return out


class DDRNet(nn.Module):
	def __init__(self, block, layers, num_classes=19, channels=64, spp_channels=128, head_channels=128, augment=False):
		super(DDRNet, self).__init__()
		
		highres_channel = channels * 2
		self.augment = augment
	
		# conv1, conv2 (half above)
		self.conv_1 = nn.Sequential(
	  		nn.Conv2d(3, channels, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(channels, momentum=BATCH_NORM_MOM),
			nn.ReLU(inplace=True),
			nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(channels, momentum=BATCH_NORM_MOM),
			nn.ReLU(inplace=True),
		)

		self.relu = nn.ReLU(inplace=True)

		# conv2 (half below), conv3, conv4, conv5 (half above)
		self.layer_1 = self.create_layer(block, channels, channels, layers[0])
		self.layer_2 = self.create_layer(block, channels, channels*2, layers[1], stride=2)
		self.layer_3 = self.create_layer(block, channels * 2, channels * 4, layers[2], stride=2)
		self.layer_4 = self.create_layer(block, channels * 4, channels * 8, layers[3], stride=2)
  
		self.compression_3 = nn.Sequential(
			nn.Conv2d(channels * 4, highres_channel, kernel_size=1, bias=False),
			nn.BatchNorm2d(highres_channel, momentum=BATCH_NORM_MOM),
		)
  
		self.compression_4 = nn.Sequential(
			nn.Conv2d(channels * 8, highres_channel, kernel_size=1, bias=False),
			nn.BatchNorm2d(highres_channel, momentum=BATCH_NORM_MOM),
		)
  
		self.down_3 = nn.Sequential(
			nn.Conv2d(highres_channel, channels * 4, kernel_size=3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(channels * 4, momentum=BATCH_NORM_MOM),
		)
  
		self.down_4 = nn.Sequential(
			nn.Conv2d(highres_channel, channels * 4, kernel_size=3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(channels * 4, momentum=BATCH_NORM_MOM),
			nn.ReLU(inplace=True),
			nn.Conv2d(channels * 4, channels * 8, kernel_size=3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(channels * 8, momentum=BATCH_NORM_MOM),
		)
  
		self.layer3_ = self.create_layer(block, channels * 2, highres_channel, 2)
		self.layer4_ = self.create_layer(block, highres_channel, highres_channel, 2)

		self.layer5_ = self.create_layer(BottleNeck, highres_channel, highres_channel, 1)
		self.layer_5 = self.create_layer(BottleNeck, channels * 8, channels * 8, 1, stride=2)
	
		self.spp = DAPPM(channels * 16, spp_channels, channels * 4)
  
		if self.augment:
			self.seghead_extra = SegmentationHead(highres_channel, head_channels, num_classes)

		self.final_layer = SegmentationHead(channels * 4, head_channels, num_classes)
  
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			if isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

  
	def create_layer(self, block, in_channels, out_channels, blocks, stride=1):
		downsample = None

		if stride != 1 or in_channels != out_channels * block.EXPANSION:
			downsample = nn.Sequential(
				nn.Conv2d(in_channels, out_channels * block.EXPANSION, 
			  			  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(out_channels * block.EXPANSION, momentum=BATCH_NORM_MOM),
			)
   
		layers = []
		layers.append(block(in_channels, out_channels, stride, downsample))
		in_channels = out_channels * block.EXPANSION
  
		for i in range(1, blocks):
			if i == (blocks - 1):
				layers.append(block(in_channels, out_channels, stride=1, no_relu=True))
			else:
				layers.append(block(in_channels, out_channels, stride=1, no_relu=False))
	
		return nn.Sequential(*layers)
		
	
	def forward(self, x):
		width_output = x.shape[-1] // 8
		height_output = x.shape[-2] // 8
  
		layers = []
		x = self.conv_1(x)

		x = self.layer_1(x)
		layers.append(x)
		
		x = self.layer_2(self.relu(x))
		layers.append(x)
		
		x = self.layer_3(self.relu(x))
		layers.append(x)
		
		x_ = self.layer3_(self.relu(layers[1]))
  
		x = x + self.down_3(self.relu(x_))
		x_ = x_ + F.interpolate(
			self.compression_3(self.relu(layers[2])),
			size=[height_output, width_output],
			mode='bilinear'
		)
  
		if self.augment:
			temp = x_
   
		x = self.layer_4(self.relu(x))
		layers.append(x)

		x_ = self.layer4_(self.relu(x_))

		x = x + self.down_4(self.relu(x_))
		x_ = x_ + F.interpolate(
			self.compression_4(self.relu(layers[3])),
			size=[height_output, width_output],
			mode='bilinear'
		)

		x_ = self.layer5_(self.relu(x_))
		x = F.interpolate(
			self.spp(self.layer_5(self.relu(x))),
			size=[height_output, width_output],
			mode='bilinear'
		)
	
		x_ = self.final_layer(x + x_)
  
		if self.augment:
			x_extra = self.seghead_extra(temp)
			return [x_, x_extra]

		return x_

def DDRNet_imagenet(pretrained=None):
	model = DDRNet(Block, 
				   [2, 2, 2, 2], 
				   num_classes=19, 
				   channels=64, 
				   spp_channels=128, 
				   head_channels=128, 
				   augment=False)
	
	if pretrained is not None:
		checkpoint = torch.load(pretrained, map_location='cpu')
		'''      
		new_state_dict = OrderedDict()
		for k, v in checkpoint['state_dict'].items():
			name = k[7:]  
			new_state_dict[name] = v
		#model_dict.update(new_state_dict)
		#model.load_state_dict(model_dict)
		'''
		model.load_state_dict(checkpoint)
	
	return model

if __name__ == '__main__':
	model = DDRNet_imagenet(pretrained='DDRNet23_imagenet.pth')


