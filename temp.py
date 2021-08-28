class DAPPM(nn.Module):
	def __init__(self, in_channels, branch_channels, out_channels):
		super(DAPPM, self).__init__()
  
		self.scale1 = nn.Sequential(
			nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
			BatchNorm2d(in_channels, momentum=bn_mom),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False)
		)
  
		self.scale2 = nn.Sequential(
			nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
			BatchNorm2d(in_channels, momentum=bn_mom),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False)
		)
  
		self.scale3 = nn.Sequential(
			nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
			BatchNorm2d(in_channels, momentum=bn_mom),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False)
		)
  
		self.scale4 = nn.Sequential(
			nn.AdaptiveAvgPool2d((1, 1)),
			BatchNorm2d(in_channels, momentum=bn_mom),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False)
		)
  
		self.scale0 = nn.Sequential(
			BatchNorm2d(in_channels, momentum=bn_mom),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False)
		)
  
		self.process1 = nn.Sequential(
			BatchNorm2d(in_channels, momentum=bn_mom),
			nn.ReLU(inplace=True),
			nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1, bias=False)
		)
  
		self.process2 = nn.Sequential(
			BatchNorm2d(in_channels, momentum=bn_mom),
			nn.ReLU(inplace=True),
			nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1, bias=False)
		)
  
		self.process3 = nn.Sequential(
			BatchNorm2d(in_channels, momentum=bn_mom),
			nn.ReLU(inplace=True),
			nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1, bias=False)
		)
  
		self.process4 = nn.Sequential(
			BatchNorm2d(in_channels, momentum=bn_mom),
			nn.ReLU(inplace=True),
			nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1, bias=False)
		)
  
		self.compression = nn.Sequential(
			BatchNorm2d(branch_channels * 5, momentum=bn_mom),
			nn.ReLU(inplace=True),
			nn.Conv2d(branch_channels * 5, out_channels, kernel_size=1, bias=False)
		)
  
		self.shortcut = nn.Sequential(
			BatchNorm2d(in_channels, momentum=bn_mom),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
		)
  
	def forward(self, x):
		width = x.shape[-1]
		height = x.shape[-2]

		x_list = []
  
		x_list.append(self.scale0(x))
		x_list.append(self.process1((F.interpolate(
			self.scale1(x),
			size=[height, width],
			mode='bilinear') + x_list[0])))
		x_list.append(self.process2((F.interpolate(
			self.scale2(x),
			size=[height, width],
			mode='bilinear') + x_list[1])))
		x_list.append(self.process3((F.interpolate(
			self.scale3(x),
			size=[height, width],
			mode='bilinear') + x_list[2])))
		x_list.append(self.process4((F.interpolate(
			self.scale4(x),
			size=[height, width],
			mode='bilinear') + x_list[3])))
  
		out = self.compression(torch.cat(x_list, 1) + self.shortcut(x))
		return out