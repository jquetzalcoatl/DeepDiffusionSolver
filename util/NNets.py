import torch.nn as nn

class Reshape(nn.Module):
	def __init__(self, *args):
		super(Reshape, self).__init__()
		self.shape = args
	def forward(self, x):
		sh = (x.shape[0], ) + self.shape
		return x.view(sh)

class SimpleClas(nn.Module):
	def __init__(self):
		super(SimpleClas, self).__init__()
		self.seqIn = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1),
								nn.LeakyReLU(negative_slope=0.02),
								nn.BatchNorm2d(64),
								nn.Dropout2d(0.1),
								nn.AvgPool2d(2),

								nn.Conv2d(64,128,3,1,1),
								nn.LeakyReLU(negative_slope=0.02),
								nn.AvgPool2d(2),

								nn.Conv2d(128,256,3,1,1),
								nn.LeakyReLU(negative_slope=0.02),
								nn.AvgPool2d(2),

								nn.Conv2d(256,512,3,1,1),
								nn.LeakyReLU(negative_slope=0.02),
								nn.AvgPool2d(2),

								nn.Conv2d(512,1024,3,1,1),
								nn.LeakyReLU(negative_slope=0.02),
								nn.AvgPool2d(2),

								nn.Conv2d(1024,2048,3,1,0),
								nn.LeakyReLU(negative_slope=0.02),
								# nn.AvgPool2d(2), # USE for IMAGES 128x128
								nn.Conv2d(2048, 4096,2,1,0),  # USE for IMAGES 128x128

								nn.Flatten(),

								nn.Linear(4096, 512),
								nn.Linear(512, 10),
								nn.LogSoftmax(dim=1))#

	def forward(self, x):
		x = self.seqIn(x)
		return x

class SimpleCNN(nn.Module):
	def __init__(self):
		super(SimpleCNN, self).__init__()
		self.seqIn = nn.Sequential(nn.Conv2d(1,64,3,1,1),
							 nn.BatchNorm2d(64),
							 nn.LeakyReLU(negative_slope=0.02),
							 nn.AvgPool2d(2),

							 nn.Conv2d(64,128,3,1,1),
							 nn.LeakyReLU(negative_slope=0.02),
							 nn.AvgPool2d(2),

							 nn.Conv2d(128,256,3,1,1),
							 nn.LeakyReLU(negative_slope=0.02),
							 nn.AvgPool2d(2),

							 nn.Conv2d(256,512,3,1,1),
							 nn.LeakyReLU(negative_slope=0.02),
							 nn.AvgPool2d(2),

							 nn.Conv2d(512,1024,3,1,1),
							 nn.LeakyReLU(negative_slope=0.02),
							 nn.AvgPool2d(2),

							 nn.Conv2d(1024,2048,3,1,1),
							 nn.LeakyReLU(negative_slope=0.02),
							 # nn.AvgPool2d(2),
							 )

		self.seqOut = nn.Sequential(nn.ConvTranspose2d(2048,1024,3,1,1),
							  nn.LeakyReLU(negative_slope=0.02),

							  nn.ConvTranspose2d(1024,512,3,1,1),
							  nn.LeakyReLU(negative_slope=0.02),
							  nn.Upsample(scale_factor=2),

							  nn.ConvTranspose2d(512,256,3,1,1),
							  nn.LeakyReLU(negative_slope=0.02),
							  nn.Upsample(scale_factor=2),

							  nn.ConvTranspose2d(256,128,3,1,1),
							  nn.LeakyReLU(negative_slope=0.02),
							  nn.Upsample(scale_factor=2),

							  nn.ConvTranspose2d(128,64,3,1,1),
							  nn.LeakyReLU(negative_slope=0.02),
							  nn.Upsample(scale_factor=2),

							  nn.ConvTranspose2d(64,32,3,1,1),
							  nn.LeakyReLU(negative_slope=0.02),
							  nn.Upsample(scale_factor=2),

							  nn.ConvTranspose2d(32,1,3,1,1),
							  nn.LeakyReLU(negative_slope=0.02))
	def forward(self, x):
		x1 = self.seqIn(x)
		x1 = self.seqOut(x1)
		return x1

class DNN(nn.Module):
	def __init__(self, p1=1.0, p2=0.0, dout1=0.1, dout2=0.1):
		super(DNN, self).__init__()
		self.p1 = p1
		self.p2 = p2
		self.dout1 = dout1
		self.dout2 = dout2
		self.seqIn = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1),
								nn.LeakyReLU(negative_slope=0.02),
								nn.Dropout2d(self.dout1),
								nn.BatchNorm2d(64),
								nn.AvgPool2d(2),

								nn.Conv2d(64,128,3,1,1),
								nn.LeakyReLU(negative_slope=0.02),
								nn.AvgPool2d(2),
								# nn.LeakyReLU(negative_slope=0.02),

								nn.Conv2d(128,256,3,1,1),
								nn.LeakyReLU(negative_slope=0.02),
								nn.AvgPool2d(2),
								# nn.LeakyReLU(negative_slope=0.02),

								nn.Conv2d(256,512,3,1,1),
								nn.LeakyReLU(negative_slope=0.02),
								nn.AvgPool2d(2),
								# nn.LeakyReLU(negative_slope=0.02),

								nn.Conv2d(512,1024,3,1,1),
								nn.LeakyReLU(negative_slope=0.02),
								nn.AvgPool2d(2),
								# nn.LeakyReLU(negative_slope=0.02),

								nn.Conv2d(1024, 1024,3,1,1),
								nn.LeakyReLU(negative_slope=0.02),
								nn.AvgPool2d(2),
								# nn.LeakyReLU(negative_slope=0.02),

								nn.Conv2d(1024, 1024,3,1,1),
								nn.LeakyReLU(negative_slope=0.02),
								nn.AvgPool2d(2),
								# nn.LeakyReLU(negative_slope=0.02),

								nn.Conv2d(1024, 1024,3,1,1),
								nn.LeakyReLU(negative_slope=0.02),
								nn.AvgPool2d(4),
								# nn.LeakyReLU(negative_slope=0.02),

								nn.ConvTranspose2d(1024,1024,3,1,1),
								nn.LeakyReLU(negative_slope=0.02),
								nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
								# nn.LeakyReLU(negative_slope=0.02),

								nn.ConvTranspose2d(1024,1024,3,1,1),
								nn.LeakyReLU(negative_slope=0.02),
								nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
								# nn.LeakyReLU(negative_slope=0.02),

								nn.ConvTranspose2d(1024,1024,3,1,1),
								nn.LeakyReLU(negative_slope=0.02),
								nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
								# nn.LeakyReLU(negative_slope=0.02),

								nn.ConvTranspose2d(1024,512,3,1,1),
								nn.LeakyReLU(negative_slope=0.02),
								nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
								# nn.LeakyReLU(negative_slope=0.02),

								nn.ConvTranspose2d(512,256,3,1,1),
								nn.LeakyReLU(negative_slope=0.02),
								nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
								# nn.LeakyReLU(negative_slope=0.02),

								nn.ConvTranspose2d(256,128,3,1,1),
								nn.LeakyReLU(negative_slope=0.02),
								nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
								# nn.LeakyReLU(negative_slope=0.02),

								nn.ConvTranspose2d(128,64,3,1,1),
								nn.LeakyReLU(negative_slope=0.02),
								nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
								# nn.LeakyReLU(negative_slope=0.02),

								nn.Dropout2d(self.dout2),
								nn.ConvTranspose2d(64,1,3,1,1),
								nn.LeakyReLU(negative_slope=0.0),
								nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
								# nn.ReLU(),
								# nn.BatchNorm2d(1),

								)

	def forward(self, x):
		x = self.p1 * self.seqIn(x)
		return x

class UNet(nn.Module):
	def __init__(self):
		super(UNet, self).__init__()
		self.blk1 = nn.Sequential(
								nn.Conv2d(1, 64, 3, 1, 1),
								nn.LeakyReLU(negative_slope=0.0),

								nn.Conv2d(64, 64, 3, 1, 1),
								nn.LeakyReLU(negative_slope=0.0),
		)
		self.blk2 = nn.Sequential(
								nn.Conv2d(64, 128, 3, 1, 1),
								nn.LeakyReLU(negative_slope=0.0),

								nn.Conv2d(128, 128, 3, 1, 1),
								nn.LeakyReLU(negative_slope=0.0),
		)
		self.blk3 = nn.Sequential(
								nn.Conv2d(128, 256, 3, 1, 1),
								nn.LeakyReLU(negative_slope=0.0),

								nn.Conv2d(256, 256, 3, 1, 1),
								nn.LeakyReLU(negative_slope=0.0),
		)
		self.blk4 = nn.Sequential(
								nn.Conv2d(256, 512, 3, 1, 1),
								nn.LeakyReLU(negative_slope=0.0),

								nn.Conv2d(512, 512, 3, 1, 1),
								nn.LeakyReLU(negative_slope=0.0),
		)
		self.blk5 = nn.Sequential(
								nn.Conv2d(512, 1024, 3, 1, 1),
								nn.LeakyReLU(negative_slope=0.0),

								nn.Conv2d(1024, 1024, 3, 1, 1),
								nn.LeakyReLU(negative_slope=0.0),
		)
		self.blkUp1 = nn.Sequential(
								nn.Conv2d(1024, 512, 3, 1, 1),
								nn.LeakyReLU(negative_slope=0.0),

								nn.Conv2d(512, 512, 3, 1, 1),
								nn.LeakyReLU(negative_slope=0.0),
		)
		self.blkUp2 = nn.Sequential(
								nn.Conv2d(512, 256, 3, 1, 1),
								nn.LeakyReLU(negative_slope=0.0),

								nn.Conv2d(256, 256, 3, 1, 1),
								nn.LeakyReLU(negative_slope=0.0),
		)
		self.blkUp3 = nn.Sequential(
								nn.Conv2d(256, 128, 3, 1, 1),
								nn.LeakyReLU(negative_slope=0.0),

								nn.Conv2d(128, 128, 3, 1, 1),
								nn.LeakyReLU(negative_slope=0.0),
		)
		self.blkUp4 = nn.Sequential(
								nn.Conv2d(128, 64, 3, 1, 1),
								nn.LeakyReLU(negative_slope=0.0),

								nn.Conv2d(64, 64, 3, 1, 1),
								nn.LeakyReLU(negative_slope=0.0),
		)
		self.upConv1 = nn.Sequential(
								nn.Upsample(scale_factor=2),
								nn.ConvTranspose2d(1024,512,3,1,1),
		)
		self.upConv2 = nn.Sequential(
								nn.Upsample(scale_factor=2),
								nn.ConvTranspose2d(512,256,3,1,1),
		)
		self.upConv3 = nn.Sequential(
								nn.Upsample(scale_factor=2),
								nn.ConvTranspose2d(256,128,3,1,1),
		)
		self.upConv4 = nn.Sequential(
								nn.Upsample(scale_factor=2),
								nn.ConvTranspose2d(128,64,3,1,1),
		)

	def forward(self, x):
		x1 = self.blk1(x)
		x2 = self.blk2(nn.MaxPool2d(2, stride=2)(x1))
		x3 = self.blk3(nn.MaxPool2d(2, stride=2)(x2))
		x4 = self.blk4(nn.MaxPool2d(2, stride=2)(x3))
		x5 = self.blk5(nn.MaxPool2d(2, stride=2)(x4))

		x6 = self.blkUp1(torch.cat((self.upConv1(x5), x4 ), dim=1))
		x7 = self.blkUp2(torch.cat((self.upConv2(x6), x3 ), dim=1))
		x8 = self.blkUp3(torch.cat((self.upConv3(x7), x2 ), dim=1))
		x9 = self.blkUp4(torch.cat((self.upConv4(x8), x1 ), dim=1))
		xfinal = nn.ConvTranspose2d(64,1,3,1,1)(x9)

		return xfinal


class SimpleDisc(nn.Module):
	def __init__(self):
		super(Disc, self).__init__()
		self.seqIn = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1),
								nn.LeakyReLU(negative_slope=0.02),
								nn.BatchNorm2d(64),
								nn.Dropout2d(0.1),
								nn.AvgPool2d(2),

								nn.Conv2d(64,128,3,1,1),
								nn.LeakyReLU(negative_slope=0.02),
								nn.AvgPool2d(2),

								nn.Conv2d(128,256,3,1,1),
								nn.LeakyReLU(negative_slope=0.02),
								nn.AvgPool2d(2),

								nn.Conv2d(256,512,3,1,1),
								nn.LeakyReLU(negative_slope=0.02),
								nn.AvgPool2d(2),

								nn.Conv2d(512,1024,3,1,1),
								nn.LeakyReLU(negative_slope=0.02),
								nn.AvgPool2d(2),

								nn.Conv2d(1024,2048,3,1,0),
								nn.LeakyReLU(negative_slope=0.02),

								nn.Flatten(),

								nn.Linear(2048, 512),
								nn.Linear(512, 1),
								nn.Sigmoid())#

	def forward(self, x):
		x = self.seqIn(x)
		return x


class SimpleGen(nn.Module):
	def __init__(self):
		super(Gen, self).__init__()
		self.seqIn = nn.Sequential(nn.Linear(100,160000),
								nn.BatchNorm1d(160000),
								nn.ReLU(),
								Reshape(256,25,25),
								nn.ConvTranspose2d(256,128,5,2,1),
								nn.BatchNorm2d(128),
								nn.ReLU(),
								nn.ConvTranspose2d(128,64,4,1,2),
								nn.BatchNorm2d(64),
								nn.ReLU(),
								nn.ConvTranspose2d(64,1,4,2,1),
								nn.Hardtanh()
								)#

	def forward(self, x):
		x = self.seqIn(x)
		return x