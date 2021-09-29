import torch

class accuracy(object):
	def __init__(self, nLabels=10):
		self.nLabels = nLabels

	def perLabel(self, x,y):
		totalPerLabel = [sum(y == torch.zeros_like(y).to(device))]
		correctPerLabel = [sum((y == x)*(y == torch.zeros_like(y).to(device)))]
		for i in range(1, self.nLabels):
			totalPerLabel.append(sum(y == i*torch.ones_like(y).to(device)))
			correctPerLabel.append(sum((y == x)*(y == i*torch.ones_like(y).to(device))))
		l = torch.Tensor(totalPerLabel)
		correct = torch.Tensor(correctPerLabel)
		r = torch.zeros(self.nLabels)
		for idx in range(len(l)):
			if l[idx] != 0:
				# print(idx, l[idx], correct[idx])
				r[idx] = correct[idx]/l[idx]
			else:
				r[idx] = 0.0
	#     print(r)
		return r, l

	def overAllLabel(self, x,y):
		totalPerLabel = [sum(y == torch.zeros_like(y).to(device))]
		correctPerLabel = [sum((y == x)*(y == torch.zeros_like(y).to(device)))]
		for i in range(1, self.nLabels):
			totalPerLabel.append(sum(y == i*torch.ones_like(y).to(device)))
			correctPerLabel.append(sum((y == x)*(y == i*torch.ones_like(y).to(device))))
		l = torch.Tensor(totalPerLabel)
		correct = torch.Tensor(correctPerLabel)
	#     print(r)
		return sum(correct)/sum(l), l

	def validationPerLabel(self, dataL, model):
		acc = torch.zeros(self.nLabels)
		ds = torch.zeros(self.nLabels)
		for i, data in enumerate(dataL, 0):
	#         r1.zero_grad()
			# Format batch
			x = data[0].to(device)
			y = data[1].to(device)
			output = model(x).max(1)[1]
			r, l = self.perLabel(output,y)
			acc = acc + r
			ds = ds + l
	#     print(i)
		return acc/(i+1)

	def validation(self, dataL, model):
		acc = 0
		ds = 0
		for i, data in enumerate(dataL, 0):
	#         r1.zero_grad()
			# Format batch
			x = data[0].to(device)
			y = data[1].to(device)
			output = model(x).max(1)[1]
			r, l = self.overAllLabel(output,y)
			acc = acc + r
			ds = ds + l
	#     print(i)
		return acc/(i+1)
