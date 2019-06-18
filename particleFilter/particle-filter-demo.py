#demo code from  https://salzis.wordpress.com/2015/05/25/particle-filters-with-python/

from math import *
import random
import matplotlib.pyplot as plt

# landmarks which can be sensed by the robot (in meters)
landmarks = [[20.0, 20.0], [20.0, 80.0], [20.0, 50.0],
             [50.0, 20.0], [50.0, 80.0], [80.0, 80.0],
             [80.0, 20.0], [80.0, 50.0]]
 
# size of one dimension (in meters)
world_size = 100.0


class RobotClass:

	def __init__(self):
		self.x = random.random() * world_size
		self.y = random.random() * world_size
		self.orientation = random.random() *2.0*pi

		self.forward_noise = 0.0
		self.turn_noise = 0.0
		self.sense_noise = 0.0

	def set(self, new_x, new_y, new_orientation):
		if new_x <= 0 or new_x >=  world_size:
			raise ValueError('X coordinate out of bound')
		if new_y <= 0 or new_y >= world_size:
			raise ValueError('Y coordinate out of bound')
		if new_orientation <= 0 or new_orientation >= 2.0*pi:
			raise ValueError('orientation must bi in [0-2pi]')

		self.x = float(new_x)
		self.y = float(new_y)
		self.orientation = float(new_orientation)

	def set_noise(self, new_forward_noise, new_turn_noise, new_sense_noise):

		self.forward_noise = float(new_forward_noise)
		self.turn_noise = float(new_turn_noise)
		self.sense_noise = float(new_sense_noise)

	def sense(self):
		z = []
		for i in range(len(landmarks)):
			dist = sqrt((self.x - landmarks[i][0])**2+(self.y - landmarks[i][1])**2)
			dist += random.gauss(0.0, self.sense_noise)
			z.append(dist)
		return z

	def move(self, turn, forward):
		if forward <=0:
			raise ValueError('Robot cannot move backwards')

		orientation = self.orientation + float(turn) + random.gauss(0.0, self.turn_noise)
		orientation %= 2.0*pi

		dist = float(forward) + random.gauss(0.0, self.forward_noise)
		x = self.x + cos(orientation) * dist
		y = self.y + sin(orientation) * dist
		x %= world_size
		y %=  world_size
		
		res = RobotClass()
		res.set(x,y,orientation)
		res.set_noise(self.forward_noise, self.turn_noise, self.sense_noise)

		return res

	@staticmethod
	def gaussian(mu, sigma, x):
		return exp(-((mu-x)**2)/(sigma**2)/2.0)/sqrt(2.0*pi*(sigma**2))


	#calculate the probablity based on the distance between measurements and the sensors
	def measurement_prob(self, measurement):
		prob = 1.0
		for i in range(len(landmarks)):
			dist = sqrt((self.x - landmarks[i][0])**2+(self.y - landmarks[i][1])**2)
			prob *= self.gaussian(dist, self.sense_noise, measurement[i])
		return prob



def evaluation(r, p):
	sum = 0.0
	for i in range(len(p)):
		dx =(p[i].x - r.x + (world_size/2.0))%world_size - (world_size/2.0)
		dy =(p[i].y - r.y + (world_size/2.0))%world_size - (world_size/2.0)
		err = sqrt(dx**2 + dy**2)
		sum += err
	return sum/float(len(p))





def visualization(robot, step, p, pr):
	plt.figure()
	plt.title('Particle filter, step ' + str(step))
	grid  =[0, world_size, 0, world_size]
	plt.axis(grid)
	plt.grid(b=True, which='major', color='0.75', linestyle='--')
	plt.xticks([i for i in range(0, int(world_size),5)])
	plt.yticks([i for i in range(0, int(world_size),5)])


	for ind in range(len(p)):
		circle = plt.Circle((p[ind].x, p[ind].y), 1., facecolor='#ffb266', edgecolor='#994c00', alpha=0.5)
		plt.gca().add_patch(circle)
		arrow = plt.Arrow(p[ind].x, p[ind].y, 2*cos(p[ind].orientation), 2*sin(p[ind].orientation), alpha=1., facecolor='#994c00', edgecolor='#994c00')
		plt.gca().add_patch(arrow)

	for ind in range(len(pr)):

		# particle
		circle = plt.Circle((pr[ind].x, pr[ind].y), 1., facecolor='#66ff66', edgecolor='#009900', alpha=0.5)
		plt.gca().add_patch(circle)

		# particle's orientation
		arrow = plt.Arrow(pr[ind].x, pr[ind].y, 2*cos(pr[ind].orientation), 2*sin(pr[ind].orientation), alpha=1., facecolor='#006600', edgecolor='#006600')
		plt.gca().add_patch(arrow)

	for lm in landmarks:
		circle = plt.Circle((lm[0], lm[1]), 1., facecolor='#cc0000', edgecolor='#330000')
		plt.gca().add_patch(circle)

	circle = plt.Circle((robot.x, robot.y), 1., facecolor='#6666ff', edgecolor='#0000cc')
	plt.gca().add_patch(circle)

	# robot's orientation
	arrow = plt.Arrow(robot.x, robot.y, 2*cos(robot.orientation), 2*sin(robot.orientation), alpha=0.5, facecolor='#000000', edgecolor='#000000')
	plt.gca().add_patch(arrow)

	plt.savefig('figure_' + str(step) + '.png')
	plt.close()

# main script
MyRobot = RobotClass()
MyRobot = MyRobot.move(0.1, 5.0)

n = 1000  # number of particles
p = []    # list of particles
 
steps = 50

for i in range(n):
    x = RobotClass()
    x.set_noise(0.05, 0.05, 5.0)
    p.append(x)

for t in range(steps):
	MyRobot = MyRobot.move(0.1,5.)
	z = MyRobot.sense()

	p2 = []

	for i in range(n):
		p2.append(p[i].move(0.1,5.))

	p = p2
	w = []

	for i in range(n):
		w.append(p[i].measurement_prob(z))

	# resampling with a sample probability proportional
	# to the importance weight
	p3 = []
	 
	index = int(random.random() * n)
	beta = 0.0
	mw = max(w)
	 
	for i in range(n):
	    beta += random.random() * 2.0 * mw
	 
	    while beta >= w[index]:
	        beta -= w[index]
	        index = (index + 1) % n
	 
	    p3.append(p[index])
	 
	visualization(MyRobot, t, p, p3)
	# here we get a set of co-located particles
	p = p3
	
	print('Step = ', t, ', Evaluation = ', evaluation(MyRobot, p), 'len(p) = ', len(p)) 
